"""Historical robustness aggregation for immutable Saved Model artifacts."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
from statistics import median, pstdev
from typing import Any

try:
    from learning_adaptation.comparison_storage import ComparisonStore
except ModuleNotFoundError:
    from comparison_storage import ComparisonStore


ROBUSTNESS_VERSION = "cloudsentinel.comparison-robustness/v1"
TERMINAL_RUN_STATUSES = {"completed", "partial"}


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _artifact_identity(model: dict[str, Any]) -> tuple[str, str, str] | None:
    identity = model.get("artifact_identity") or {}
    values = (
        model.get("model_id"),
        identity.get("artifact_id"),
        identity.get("artifact_manifest_sha256"),
    )
    if not all(isinstance(value, str) and value for value in values):
        return None
    return values


def _safe_artifact_identity(model: dict[str, Any]) -> dict[str, Any]:
    identity = model.get("artifact_identity") or {}
    return {
        key: identity.get(key)
        for key in (
            "artifact_id",
            "artifact_manifest_sha256",
            "contract",
            "artifact_format",
        )
        if identity.get(key) is not None
    }


def _context(record: dict[str, Any]) -> dict[str, Any] | None:
    dataset = record.get("evaluation_dataset") or {}
    checksum = dataset.get("partition_checksum")
    if not isinstance(checksum, str) or not checksum:
        return None
    workload = dataset.get("workload_context") or {}
    incidents = record.get("known_incident_windows")
    if not isinstance(incidents, list):
        incident = record.get("known_incident_window")
        incidents = [incident] if isinstance(incident, dict) else []
    safe_incidents = [
        {
            key: incident.get(key)
            for key in ("incident_id", "start_time", "end_time", "scenario")
            if incident.get(key) is not None
        }
        for incident in incidents
        if isinstance(incident, dict)
    ]
    return {
        "dataset_id": dataset.get("dataset_id"),
        "display_name": dataset.get("display_name") or dataset.get("dataset_id"),
        "version": dataset.get("version"),
        "partition_id": dataset.get("partition_id"),
        "partition_checksum": checksum,
        "modality": record.get("modality") or dataset.get("modality"),
        "workload_context": {
            key: workload.get(key)
            for key in (
                "workload_intensity",
                "dominant_workload_characteristic",
                "anomaly_scenario",
            )
            if workload.get(key) is not None
        },
        "scenario": workload.get("anomaly_scenario"),
        "dominant_characteristic": workload.get(
            "dominant_workload_characteristic"
        ),
        "ground_truth_available": bool(
            record.get("ground_truth_available", dataset.get("ground_truth_available"))
        ),
        "known_incident_windows": safe_incidents,
    }


def _result_row(
    record: dict[str, Any],
    selected: dict[str, Any],
    result: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    labelled = bool(result.get("ground_truth_available"))
    raw_metrics = result.get("metrics") if labelled else {}
    raw_metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
    metrics = {
        key: value
        for key in ("precision", "recall", "f1_score")
        if (value := _finite_number(raw_metrics.get(key))) is not None
    }
    lead_time = None
    if result.get("lead_time_status") == "before_or_at_incident_start":
        lead_time = _finite_number(result.get("lead_time_seconds"))
    return {
        "comparison_id": record.get("comparison_id"),
        "comparison_name": record.get("name") or record.get("comparison_id"),
        "comparison_date": record.get("completed_at")
        or record.get("updated_at")
        or record.get("created_at"),
        "model_id": selected.get("model_id"),
        "display_name": selected.get("display_name") or selected.get("model_id"),
        "detector_id": selected.get("detector_id"),
        "detector_version": selected.get("detector_version"),
        "artifact_identity": _safe_artifact_identity(selected),
        "context": deepcopy(context),
        "ground_truth_available": labelled,
        "metrics": metrics,
        "runtime_ms": _finite_number(result.get("runtime_ms")),
        "lead_time_seconds": lead_time,
        "lead_time_status": result.get("lead_time_status") or "unavailable",
    }


def _series(values: list[float], *, standard_deviation: bool = False) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "minimum": None, "maximum": None}
    summary: dict[str, Any] = {
        "count": len(values),
        "median": median(values),
        "minimum": min(values),
        "maximum": max(values),
    }
    if standard_deviation:
        summary["range"] = max(values) - min(values)
        summary["standard_deviation"] = pstdev(values) if len(values) >= 2 else None
    return summary


def _aggregate(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    visible_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    labelled = [row for row in rows if row["ground_truth_available"]]
    f1 = [row["metrics"]["f1_score"] for row in labelled if "f1_score" in row["metrics"]]
    precision = [row["metrics"]["precision"] for row in labelled if "precision" in row["metrics"]]
    recall = [row["metrics"]["recall"] for row in labelled if "recall" in row["metrics"]]
    runtimes = [row["runtime_ms"] for row in rows if row["runtime_ms"] is not None]
    lead_times = [
        row["lead_time_seconds"]
        for row in labelled
        if row["lead_time_seconds"] is not None
    ]
    return {
        "model_id": model.get("model_id"),
        "display_name": model.get("display_name") or model.get("model_id"),
        "detector_id": model.get("detector_id"),
        "detector_version": model.get("detector_version"),
        "artifact_identity": _safe_artifact_identity(model),
        "context_count": len(rows),
        "labelled_context_count": len(labelled),
        "sufficient_contexts": len(rows) >= 2,
        "statistics": {
            "f1_score": _series(f1, standard_deviation=True),
            "precision": _series(precision),
            "recall": _series(recall),
            "runtime_ms": _series(runtimes),
            "lead_time_seconds": _series(lead_times),
        },
        "contexts": visible_rows,
        "related_comparison_ids": sorted(
            {row["comparison_id"] for row in visible_rows if row.get("comparison_id")}
        ),
    }


def _matches_filters(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    context = row["context"]
    workload = context.get("workload_context") or {}
    requested_workload = filters.get("workload")
    if requested_workload and requested_workload not in workload.values():
        return False
    if filters.get("scenario") and context.get("scenario") != filters["scenario"]:
        return False
    if filters.get("labelled_only") and not row["ground_truth_available"]:
        return False
    return True


def _terminal_order(record: dict[str, Any]) -> tuple[float, str]:
    value = record.get("completed_at") or record.get("updated_at") or record.get("created_at")
    timestamp = float("-inf")
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            timestamp = parsed.astimezone(timezone.utc).timestamp()
        except ValueError:
            pass
    return timestamp, str(record.get("comparison_id") or "")


def aggregate_robustness(
    store: ComparisonStore,
    current: dict[str, Any],
    *,
    filters: dict[str, Any] | None = None,
    maximum_contexts_per_model: int = 200,
) -> dict[str, Any]:
    """Aggregate latest terminal results per exact artifact and partition checksum."""

    filters = filters or {}
    identities: dict[tuple[str, str, str], dict[str, Any]] = {}
    for model in current.get("selected_models", []):
        identity = _artifact_identity(model)
        if identity is not None and identity not in identities:
            safe_model = {
                key: deepcopy(model.get(key))
                for key in (
                    "model_id",
                    "display_name",
                    "detector_id",
                    "detector_version",
                )
            }
            safe_model["artifact_identity"] = _safe_artifact_identity(model)
            identities[identity] = safe_model

    canonical: dict[
        tuple[tuple[str, str, str], str],
        tuple[tuple[float, str], dict[str, Any]],
    ] = {}
    eligible_run_count = 0
    skipped_corrupt_records = 0
    repeated_result_count = 0
    for comparison_id in store.identifiers():
        try:
            record = store.read(comparison_id)
        except Exception:
            skipped_corrupt_records += 1
            continue
        if record.get("status") not in TERMINAL_RUN_STATUSES:
            continue
        context = _context(record)
        if context is None:
            continue
        eligible_run_count += 1
        selected_by_model = {
            item.get("model_id"): item
            for item in record.get("selected_models", [])
            if isinstance(item, dict)
        }
        terminal_key = _terminal_order(record)
        for result in record.get("results", []):
            if not isinstance(result, dict) or result.get("status") != "completed":
                continue
            selected = selected_by_model.get(result.get("model_id"))
            if not isinstance(selected, dict):
                continue
            identity = _artifact_identity(selected)
            if identity not in identities:
                continue
            key = (identity, context["partition_checksum"])
            row = _result_row(record, selected, result, context)
            previous = canonical.get(key)
            if previous is not None:
                repeated_result_count += 1
            if previous is None or terminal_key > previous[0]:
                canonical[key] = (terminal_key, row)

    rows_by_identity: dict[tuple[str, str, str], list[dict[str, Any]]] = {
        identity: [] for identity in identities
    }
    for (identity, _checksum), (_terminal, row) in canonical.items():
        if _matches_filters(row, filters):
            rows_by_identity[identity].append(row)

    if filters.get("shared_only"):
        coverage: dict[str, int] = {}
        for rows in rows_by_identity.values():
            for row in rows:
                checksum = row["context"]["partition_checksum"]
                coverage[checksum] = coverage.get(checksum, 0) + 1
        shared = {checksum for checksum, count in coverage.items() if count >= 2}
        for identity, rows in rows_by_identity.items():
            rows_by_identity[identity] = [
                row for row in rows if row["context"]["partition_checksum"] in shared
            ]

    truncated = False
    aggregates = []
    for identity, model in identities.items():
        rows = sorted(
            rows_by_identity[identity],
            key=lambda row: _terminal_order(
                {
                    "completed_at": row.get("comparison_date"),
                    "comparison_id": row.get("comparison_id"),
                }
            ),
            reverse=True,
        )
        visible_rows = rows
        if len(rows) > maximum_contexts_per_model:
            visible_rows = rows[:maximum_contexts_per_model]
            truncated = True
        aggregates.append(_aggregate(model, rows, visible_rows))

    aggregate_by_identity = {
        _artifact_identity(model): model for model in aggregates
    }
    shared_matrix = []
    all_checksums = sorted(
        {
            row["context"]["partition_checksum"]
            for model in aggregates
            for row in model["contexts"]
        }
    )
    for checksum in all_checksums:
        entries = []
        context = None
        for identity in identities:
            aggregate = aggregate_by_identity.get(identity)
            if aggregate is None:
                continue
            row = next(
                (
                    item
                    for item in aggregate["contexts"]
                    if item["context"]["partition_checksum"] == checksum
                ),
                None,
            )
            if row is not None:
                context = context or row["context"]
                entries.append(
                    {
                        "model_id": row["model_id"],
                        "display_name": row["display_name"],
                        "comparison_id": row["comparison_id"],
                        "metrics": row["metrics"],
                        "runtime_ms": row["runtime_ms"],
                        "lead_time_seconds": row["lead_time_seconds"],
                        "lead_time_status": row["lead_time_status"],
                    }
                )
        if len(entries) >= 2:
            shared_matrix.append({"context": context, "models": entries})

    counts = [model["context_count"] for model in aggregates]
    labelled_counts = [model["labelled_context_count"] for model in aggregates]
    warnings = []
    if len(set(counts)) > 1 or len(set(labelled_counts)) > 1:
        details = "; ".join(
            f"{model['display_name']} has {model['labelled_context_count']} labelled contexts"
            for model in aggregates
        )
        warnings.append(f"Comparison coverage differs: {details}.")
    if repeated_result_count:
        warnings.append("Showing the latest completed result per evaluation context.")

    return {
        "comparison_id": current.get("comparison_id"),
        "aggregation_version": ROBUSTNESS_VERSION,
        "definition": (
            "Variation in performance of the same immutable Saved Model artifact "
            "across stored independent evaluation contexts."
        ),
        "filters": deepcopy(filters),
        "models": aggregates,
        "shared_context_matrix": shared_matrix,
        "coverage_warnings": warnings,
        "eligible_comparison_count": eligible_run_count,
        "deduplicated_repeat_count": repeated_result_count,
        "skipped_corrupt_records": skipped_corrupt_records,
        "truncated": truncated,
    }
