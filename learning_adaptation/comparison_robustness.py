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
ROBUSTNESS_WORKLOAD_FILTERS = {"Low", "Normal", "High", "Variable"}
ROBUSTNESS_SCENARIO_FILTERS = {
    "Normal operation",
    "CPU stress",
    "Memory stress",
    "Network delay",
    "Unknown",
    "Other",
}


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
    scenario_detail = workload.get("anomaly_scenario_detail")
    if not isinstance(scenario_detail, str) or not scenario_detail.strip():
        scenario_detail = None
    else:
        scenario_detail = scenario_detail.strip()[:160]
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
            **{
                key: workload.get(key)
                for key in (
                    "workload_intensity",
                    "dominant_workload_characteristic",
                    "anomaly_scenario",
                )
                if workload.get(key) is not None
            },
            **(
                {"anomaly_scenario_detail": scenario_detail}
                if scenario_detail is not None
                else {}
            ),
        },
        "scenario": workload.get("anomaly_scenario"),
        "scenario_detail": scenario_detail,
        "scenario_display": (
            f"Other — {scenario_detail}"
            if workload.get("anomaly_scenario") == "Other"
            and scenario_detail
            else workload.get("anomaly_scenario")
        ),
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
        "range": max(values) - min(values),
    }
    if standard_deviation:
        summary["standard_deviation"] = pstdev(values) if len(values) >= 2 else None
    return summary


def _context_description(context: dict[str, Any]) -> str:
    workload = context.get("workload_context") or {}
    parts = [
        workload.get("workload_intensity"),
        context.get("scenario_display") or context.get("scenario"),
    ]
    value = " / ".join(str(part) for part in parts if part)
    return value or str(context.get("display_name") or "this evaluation context")


def _model_insight(rows: list[dict[str, Any]], statistics: dict[str, Any]) -> dict[str, Any]:
    ground_truth_count = sum(1 for row in rows if row.get("ground_truth_available"))
    labelled_rows = [
        row for row in rows if _finite_number(row.get("metrics", {}).get("f1_score")) is not None
    ]
    worst = min(
        labelled_rows,
        key=lambda row: (
            float(row["metrics"]["f1_score"]),
            str(row["context"].get("partition_checksum") or ""),
            str(row.get("comparison_id") or ""),
        ),
        default=None,
    )
    context_count = len(rows)
    usable_f1_count = len(labelled_rows)
    f1 = statistics["f1_score"]
    if context_count == 0:
        status = "no_contexts"
        rationale = "No completed comparisons were found for this exact saved model artefact."
    elif context_count == 1:
        status = "insufficient_contexts"
        rationale = (
            "One evaluation context is available. At least two distinct contexts are "
            "needed to observe performance variation."
        )
    elif usable_f1_count == 0:
        status = "metrics_unavailable"
        rationale = (
            f"Evaluated in {context_count} contexts, of which {ground_truth_count} have "
            "ground truth, but none contains a usable F1 value."
        )
    elif usable_f1_count == 1:
        status = "insufficient_labelled_contexts"
        rationale = (
            f"Evaluated in {context_count} contexts, of which {ground_truth_count} have "
            "ground truth. Only one contains a usable F1 value; at least two are needed "
            "to observe F1 variation."
        )
    else:
        status = "variation_available"
        rationale = (
            f"Evaluated in {context_count} contexts, of which {ground_truth_count} have ground truth. "
            f"F1 ranged from {f1['minimum']:.3f} to {f1['maximum']:.3f}. "
            f"Performance was lowest under {_context_description(worst['context'])}."
        )
    return {
        "status": status,
        "context_count": context_count,
        "labelled_context_count": ground_truth_count,
        "usable_f1_context_count": usable_f1_count,
        "rationale": rationale,
        "worst_labelled_context": (
            {
                "comparison_id": worst.get("comparison_id"),
                "context": deepcopy(worst.get("context")),
                "f1_score": worst["metrics"]["f1_score"],
            }
            if worst is not None
            else None
        ),
    }


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
    statistics = {
        "f1_score": _series(f1, standard_deviation=True),
        "precision": _series(precision),
        "recall": _series(recall),
        "runtime_ms": _series(runtimes),
        "lead_time_seconds": _series(lead_times),
    }
    return {
        "model_id": model.get("model_id"),
        "display_name": model.get("display_name") or model.get("model_id"),
        "detector_id": model.get("detector_id"),
        "detector_version": model.get("detector_version"),
        "artifact_identity": _safe_artifact_identity(model),
        "context_count": len(rows),
        "labelled_context_count": len(labelled),
        "sufficient_contexts": len(rows) >= 2,
        "statistics": statistics,
        "insight": _model_insight(rows, statistics),
        "contexts": visible_rows,
        "related_comparison_ids": sorted(
            {row["comparison_id"] for row in visible_rows if row.get("comparison_id")}
        ),
    }


def _shared_evidence(
    identities: dict[tuple[str, str, str], dict[str, Any]],
    rows_by_identity: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Compare all selected artifacts only on the exact same labelled contexts."""

    ordered = list(identities)
    if len(ordered) < 2:
        return {
            "status": "insufficient",
            "context_count": 0,
            "model_count": len(ordered),
            "models": [],
            "lowest_variation_models": [],
            "tie": False,
            "rationale": "At least two immutable model artefacts are required.",
        }
    indexed = {
        identity: {
            row["context"]["partition_checksum"]: row
            for row in rows_by_identity.get(identity, [])
        }
        for identity in ordered
    }
    common = set(indexed[ordered[0]])
    for identity in ordered[1:]:
        common &= set(indexed[identity])
    comparable = []
    for checksum in sorted(common):
        rows = [indexed[identity][checksum] for identity in ordered]
        if all(
            row.get("ground_truth_available")
            and _finite_number(row.get("metrics", {}).get("f1_score")) is not None
            for row in rows
        ):
            comparable.append((checksum, rows))

    wins = {identity: 0 for identity in ordered}
    outright = {identity: 0 for identity in ordered}
    tied = {identity: 0 for identity in ordered}
    for _checksum, rows in comparable:
        values = [float(row["metrics"]["f1_score"]) for row in rows]
        best = max(values)
        winner_indexes = [index for index, value in enumerate(values) if value == best]
        for index in winner_indexes:
            identity = ordered[index]
            wins[identity] += 1
            if len(winner_indexes) == 1:
                outright[identity] += 1
            else:
                tied[identity] += 1

    models = []
    for index, identity in enumerate(ordered):
        values = [float(rows[index]["metrics"]["f1_score"]) for _checksum, rows in comparable]
        model = identities[identity]
        models.append(
            {
                "model_id": model.get("model_id"),
                "display_name": model.get("display_name") or model.get("model_id"),
                "detector_id": model.get("detector_id"),
                "detector_version": model.get("detector_version"),
                "artifact_identity": deepcopy(model.get("artifact_identity") or {}),
                "context_count": len(values),
                "f1_score": _series(values, standard_deviation=True),
                "highest_f1_context_count": wins[identity],
                "outright_highest_f1_context_count": outright[identity],
                "tied_highest_f1_context_count": tied[identity],
            }
        )

    sufficient = len(comparable) >= 2
    lowest = []
    tie = False
    if sufficient:
        ranges = [model["f1_score"]["range"] for model in models]
        best_range = min(ranges)
        lowest = [
            {"model_id": model["model_id"], "display_name": model["display_name"]}
            for model in models
            if model["f1_score"]["range"] == best_range
        ]
        tie = len(lowest) > 1
        if tie:
            rationale = (
                f"{len(lowest)} models share the lowest observed F1 variation across "
                f"{len(comparable)} shared labelled contexts."
            )
        else:
            rationale = (
                f"{lowest[0]['display_name']} has the lowest observed F1 variation across "
                f"{len(comparable)} shared labelled contexts."
            )
    else:
        rationale = (
            "The models have not yet been evaluated on at least two of the same labelled "
            "contexts, so their performance stability cannot be compared fairly."
        )
    return {
        "status": "sufficient" if sufficient else "insufficient",
        "context_count": len(comparable),
        "model_count": len(ordered),
        "models": models,
        "lowest_variation_models": lowest,
        "tie": tie,
        "rationale": rationale,
        "contexts": [deepcopy(rows[0]["context"]) for _checksum, rows in comparable],
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

    shared_evidence = _shared_evidence(identities, rows_by_identity)

    return {
        "comparison_id": current.get("comparison_id"),
        "aggregation_version": ROBUSTNESS_VERSION,
        "definition": (
            "Variation in performance of the same immutable Saved Model artifact "
            "across stored independent evaluation contexts."
        ),
        "filters": deepcopy(filters),
        "models": aggregates,
        "model_insights": [
            {
                "model_id": model["model_id"],
                "display_name": model["display_name"],
                "artifact_identity": deepcopy(model["artifact_identity"]),
                **deepcopy(model["insight"]),
            }
            for model in aggregates
        ],
        "shared_evidence": shared_evidence,
        "shared_context_matrix": shared_matrix,
        "coverage_warnings": warnings,
        "eligible_comparison_count": eligible_run_count,
        "deduplicated_repeat_count": repeated_result_count,
        "skipped_corrupt_records": skipped_corrupt_records,
        "truncated": truncated,
    }
