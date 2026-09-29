"""Durable status and saved-model lifecycle helpers for generic training runs."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Callable

try:
    from learning_adaptation.model_catalogue import (
        ModelCatalogueStore,
        ModelNotFoundError,
        ModelRecordError,
    )
    from learning_adaptation.training_run_storage import TrainingRunStore
except ModuleNotFoundError:  # Service image copies modules beside tasks.py.
    from model_catalogue import ModelCatalogueStore, ModelNotFoundError, ModelRecordError
    from training_run_storage import TrainingRunStore


TERMINAL_CHILD_STATUSES = {
    "completed",
    "failed",
    "dispatch_failed",
    "validation_failed",
}
TERMINAL_PARENT_STATUSES = {"completed", "partial_success", "failed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_value(value: Any, maximum: int = 4096) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return value[:1000]
    if isinstance(value, dict):
        try:
            encoded = json.dumps(value)
        except (TypeError, ValueError):
            return str(value)[:1000]
        return deepcopy(value) if len(encoded) <= maximum else "result metadata omitted because it is too large"
    return str(value)[:1000]


def safe_failure_summary(value: Any) -> str | None:
    if value is None:
        return None
    message = str(value).replace("\n", " ").replace("\r", " ")[:1000]
    message = re.sub(r"\b(?:https?|redis)://\S+", "[service URL redacted]", message)
    message = re.sub(
        r"(?<![A-Za-z0-9_.-])/(?:[^\s/:]+/)+[^\s:]+",
        "[path redacted]",
        message,
    )
    return message


def evaluation_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    omitted = {
        "binary_predictions",
        "anomaly_scores",
        "predictions",
        "scores",
        "raw_predictions",
        "raw_scores",
    }
    summary: dict[str, Any] = {}
    for key, item in value.items():
        if key in omitted:
            continue
        if isinstance(item, str):
            summary[str(key)] = safe_failure_summary(item)
        elif item is None or isinstance(item, (int, float, bool)):
            summary[str(key)] = item
        elif isinstance(item, dict):
            nested = evaluation_summary(item)
            if nested:
                summary[str(key)] = nested
    try:
        if len(json.dumps(summary)) > 16_384:
            return {"summary": "evaluation metadata omitted because it is too large"}
    except (TypeError, ValueError):
        return {}
    return summary


def task_result_summary(value: Any) -> Any:
    """Keep only stable, compact task-result fields in run metadata."""
    if not isinstance(value, dict):
        return compact_value(value)
    allowed = {
        "status",
        "detector_id",
        "model_id",
        "training_parameters",
        "n_features",
        "evaluation",
        "promotion",
    }
    summary = {key: deepcopy(value[key]) for key in allowed if key in value}
    if "evaluation" in summary:
        summary["evaluation"] = evaluation_summary(summary["evaluation"])
    compact = compact_value(summary)
    return compact if isinstance(compact, dict) else {"summary": compact}


def calculate_parent_status(children: list[dict[str, Any]]) -> str:
    statuses = [child.get("status", "queued") for child in children]
    if statuses and all(status == "completed" for status in statuses):
        return "completed"
    if statuses and all(status in {"failed", "dispatch_failed", "validation_failed"} for status in statuses):
        return "failed"
    if "validation_failed" in statuses and any(
        status not in {"validation_failed", "failed", "dispatch_failed"}
        for status in statuses
    ):
        return "partial_success"
    if statuses and all(status in TERMINAL_CHILD_STATUSES for status in statuses):
        return "partial_success"
    if statuses and all(status == "queued" for status in statuses):
        return "queued"
    return "running"


def normalize_run_record(run: dict[str, Any]) -> dict[str, Any]:
    run = deepcopy(run)
    created_at = run.get("created_at") or utc_now()
    run.setdefault("created_at", created_at)
    run.setdefault("started_at", None)
    run.setdefault("updated_at", created_at)
    run.setdefault("completed_at", None)
    for child in run.get("children", []):
        child.setdefault("created_at", created_at)
        child.setdefault("started_at", None)
        child.setdefault("updated_at", child["created_at"])
        child.setdefault("completed_at", None)
        child.setdefault("validation_error", None)
        child.setdefault("failure_summary", child.get("dispatch_error"))
        child.setdefault("result_metadata", None)
        child.setdefault("model_status", None)
    run["status"] = calculate_parent_status(run.get("children", []))
    return _refresh_parent_timestamps(run, run.get("updated_at") or created_at)


def _refresh_parent_timestamps(run: dict[str, Any], now: str | None = None) -> dict[str, Any]:
    now = now or utc_now()
    children = run.get("children", [])
    run["status"] = calculate_parent_status(children)
    run["child_summary"] = {
        status: sum(1 for child in children if child.get("status") == status)
        for status in (
            "queued",
            "running",
            "completed",
            "failed",
            "dispatch_failed",
            "validation_failed",
        )
    }
    starts = [item.get("started_at") for item in children if item.get("started_at")]
    if starts and not run.get("started_at"):
        run["started_at"] = min(starts)
    run["updated_at"] = now
    if run["status"] in TERMINAL_PARENT_STATUSES and all(
        child.get("status") in TERMINAL_CHILD_STATUSES for child in children
    ):
        run["completed_at"] = run.get("completed_at") or now
    elif run["status"] not in TERMINAL_PARENT_STATUSES:
        run["completed_at"] = None
    return run


def update_child(
    store: TrainingRunStore,
    run_id: str,
    detector_id: str,
    updater: Callable[[dict[str, Any], str], None],
) -> dict[str, Any]:
    def mutate(stored: dict[str, Any]) -> dict[str, Any]:
        run = normalize_run_record(stored)
        now = utc_now()
        child = next(
            (item for item in run.get("children", []) if item.get("detector_id") == detector_id),
            None,
        )
        if child is None:
            return run
        updater(child, now)
        child["updated_at"] = now
        return _refresh_parent_timestamps(run, now)

    return store.update(run_id, mutate)


def configured_stores() -> tuple[TrainingRunStore, ModelCatalogueStore]:
    trained_root = Path(os.getenv("TRAINED_MODELS_TEMP_ROOT", "trained_models_temp"))
    shared_root = trained_root.parent
    run_root = Path(os.getenv("TRAINING_RUN_STORAGE_ROOT", shared_root / "training_runs"))
    model_root = Path(os.getenv("MODEL_CATALOGUE_STORAGE_ROOT", shared_root / "models"))
    return TrainingRunStore(run_root), ModelCatalogueStore(model_root)


def mark_child_running(context: dict[str, Any] | None) -> None:
    if not context:
        return
    run_store, _model_store = configured_stores()

    def mutate(child: dict[str, Any], now: str) -> None:
        if child.get("status") in TERMINAL_CHILD_STATUSES:
            return
        child["status"] = "running"
        child["started_at"] = child.get("started_at") or now

    update_child(run_store, context["training_run_id"], context["detector_id"], mutate)


def mark_child_failed(context: dict[str, Any] | None, error: Any) -> None:
    if not context:
        return
    run_store, _model_store = configured_stores()

    def mutate(child: dict[str, Any], now: str) -> None:
        if child.get("status") == "completed":
            return
        child["status"] = "failed"
        child["started_at"] = child.get("started_at") or now
        child["completed_at"] = child.get("completed_at") or now
        child["failure_summary"] = safe_failure_summary(error or "Training task failed")
        child["result_metadata"] = None
        child["model_status"] = None

    update_child(run_store, context["training_run_id"], context["detector_id"], mutate)


def record_successful_model(
    context: dict[str, Any] | None,
    *,
    evaluation: dict[str, Any],
    promotion: dict[str, Any],
) -> dict[str, Any] | None:
    if not context:
        return None
    run_store, model_store = configured_stores()
    now = utc_now()
    evaluation = evaluation_summary(evaluation)
    promotion_status = promotion.get("status", "not_applicable")
    if promotion_status == "available":
        promotion_status = "promoted"
    record = {
        "schema_version": 1,
        "model_id": context["model_id"],
        "detector_id": context["detector_id"],
        "detector_version": context["detector_version"],
        "status": "available",
        "created_at": now,
        "updated_at": now,
        "training_run_id": context["training_run_id"],
        "training_child_id": context.get("training_child_id"),
        "dataset": deepcopy(context["dataset"]),
        "snapshot": deepcopy(context["snapshot"]),
        "feature_identity": deepcopy(context["feature_identity"]),
        "training_parameters": deepcopy(context["training_parameters"]),
        "evaluation": deepcopy(evaluation),
        "promotion": {
            "status": promotion_status,
            "promoted_at": now if promotion_status == "promoted" else None,
            "safe_reference": context["model_id"] if promotion_status == "promoted" else None,
        },
    }
    try:
        existing = model_store.read(record["model_id"])
    except ModelNotFoundError:
        model_store.write(record)
    else:
        identity_fields = (
            "model_id",
            "detector_id",
            "detector_version",
            "training_run_id",
            "training_child_id",
            "dataset",
            "snapshot",
            "feature_identity",
            "training_parameters",
        )
        if any(existing.get(field) != record.get(field) for field in identity_fields):
            raise ModelRecordError(
                f"model record {record['model_id']!r} already belongs to different training provenance"
            )
        # A task retry after the catalogue write must reuse the stable record.
        # Keep its original timestamps and any later promotion update.
        record = existing

    def mutate(child: dict[str, Any], completed_at: str) -> None:
        child["status"] = "completed"
        child["started_at"] = child.get("started_at") or completed_at
        child["completed_at"] = child.get("completed_at") or completed_at
        child["failure_summary"] = None
        child["result_metadata"] = {
            "model_id": record["model_id"],
            "evaluation": compact_value(evaluation),
            "promotion": deepcopy(record["promotion"]),
        }
        child["model_status"] = record["status"]

    update_child(run_store, context["training_run_id"], context["detector_id"], mutate)
    return record


def mark_cgnn_promoted(model_id: str) -> None:
    run_store, model_store = configured_stores()
    try:
        record = model_store.read(model_id)
    except ModelNotFoundError:
        return
    now = utc_now()
    record["updated_at"] = now
    record["promotion"] = {
        "status": "promoted",
        "promoted_at": now,
        "safe_reference": model_id,
    }
    model_store.replace(record)

    def mutate(child: dict[str, Any], _updated_at: str) -> None:
        if child.get("model_id") != model_id:
            return
        result = child.get("result_metadata")
        if not isinstance(result, dict):
            result = {"model_id": model_id}
            child["result_metadata"] = result
        result["promotion"] = deepcopy(record["promotion"])

    try:
        update_child(
            run_store,
            record["training_run_id"],
            record["detector_id"],
            mutate,
        )
    except Exception:
        # The confirmed model promotion remains authoritative even when an old
        # or externally removed run record can no longer be updated.
        return


def normalize_celery_status(state: str) -> str:
    state = (state or "PENDING").upper()
    if state in {"SUCCESS", "COMPLETED"}:
        return "completed"
    if state in {"FAILURE", "REVOKED"}:
        return "failed"
    if state == "PENDING":
        return "queued"
    return "running"


def reconcile_run(
    store: TrainingRunStore,
    run_id: str,
    status_reader: Callable[[str], Any],
) -> dict[str, Any]:
    """Persist trustworthy Celery observations without downgrading terminal state."""

    def mutate(stored: dict[str, Any]) -> dict[str, Any]:
        run = normalize_run_record(stored)
        now = utc_now()
        changed = False
        for child in run.get("children", []):
            if child.get("status") in TERMINAL_CHILD_STATUSES or not child.get("task_id"):
                continue
            try:
                task = status_reader(child["task_id"])
                observed = normalize_celery_status(getattr(task, "state", "PENDING"))
                # PENDING is ambiguous after Redis result expiry. Never downgrade a
                # child that was already observed as running.
                if observed == "queued" and child.get("status") == "running":
                    continue
                previous_status = child.get("status")
                child["status"] = observed
                if observed != previous_status:
                    changed = True
                    child["updated_at"] = now
                if observed == "running":
                    child["started_at"] = child.get("started_at") or now
                raw_detail = getattr(task, "info", None)
                if observed == "completed":
                    child["started_at"] = child.get("started_at") or now
                    child["completed_at"] = child.get("completed_at") or now
                    child["result_metadata"] = task_result_summary(raw_detail)
                    child["failure_summary"] = None
                elif observed == "failed":
                    child["started_at"] = child.get("started_at") or now
                    child["completed_at"] = child.get("completed_at") or now
                    child["failure_summary"] = safe_failure_summary(
                        raw_detail or "Training task failed"
                    )
                    child["result_metadata"] = None
                elif raw_detail is not None:
                    child["detail"] = compact_value(raw_detail)
            except Exception as exc:
                status_detail = (
                    "Training status is temporarily unavailable: "
                    + (safe_failure_summary(exc) or "unknown error")
                )
                if child.get("status_detail") != status_detail:
                    child["status_detail"] = status_detail
                    changed = True
        return _refresh_parent_timestamps(
            run, now if changed else run.get("updated_at") or now
        )

    return store.update(run_id, mutate)


def history_summary(run: dict[str, Any]) -> dict[str, Any]:
    run = normalize_run_record(run)
    children = []
    for child in run.get("children", []):
        children.append(
            {
                "detector_id": child.get("detector_id"),
                "model_id": child.get("model_id"),
                "status": child.get("status"),
                "created_at": child.get("created_at"),
                "started_at": child.get("started_at"),
                "updated_at": child.get("updated_at"),
                "completed_at": child.get("completed_at"),
                "validation_error": safe_failure_summary(child.get("validation_error")),
                "failure_summary": safe_failure_summary(
                    child.get("failure_summary") or child.get("dispatch_error")
                ),
                "model_status": child.get("model_status"),
            }
        )
    dataset = run.get("dataset", {})
    safe_dataset = {
        key: dataset.get(key)
        for key in ("source", "dataset_id", "version", "partition_id")
        if dataset.get(key) is not None
    }
    return {
        "run_id": run.get("run_id"),
        "status": run.get("status"),
        "created_at": run.get("created_at"),
        "started_at": run.get("started_at"),
        "updated_at": run.get("updated_at"),
        "completed_at": run.get("completed_at"),
        "dataset": safe_dataset,
        "snapshot_id": run.get("snapshot_id"),
        "children": children,
    }


def model_summary(record: dict[str, Any]) -> dict[str, Any]:
    dataset = record["dataset"]
    return {
        "model_id": record["model_id"],
        "detector_id": record["detector_id"],
        "detector_version": record["detector_version"],
        "status": record["status"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "training_run_id": record["training_run_id"],
        "dataset": {
            key: dataset.get(key)
            for key in ("source", "dataset_id", "version", "partition_id")
            if dataset.get(key) is not None
        },
        "promotion": {"status": record["promotion"]["status"]},
    }


def public_model_record(record: dict[str, Any]) -> dict[str, Any]:
    """Project a storage record onto the stable, path-free public contract."""
    return {
        "schema_version": record["schema_version"],
        "model_id": record["model_id"],
        "detector_id": record["detector_id"],
        "detector_version": record["detector_version"],
        "status": record["status"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "training_run_id": record["training_run_id"],
        "training_child_id": record.get("training_child_id"),
        "dataset": deepcopy(record["dataset"]),
        "snapshot": deepcopy(record["snapshot"]),
        "feature_identity": deepcopy(record["feature_identity"]),
        "training_parameters": deepcopy(record["training_parameters"]),
        "evaluation": evaluation_summary(record["evaluation"]),
        "promotion": deepcopy(record["promotion"]),
    }
