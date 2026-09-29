"""Manifest-driven execution core for generic detector TrainingRun children."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable

from detectors import registry
from detectors.contracts import (
    ArtifactResult,
    DetectorCompatibilityError,
    DetectorExecutionError,
    DetectorPromotionError,
    DetectorTrainingError,
    ProgressReporter,
    TrainingContext,
    TrainingData,
    TrainingResult,
)

try:
    from learning_adaptation.training_lifecycle import (
        mark_child_failed,
        mark_child_running,
        mark_child_validation_failed,
        evaluation_summary,
        record_successful_model,
        safe_failure_summary,
    )
    from learning_adaptation.training_run_storage import (
        TrainingRunStore,
        TrainingRunValidationError,
        sha256_file,
        validate_identifier,
    )
except ModuleNotFoundError:  # Service image copies modules into /app.
    from training_lifecycle import (
        mark_child_failed,
        mark_child_running,
        mark_child_validation_failed,
        evaluation_summary,
        record_successful_model,
        safe_failure_summary,
    )
    from training_run_storage import (
        TrainingRunStore,
        TrainingRunValidationError,
        sha256_file,
        validate_identifier,
    )


TRAINING_PROTOCOL = "cloudsentinel.training/v1"
INPUT_PROFILE = "metrics-partition-v1"
PROMOTION_STATUSES = {
    "not_promoted",
    "promoted",
    "promotion_failed",
    "not_applicable",
}


def _safe_model_metadata(value: Any, depth: int = 0) -> Any:
    """Bound and redact adapter-provided metadata before durable publication."""

    if depth > 5:
        return "metadata depth limit reached"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return safe_failure_summary(value)
    if isinstance(value, (list, tuple)):
        return [_safe_model_metadata(item, depth + 1) for item in list(value)[:200]]
    if isinstance(value, dict):
        return {
            str(key)[:100]: _safe_model_metadata(item, depth + 1)
            for key, item in list(value.items())[:200]
        }
    return str(value)[:1000]


def _safe_child_path(root: Path, value: str, field: str) -> Path:
    identifier = validate_identifier(value, field)
    path = (root / identifier).resolve()
    if not path.is_relative_to(root):
        raise DetectorExecutionError(f"{field} resolves outside configured storage")
    return path


def _csv_shape(path: Path) -> tuple[int, int]:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.reader(source))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DetectorExecutionError(f"snapshot CSV cannot be read: {exc}") from exc
    if not rows or any(not row for row in rows):
        raise DetectorExecutionError("snapshot CSV is empty or malformed")
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        raise DetectorExecutionError("snapshot CSV is not rectangular")
    return len(rows), width


def _load_verified_snapshot(snapshot_id: str) -> tuple[dict[str, Any], Path]:
    root = Path(
        os.getenv("DATASET_SNAPSHOT_STORAGE_ROOT", "dataset_snapshots")
    ).resolve()
    directory = _safe_child_path(root, snapshot_id, "snapshot_id")
    metadata_path = directory / "snapshot.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DetectorExecutionError("immutable snapshot metadata is unavailable") from exc
    if not isinstance(metadata, dict) or metadata.get("snapshot_id") != snapshot_id:
        raise DetectorExecutionError("immutable snapshot identity mismatch")
    files = metadata.get("files")
    if not isinstance(files, dict):
        raise DetectorExecutionError("immutable snapshot file metadata is invalid")
    hashes = []
    for logical_name in ("train", "test", "labels"):
        record = files.get(logical_name)
        if not isinstance(record, dict):
            raise DetectorExecutionError(
                f"{INPUT_PROFILE} snapshot is missing {logical_name}"
            )
        filename = record.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise DetectorExecutionError("snapshot contains an unsafe filename")
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or path.is_symlink():
            raise DetectorExecutionError(f"snapshot {logical_name} artifact is unavailable")
        actual = sha256_file(path)
        if actual != record.get("sha256"):
            raise DetectorExecutionError(f"snapshot {logical_name} checksum mismatch")
        hashes.append(actual)
    combined = hashlib.sha256("".join(hashes).encode("ascii")).hexdigest()
    if combined != metadata.get("sha256"):
        raise DetectorExecutionError("snapshot aggregate checksum mismatch")
    return metadata, directory


def _feature_identity(snapshot: dict[str, Any]) -> dict[str, Any]:
    provenance = snapshot.get("catalogue_provenance") or {}
    feature_order = provenance.get("feature_order")
    feature_hash = provenance.get("feature_order_sha256")
    if not feature_order:
        details = snapshot.get("dataset_details", {})
        containers = details.get("containers", [])
        metrics = details.get("metrics", [])
        feature_order = [
            f"{container}_{metric}" for container in containers for metric in metrics
        ]
        if feature_order:
            feature_hash = hashlib.sha256(
                json.dumps(
                    feature_order, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
    return {
        "feature_order": feature_order or [],
        "feature_order_sha256": feature_hash,
    }


def _build_context(payload: dict[str, Any]) -> TrainingContext:
    if not isinstance(payload, dict):
        raise DetectorExecutionError("training context payload must be an object")
    run_id = validate_identifier(payload.get("run_id"), "run_id")
    detector_id = validate_identifier(payload.get("detector_id"), "detector_id")
    model_id = validate_identifier(payload.get("model_id"), "model_id")
    snapshot_id = validate_identifier(payload.get("snapshot_id"), "snapshot_id")

    run_root = Path(os.getenv("TRAINING_RUN_STORAGE_ROOT", "training_runs"))
    run = TrainingRunStore(run_root).read(run_id)
    child = next(
        (
            item
            for item in run.get("children", [])
            if item.get("detector_id") == detector_id
        ),
        None,
    )
    if child is None or child.get("model_id") != model_id:
        raise DetectorExecutionError("training child identity mismatch")
    if run.get("snapshot_id") != snapshot_id or child.get("snapshot_id") != snapshot_id:
        raise DetectorExecutionError("training child snapshot mismatch")

    snapshot, directory = _load_verified_snapshot(snapshot_id)
    manifest = registry.get_manifest(detector_id)
    training = manifest["capabilities"]["training"]
    if (
        not training.get("enabled")
        or training.get("protocol") != TRAINING_PROTOCOL
        or training.get("input_profile") != INPUT_PROFILE
    ):
        raise DetectorExecutionError("detector training capability changed after dispatch")
    parameters = registry.validate_training_parameters(
        detector_id, child.get("parameters"), manifest
    )

    train_record = snapshot["files"]["train"]
    test_record = snapshot["files"]["test"]
    labels_record = snapshot["files"]["labels"]
    train_path = directory / train_record["filename"]
    test_path = directory / test_record["filename"]
    labels_path = directory / labels_record["filename"]
    train_count, train_width = _csv_shape(train_path)
    test_count, test_width = _csv_shape(test_path)
    label_count, label_width = _csv_shape(labels_path)
    if train_width != test_width or label_width != 1 or label_count != test_count:
        raise DetectorExecutionError(f"{INPUT_PROFILE} snapshot shapes are inconsistent")

    artifact_root = Path(
        os.getenv("TRAINED_MODELS_TEMP_ROOT", "trained_models_temp")
    ).resolve()
    suggested = (artifact_root / "plugins" / detector_id / model_id).resolve()
    if not suggested.is_relative_to(artifact_root):
        raise DetectorExecutionError("artifact workspace resolves outside configured storage")

    dataset = dict(run.get("dataset", {}))
    provenance = snapshot.get("catalogue_provenance") or {}
    model_context = {
        "training_run_id": run_id,
        "training_child_id": child.get("child_id") or f"{run_id}-{detector_id}",
        "model_id": model_id,
        "detector_id": detector_id,
        "detector_version": manifest["version"],
        "dataset": {
            **dataset,
            **(
                {"partition_checksum": provenance.get("partition_checksum")}
                if dataset.get("source") == "catalogue"
                else {}
            ),
        },
        "snapshot": {
            "snapshot_id": snapshot_id,
            "snapshot_sha256": snapshot["sha256"],
        },
        "feature_identity": _feature_identity(snapshot),
        "training_parameters": parameters,
    }
    return TrainingContext(
        run_id=run_id,
        child_id=model_context["training_child_id"],
        model_id=model_id,
        detector_id=detector_id,
        detector_version=manifest["version"],
        parameters=parameters,
        dataset=dataset,
        snapshot={**snapshot, "dataset_details": snapshot.get("dataset_details", {})},
        feature_identity=model_context["feature_identity"],
        data=TrainingData(
            train_path=train_path,
            test_path=test_path,
            labels_path=labels_path,
            train_observations=train_count,
            test_observations=test_count,
            feature_count=train_width,
        ),
        artifact_workspace=artifact_root,
        suggested_artifact_directory=suggested,
        model_record_context=model_context,
    )


def _validate_result(context: TrainingContext, result: Any) -> TrainingResult:
    if not isinstance(result, TrainingResult) or result.status != "completed":
        raise DetectorExecutionError("training adapter returned an invalid TrainingResult")
    if not isinstance(result.artifact, ArtifactResult):
        raise DetectorExecutionError("training adapter returned invalid artifact metadata")
    directory = result.artifact.directory.resolve()
    workspace = context.artifact_workspace.resolve()
    if not directory.is_relative_to(workspace) or directory == workspace:
        raise DetectorExecutionError("training artifact is outside its configured workspace")
    if not directory.is_dir() or directory.is_symlink():
        raise DetectorExecutionError("training artifact directory is unavailable")
    if not result.artifact.required_files:
        raise DetectorExecutionError("training artifact declares no required files")
    if len(result.artifact.required_files) > 100:
        raise DetectorExecutionError("training artifact declares too many required files")
    if not isinstance(result.artifact.format, str) or not result.artifact.format.strip():
        raise DetectorExecutionError("training artifact format is invalid")
    for filename in result.artifact.required_files:
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise DetectorExecutionError("training artifact filename is unsafe")
        path = directory / filename
        if not path.is_file() or path.is_symlink():
            raise DetectorExecutionError(
                f"training artifact is missing required file {filename!r}"
            )
    if result.promotion.status not in PROMOTION_STATUSES:
        raise DetectorExecutionError("training result has invalid promotion status")
    if result.cleanup not in {"keep", "remove_after_catalogue"}:
        raise DetectorExecutionError("training result has invalid cleanup policy")
    return result


def execute_detector_plugin_training(
    context_payload: dict[str, Any],
    progress_callback: Callable[[str, Any], None],
) -> dict[str, Any]:
    """Execute one manifest-selected adapter and persist its durable outcome."""
    lifecycle_context = (
        {
            "training_run_id": context_payload.get("run_id"),
            "detector_id": context_payload.get("detector_id"),
        }
        if isinstance(context_payload, dict)
        else None
    )
    try:
        context = _build_context(context_payload)
        mark_child_running(lifecycle_context)
        progress = ProgressReporter(progress_callback)
        progress.report("VALIDATING", "Validating detector compatibility")
        adapter = registry.get_training_adapter(context.detector_id)
        try:
            adapter.validate_training(context)
        except DetectorCompatibilityError:
            raise
        except Exception as exc:
            raise DetectorCompatibilityError(str(exc)) from exc
        try:
            result = adapter.run_training(context, progress)
        except DetectorTrainingError:
            raise
        except Exception as exc:
            raise DetectorExecutionError(str(exc)) from exc
        result = _validate_result(context, result)
        promotion = {
            "status": result.promotion.status,
            "promoted_at": result.promotion.promoted_at,
            "safe_reference": result.promotion.safe_reference,
        }
        compact_evaluation = evaluation_summary(dict(result.evaluation))
        record_successful_model(
            dict(context.model_record_context),
            evaluation=compact_evaluation,
            promotion=promotion,
            artifact={
                "format": result.artifact.format,
                "safe_reference": result.artifact.safe_reference,
            },
            model_metadata=_safe_model_metadata(dict(result.model_metadata)),
        )
        if result.cleanup == "remove_after_catalogue":
            try:
                shutil.rmtree(result.artifact.directory)
            except OSError:
                pass
        progress.report("COMPLETED", "Detector training completed")
        return {
            "status": "completed",
            "detector_id": context.detector_id,
            "model_id": context.model_id,
            "evaluation": compact_evaluation,
            "promotion": promotion,
        }
    except DetectorCompatibilityError as exc:
        mark_child_validation_failed(lifecycle_context, exc)
        raise
    except DetectorTrainingError as exc:
        mark_child_failed(lifecycle_context, exc)
        raise
    except Exception as exc:
        wrapped = DetectorExecutionError(
            safe_failure_summary(exc) or "detector plugin execution failed"
        )
        mark_child_failed(lifecycle_context, wrapped)
        raise wrapped from exc
