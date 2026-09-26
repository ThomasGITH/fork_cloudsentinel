"""HTTP API and orchestration service for generic multi-detector training runs."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable
import uuid

from flask import Blueprint, current_app, jsonify, request
from werkzeug.exceptions import BadRequest

from detectors import registry
from detectors.isolation_forest.training_request import (
    IsolationForestRequestError,
    validate_training_parameters,
)

try:
    from learning_adaptation.catalogue_training import (
        CatalogueBundleError,
        configured_client,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {
        "learning_adaptation",
        "learning_adaptation.catalogue_training",
    }:
        raise
    from catalogue_training import CatalogueBundleError, configured_client

try:
    from learning_adaptation.training_launchers import build_training_launchers
    from learning_adaptation.training_run_storage import (
        DatasetSnapshotStore,
        TrainingRunNotFoundError,
        TrainingRunStore,
        TrainingRunValidationError,
        new_identifier,
        validate_identifier,
    )
except ModuleNotFoundError as exc:  # The service image copies modules into /app.
    if exc.name not in {
        "learning_adaptation",
        "learning_adaptation.training_launchers",
        "learning_adaptation.training_run_storage",
    }:
        raise
    from training_launchers import build_training_launchers
    from training_run_storage import (
        DatasetSnapshotStore,
        TrainingRunNotFoundError,
        TrainingRunStore,
        TrainingRunValidationError,
        new_identifier,
        validate_identifier,
    )


TERMINAL_CHILD_STATUSES = {
    "completed",
    "failed",
    "dispatch_failed",
    "validation_failed",
}


def _parameter_matches_type(value: Any, parameter_type: str) -> bool:
    if parameter_type == "boolean":
        return isinstance(value, bool)
    if parameter_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if parameter_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, str)


def validate_manifest_parameters(
    detector_id: str,
    submitted: Any,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if submitted is None:
        submitted = {}
    if not isinstance(submitted, dict):
        raise TrainingRunValidationError(
            f"parameters for detector {detector_id!r} must be an object"
        )
    definitions = manifest["training_parameters"]
    unknown = sorted(set(submitted).difference(definitions))
    if unknown:
        raise TrainingRunValidationError(
            f"unknown training parameter(s) for {detector_id!r}: {', '.join(unknown)}"
        )

    values = {name: metadata["default"] for name, metadata in definitions.items()}
    values.update(submitted)
    for name, value in values.items():
        metadata = definitions[name]
        if value is None and metadata.get("nullable", False):
            continue
        if not _parameter_matches_type(value, metadata["type"]):
            raise TrainingRunValidationError(
                f"parameter {name!r} for {detector_id!r} must be {metadata['type']}"
            )
        if "minimum" in metadata and value < metadata["minimum"]:
            raise TrainingRunValidationError(
                f"parameter {name!r} for {detector_id!r} must be at least {metadata['minimum']}"
            )
        if "maximum" in metadata and value > metadata["maximum"]:
            raise TrainingRunValidationError(
                f"parameter {name!r} for {detector_id!r} must be at most {metadata['maximum']}"
            )
    return values


def validate_request_payload(
    payload: Any,
    launchers: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    if not isinstance(payload, dict):
        raise TrainingRunValidationError("request body must be a JSON object")
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise TrainingRunValidationError("dataset must be an object")
    source_type = dataset.get("source")
    if source_type not in {"existing", "catalogue"}:
        raise TrainingRunValidationError(
            "dataset.source must be 'existing' or 'catalogue'"
        )
    dataset_id = validate_identifier(dataset.get("dataset_id"), "dataset.dataset_id")
    if source_type == "existing":
        allowed_dataset_fields = {"source", "dataset_id"}
        normalized_dataset = {"source": "existing", "dataset_id": dataset_id}
    else:
        allowed_dataset_fields = {"source", "dataset_id", "version", "partition_id"}
        version = dataset.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise TrainingRunValidationError("dataset.version must be a positive integer")
        partition_id = validate_identifier(
            dataset.get("partition_id"), "dataset.partition_id"
        )
        normalized_dataset = {
            "source": "catalogue",
            "dataset_id": dataset_id,
            "version": version,
            "partition_id": partition_id,
        }
    unknown_dataset_fields = sorted(set(dataset) - allowed_dataset_fields)
    if unknown_dataset_fields:
        raise TrainingRunValidationError(
            "unsupported dataset field(s): " + ", ".join(unknown_dataset_fields)
        )

    detector_requests = payload.get("detectors")
    if not isinstance(detector_requests, list) or not detector_requests:
        raise TrainingRunValidationError("detectors must be a non-empty list")

    manifests = {item["id"]: item for item in registry.discover_detectors()}
    validated = []
    seen = set()
    for index, detector_request in enumerate(detector_requests):
        if not isinstance(detector_request, dict):
            raise TrainingRunValidationError(f"detectors[{index}] must be an object")
        detector_id = detector_request.get("detector_id")
        if not isinstance(detector_id, str) or detector_id not in manifests:
            raise TrainingRunValidationError(f"unknown detector_id: {detector_id!r}")
        if detector_id in seen:
            raise TrainingRunValidationError(
                f"duplicate detector_id: {detector_id!r}"
            )
        seen.add(detector_id)
        manifest = manifests[detector_id]
        if "training" not in manifest.get("input_requirements", {}):
            raise TrainingRunValidationError(
                f"detector {detector_id!r} does not declare training capability"
            )
        if detector_id not in launchers:
            raise TrainingRunValidationError(
                f"detector {detector_id!r} has no training launcher"
            )
        if detector_id == "isolation-forest":
            try:
                parameters = validate_training_parameters(
                    detector_request.get("parameters")
                )
            except IsolationForestRequestError as exc:
                raise TrainingRunValidationError(str(exc)) from exc
        else:
            parameters = validate_manifest_parameters(
                detector_id, detector_request.get("parameters"), manifest
            )
        validated.append({"detector_id": detector_id, "parameters": parameters})

    client_request_id = payload.get("client_request_id")
    if client_request_id is not None and (
        not isinstance(client_request_id, str)
        or not client_request_id.strip()
        or len(client_request_id) > 200
    ):
        raise TrainingRunValidationError(
            "client_request_id must be a non-empty string of at most 200 characters"
        )
    return normalized_dataset, validated, client_request_id


def _compact_task_info(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)[:1000] if isinstance(value, str) else value
    if isinstance(value, dict):
        try:
            encoded = json.dumps(value)
        except (TypeError, ValueError):
            return str(value)[:1000]
        return value if len(encoded) <= 4096 else "task detail omitted because it is too large"
    return str(value)[:1000]


def normalize_celery_status(state: str) -> str:
    state = (state or "PENDING").upper()
    if state in {"SUCCESS", "COMPLETED"}:
        return "completed"
    if state in {"FAILURE", "REVOKED"}:
        return "failed"
    if state == "PENDING":
        return "queued"
    return "running"


def calculate_parent_status(children: list[dict[str, Any]]) -> str:
    statuses = [child["status"] for child in children]
    if statuses and all(status == "completed" for status in statuses):
        return "completed"
    if statuses and all(status in {"failed", "dispatch_failed"} for status in statuses):
        return "failed"
    if statuses and all(status in TERMINAL_CHILD_STATUSES for status in statuses):
        return "partial_success"
    if "validation_failed" in statuses and any(
        status not in {"validation_failed", "failed", "dispatch_failed"}
        for status in statuses
    ):
        return "partial_success"
    if statuses and all(status == "queued" for status in statuses):
        return "queued"
    return "running"


def create_training_runs_blueprint(
    cgnn_task: Any,
    isolation_forest_task: Any,
    status_reader: Callable[[str], Any],
) -> Blueprint:
    blueprint = Blueprint("training_runs", __name__)
    launchers = build_training_launchers(cgnn_task, isolation_forest_task)

    def stores() -> tuple[DatasetSnapshotStore, TrainingRunStore]:
        snapshot_root = current_app.config["DATASET_SNAPSHOT_STORAGE_ROOT"]
        dataset_root = current_app.config["EXISTING_DATASETS_ROOT"]
        run_root = current_app.config["TRAINING_RUN_STORAGE_ROOT"]
        return (
            DatasetSnapshotStore(snapshot_root, dataset_root),
            TrainingRunStore(run_root),
        )

    @blueprint.post("/training_runs")
    def create_training_run():
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415
        try:
            payload = request.get_json(silent=False)
            dataset, detector_requests, client_request_id = validate_request_payload(
                payload, launchers
            )
            snapshot_store, run_store = stores()
        except (BadRequest, TrainingRunValidationError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

        run_id = new_identifier("run")
        snapshot = None
        downloaded_source = None
        prepared_children = []
        try:
            if dataset["source"] == "existing":
                source = snapshot_store.validate_source(dataset["dataset_id"])
                snapshot = snapshot_store.create(dataset["dataset_id"], source)
            else:
                fetcher = current_app.config.get("CATALOGUE_BUNDLE_FETCHER")
                if fetcher is None:
                    downloaded_source = configured_client(dict(current_app.config)).download(
                        dataset
                    )
                else:
                    downloaded_source = fetcher(dataset)
                snapshot = snapshot_store.create_catalogue(downloaded_source)
        except (TrainingRunValidationError, CatalogueBundleError, ValueError) as exc:
            if downloaded_source is not None:
                temporary_root = downloaded_source.get("temporary_root")
                if temporary_root:
                    import shutil

                    shutil.rmtree(temporary_root, ignore_errors=True)
            return jsonify({"error": str(exc)}), 400
        except Exception:
            if downloaded_source is not None:
                temporary_root = downloaded_source.get("temporary_root")
                if temporary_root:
                    import shutil

                    shutil.rmtree(temporary_root, ignore_errors=True)
            raise
        finally:
            if downloaded_source is not None:
                temporary_root = downloaded_source.get("temporary_root")
                if temporary_root:
                    import shutil

                    shutil.rmtree(temporary_root, ignore_errors=True)

        child_records = []
        try:
            for detector_request in detector_requests:
                detector_id = detector_request["detector_id"]
                model_id = (
                    f"{run_id}-{detector_id}-{uuid.uuid4().hex[:8]}"
                )
                child_input_dir = run_store.child_input_directory(run_id, detector_id)
                record = {
                    "detector_id": detector_id,
                    "model_id": model_id,
                    "task_id": None,
                    "status": "queued",
                    "parameters": detector_request["parameters"],
                    "snapshot_id": snapshot["snapshot_id"],
                    "validation_error": None,
                    "dispatch_error": None,
                }
                try:
                    prepared = launchers[detector_id].prepare(
                        snapshot,
                        child_input_dir,
                        detector_request["parameters"],
                        {
                            "run_id": run_id,
                            "model_id": model_id,
                            "dataset_id": dataset["dataset_id"],
                        },
                    )
                    prepared_children.append(
                        {**record, "prepared": prepared, "record_index": len(child_records)}
                    )
                except TrainingRunValidationError as exc:
                    if dataset["source"] == "existing":
                        raise
                    record["status"] = "validation_failed"
                    record["validation_error"] = str(exc)[:1000]
                child_records.append(record)
        except TrainingRunValidationError as exc:
            run_store.remove(run_id)
            if snapshot is not None:
                snapshot_store.remove(snapshot["snapshot_id"])
            return jsonify({"error": str(exc)}), 400
        except Exception:
            run_store.remove(run_id)
            if snapshot is not None:
                snapshot_store.remove(snapshot["snapshot_id"])
            raise

        created_at = datetime.now(timezone.utc).isoformat()
        run = {
            "schema_version": 2 if dataset["source"] == "catalogue" else 1,
            "run_id": run_id,
            "status": "queued",
            "created_at": created_at,
            "client_request_id": client_request_id,
            "dataset": (
                {
                    **dataset,
                    "partition_checksum": snapshot["catalogue_provenance"]["partition_checksum"],
                    "feature_order_sha256": snapshot["catalogue_provenance"]["feature_order_sha256"],
                    "feature_order": snapshot["catalogue_provenance"]["feature_order"],
                    "label_source": snapshot["catalogue_provenance"]["label_source"],
                }
                if dataset["source"] == "catalogue"
                else dataset
            ),
            "snapshot_id": snapshot["snapshot_id"],
            "snapshot_sha256": snapshot["sha256"],
            "physical_parallelism_guaranteed": False,
            "snapshot_provenance": snapshot.get("catalogue_provenance"),
            "children": child_records,
        }
        run["status"] = calculate_parent_status(run["children"])
        run_store.write(run)

        successful_dispatches = 0
        for child in prepared_children:
            index = child["record_index"]
            try:
                task_id = launchers[child["detector_id"]].dispatch(child["prepared"])
                run["children"][index]["task_id"] = task_id
                successful_dispatches += 1
            except Exception as exc:
                run["children"][index]["status"] = "dispatch_failed"
                run["children"][index]["dispatch_error"] = str(exc)[:1000]
            run["status"] = calculate_parent_status(run["children"])
            run_store.write(run)

        if not successful_dispatches:
            run["status"] = "failed"
            run_store.write(run)
            if run["children"] and all(
                child["status"] == "validation_failed" for child in run["children"]
            ):
                return jsonify({"error": "no detector passed dataset validation", **run}), 422
            return jsonify({"error": "no child training task could be dispatched", **run}), 503
        return jsonify(run), 202

    @blueprint.get("/training_runs/<run_id>")
    def get_training_run(run_id: str):
        try:
            validate_identifier(run_id, "run_id")
            _snapshot_store, run_store = stores()
            stored_run = run_store.read(run_id)
        except TrainingRunValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except TrainingRunNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

        response = deepcopy(stored_run)
        configured_reader = current_app.config.get("TRAINING_RUN_STATUS_READER")
        reader = configured_reader or status_reader
        for child in response["children"]:
            if not child.get("task_id") or child["status"] in {
                "dispatch_failed",
                "validation_failed",
            }:
                continue
            try:
                task = reader(child["task_id"])
                child["status"] = normalize_celery_status(task.state)
                detail = _compact_task_info(getattr(task, "info", None))
                if detail is not None:
                    child["detail"] = detail
            except Exception as exc:
                child["status_detail"] = f"Celery status unavailable: {str(exc)[:500]}"
        response["status"] = calculate_parent_status(response["children"])
        return jsonify(response), 200

    return blueprint


def configure_training_run_defaults(app: Any) -> None:
    """Set local development defaults without overriding caller configuration."""
    app.config.setdefault(
        "EXISTING_DATASETS_ROOT",
        os.getenv("EXISTING_DATASETS_ROOT", str(Path(app.root_path) / "datasets")),
    )
    app.config.setdefault(
        "DATASET_SNAPSHOT_STORAGE_ROOT",
        os.getenv(
            "DATASET_SNAPSHOT_STORAGE_ROOT",
            str(Path(app.root_path) / "dataset_snapshots"),
        ),
    )
    app.config.setdefault(
        "TRAINING_RUN_STORAGE_ROOT",
        os.getenv("TRAINING_RUN_STORAGE_ROOT", str(Path(app.root_path) / "training_runs")),
    )
    app.config.setdefault(
        "API_DATA_CATALOGUE_URL",
        os.getenv("API_DATA_CATALOGUE_URL", "http://127.0.0.1:5001"),
    )
    app.config.setdefault("CATALOGUE_CONNECT_TIMEOUT_SECONDS", 5.0)
    app.config.setdefault("CATALOGUE_READ_TIMEOUT_SECONDS", 60.0)
    app.config.setdefault("CATALOGUE_MAX_BUNDLE_BYTES", 250 * 1024 * 1024)
