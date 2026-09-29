"""HTTP API and orchestration service for generic multi-detector training runs."""

from __future__ import annotations

import hashlib
import json
import math
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
    from learning_adaptation.model_catalogue import (
        ModelCatalogueStore,
        ModelNotFoundError,
        ModelRecordError,
    )
    from learning_adaptation.training_lifecycle import (
        calculate_parent_status,
        history_summary,
        model_summary,
        normalize_run_record,
        public_model_record,
        reconcile_run,
        safe_failure_summary,
        utc_now,
    )
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
        "learning_adaptation.model_catalogue",
        "learning_adaptation.training_lifecycle",
        "learning_adaptation.training_launchers",
        "learning_adaptation.training_run_storage",
    }:
        raise
    from model_catalogue import ModelCatalogueStore, ModelNotFoundError, ModelRecordError
    from training_lifecycle import (
        calculate_parent_status,
        history_summary,
        model_summary,
        normalize_run_record,
        public_model_record,
        reconcile_run,
        safe_failure_summary,
        utc_now,
    )
    from training_launchers import build_training_launchers
    from training_run_storage import (
        DatasetSnapshotStore,
        TrainingRunNotFoundError,
        TrainingRunStore,
        TrainingRunValidationError,
        new_identifier,
        validate_identifier,
    )


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
        if "exclusive_minimum" in metadata and value <= metadata["exclusive_minimum"]:
            raise TrainingRunValidationError(
                f"parameter {name!r} for {detector_id!r} must be greater than {metadata['exclusive_minimum']}"
            )
        if "exclusive_maximum" in metadata and value >= metadata["exclusive_maximum"]:
            raise TrainingRunValidationError(
                f"parameter {name!r} for {detector_id!r} must be less than {metadata['exclusive_maximum']}"
            )
        if "allowed_values" in metadata and value not in metadata["allowed_values"]:
            raise TrainingRunValidationError(
                f"parameter {name!r} for {detector_id!r} must be one of {metadata['allowed_values']}"
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


def _positive_integer(value: str | None, field: str, default: int, maximum: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TrainingRunValidationError(f"{field} must be an integer") from exc
    if parsed < 1 or parsed > maximum:
        raise TrainingRunValidationError(
            f"{field} must be between 1 and {maximum}"
        )
    return parsed


def _choice(value: str | None, field: str, choices: set[str]) -> str | None:
    if value is not None and value not in choices:
        raise TrainingRunValidationError(
            f"{field} must be one of: {', '.join(sorted(choices))}"
        )
    return value


def _model_feature_identity(snapshot: dict[str, Any]) -> dict[str, Any]:
    provenance = snapshot.get("catalogue_provenance") or {}
    feature_order = provenance.get("feature_order")
    feature_hash = provenance.get("feature_order_sha256")
    if not feature_order:
        details = snapshot.get("dataset_details", {})
        containers = details.get("containers", [])
        metrics = details.get("metrics", [])
        if isinstance(containers, list) and isinstance(metrics, list):
            feature_order = [
                f"{container}_{metric}"
                for container in containers
                for metric in metrics
            ]
        else:
            feature_order = []
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


def create_training_runs_blueprint(
    cgnn_task: Any,
    isolation_forest_task: Any,
    status_reader: Callable[[str], Any],
) -> Blueprint:
    blueprint = Blueprint("training_runs", __name__)
    launchers = build_training_launchers(cgnn_task, isolation_forest_task)

    def stores() -> tuple[DatasetSnapshotStore, TrainingRunStore, ModelCatalogueStore]:
        snapshot_root = current_app.config["DATASET_SNAPSHOT_STORAGE_ROOT"]
        dataset_root = current_app.config["EXISTING_DATASETS_ROOT"]
        run_root = current_app.config["TRAINING_RUN_STORAGE_ROOT"]
        return (
            DatasetSnapshotStore(snapshot_root, dataset_root),
            TrainingRunStore(run_root),
            ModelCatalogueStore(current_app.config["MODEL_CATALOGUE_STORAGE_ROOT"]),
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
            snapshot_store, run_store, _model_store = stores()
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
        created_at = utc_now()
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
                    "failure_summary": None,
                    "result_metadata": None,
                    "model_status": None,
                    "created_at": created_at,
                    "started_at": None,
                    "updated_at": created_at,
                    "completed_at": None,
                }
                try:
                    manifest = next(
                        item
                        for item in registry.discover_detectors()
                        if item["id"] == detector_id
                    )
                    catalogue_provenance = snapshot.get("catalogue_provenance") or {}
                    model_context = {
                        "training_run_id": run_id,
                        "training_child_id": f"{run_id}-{detector_id}",
                        "model_id": model_id,
                        "detector_id": detector_id,
                        "detector_version": manifest["version"],
                        "dataset": (
                            {
                                **dataset,
                                **(
                                    {
                                        "partition_checksum": catalogue_provenance.get(
                                            "partition_checksum"
                                        )
                                    }
                                    if dataset["source"] == "catalogue"
                                    else {}
                                ),
                            }
                        ),
                        "snapshot": {
                            "snapshot_id": snapshot["snapshot_id"],
                            "snapshot_sha256": snapshot["sha256"],
                        },
                        "feature_identity": _model_feature_identity(snapshot),
                        "training_parameters": detector_request["parameters"],
                    }
                    prepared = launchers[detector_id].prepare(
                        snapshot,
                        child_input_dir,
                        detector_request["parameters"],
                        {
                            "run_id": run_id,
                            "model_id": model_id,
                            "dataset_id": dataset["dataset_id"],
                            "model_context": model_context,
                        },
                    )
                    prepared_children.append(
                        {**record, "prepared": prepared, "record_index": len(child_records)}
                    )
                except TrainingRunValidationError as exc:
                    if dataset["source"] == "existing":
                        raise
                    record["status"] = "validation_failed"
                    record["validation_error"] = safe_failure_summary(exc)
                    record["completed_at"] = created_at
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

        run = {
            "schema_version": 2 if dataset["source"] == "catalogue" else 1,
            "run_id": run_id,
            "status": "queued",
            "created_at": created_at,
            "started_at": None,
            "updated_at": created_at,
            "completed_at": None,
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
                def dispatched(stored, task_id=task_id, index=index):
                    normalized = normalize_run_record(stored)
                    normalized["children"][index]["task_id"] = task_id
                    normalized["children"][index]["updated_at"] = utc_now()
                    normalized["updated_at"] = utc_now()
                    normalized["status"] = calculate_parent_status(normalized["children"])
                    return normalized

                run_store.update(run_id, dispatched)
                successful_dispatches += 1
            except Exception as exc:
                def dispatch_failed(stored, exc=exc, index=index):
                    normalized = normalize_run_record(stored)
                    now = utc_now()
                    target = normalized["children"][index]
                    target["status"] = "dispatch_failed"
                    target["dispatch_error"] = safe_failure_summary(exc)
                    target["failure_summary"] = safe_failure_summary(exc)
                    target["updated_at"] = now
                    target["completed_at"] = now
                    normalized["updated_at"] = now
                    normalized["status"] = calculate_parent_status(normalized["children"])
                    return normalized

                run_store.update(run_id, dispatch_failed)

        run = run_store.read(run_id)

        if not successful_dispatches:
            def no_dispatch(stored):
                normalized = normalize_run_record(stored)
                now = utc_now()
                normalized["status"] = "failed"
                normalized["updated_at"] = now
                normalized["completed_at"] = now
                return normalized

            run = run_store.update(run_id, no_dispatch)
            if run["children"] and all(
                child["status"] == "validation_failed" for child in run["children"]
            ):
                return jsonify({"error": "no detector passed dataset validation", **run}), 422
            return jsonify({"error": "no child training task could be dispatched", **run}), 503
        return jsonify(run), 202

    @blueprint.get("/training_runs")
    def list_training_runs():
        try:
            page = _positive_integer(request.args.get("page"), "page", 1, 1_000_000)
            page_size = _positive_integer(
                request.args.get("page_size"), "page_size", 20, 100
            )
            status = _choice(
                request.args.get("status"),
                "status",
                {"queued", "running", "completed", "partial_success", "failed"},
            )
            source = _choice(
                request.args.get("source"), "source", {"existing", "catalogue"}
            )
            sort = _choice(
                request.args.get("sort", "newest"), "sort", {"newest", "oldest"}
            )
            detector_id = request.args.get("detector_id")
            dataset_id = request.args.get("dataset_id")
            if detector_id is not None:
                validate_identifier(detector_id, "detector_id")
            if dataset_id is not None:
                validate_identifier(dataset_id, "dataset_id")
            _snapshot_store, run_store, _model_store = stores()
        except TrainingRunValidationError as exc:
            return jsonify({"error": str(exc)}), 400

        configured_reader = current_app.config.get("TRAINING_RUN_STATUS_READER")
        reader = configured_reader or status_reader
        items = []
        corrupt = 0
        for run_id in run_store.identifiers():
            try:
                run = reconcile_run(run_store, run_id, reader)
            except Exception:
                try:
                    run = normalize_run_record(run_store.read(run_id))
                except Exception:
                    corrupt += 1
                    continue
            if status and run.get("status") != status:
                continue
            if source and run.get("dataset", {}).get("source") != source:
                continue
            if dataset_id and run.get("dataset", {}).get("dataset_id") != dataset_id:
                continue
            if detector_id and not any(
                child.get("detector_id") == detector_id
                for child in run.get("children", [])
            ):
                continue
            items.append(history_summary(run))
        items.sort(
            key=lambda item: (item.get("created_at") or "", item.get("run_id") or ""),
            reverse=sort == "newest",
        )
        total = len(items)
        start = (page - 1) * page_size
        return jsonify(
            {
                "items": items[start : start + page_size],
                "page": page,
                "page_size": page_size,
                "total": total,
                "pages": math.ceil(total / page_size) if total else 0,
                "skipped_corrupt_records": corrupt,
            }
        )

    @blueprint.get("/training_runs/<run_id>")
    def get_training_run(run_id: str):
        try:
            validate_identifier(run_id, "run_id")
            _snapshot_store, run_store, _model_store = stores()
            stored_run = run_store.read(run_id)
        except TrainingRunValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except TrainingRunNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

        configured_reader = current_app.config.get("TRAINING_RUN_STATUS_READER")
        reader = configured_reader or status_reader
        try:
            response = reconcile_run(run_store, run_id, reader)
        except Exception:
            response = normalize_run_record(stored_run)
        return jsonify(response), 200

    @blueprint.get("/models")
    def list_models():
        try:
            page = _positive_integer(request.args.get("page"), "page", 1, 1_000_000)
            page_size = _positive_integer(
                request.args.get("page_size"), "page_size", 20, 100
            )
            status = _choice(request.args.get("status"), "status", {"available"})
            sort = _choice(
                request.args.get("sort", "newest"), "sort", {"newest", "oldest"}
            )
            detector_id = request.args.get("detector_id")
            dataset_id = request.args.get("dataset_id")
            training_run_id = request.args.get("training_run_id")
            for value, field in (
                (detector_id, "detector_id"),
                (dataset_id, "dataset_id"),
                (training_run_id, "training_run_id"),
            ):
                if value is not None:
                    validate_identifier(value, field)
            _snapshot_store, _run_store, model_store = stores()
        except TrainingRunValidationError as exc:
            return jsonify({"error": str(exc)}), 400

        records, corrupt = model_store.records()
        records = [
            record
            for record in records
            if (not status or record["status"] == status)
            and (not detector_id or record["detector_id"] == detector_id)
            and (not dataset_id or record["dataset"].get("dataset_id") == dataset_id)
            and (
                not training_run_id
                or record["training_run_id"] == training_run_id
            )
        ]
        records.sort(
            key=lambda item: (item["created_at"], item["model_id"]),
            reverse=sort == "newest",
        )
        total = len(records)
        start = (page - 1) * page_size
        return jsonify(
            {
                "items": [
                    model_summary(record)
                    for record in records[start : start + page_size]
                ],
                "page": page,
                "page_size": page_size,
                "total": total,
                "pages": math.ceil(total / page_size) if total else 0,
                "skipped_corrupt_records": corrupt,
            }
        )

    @blueprint.get("/models/<model_id>")
    def get_model(model_id: str):
        try:
            validate_identifier(model_id, "model_id")
            _snapshot_store, _run_store, model_store = stores()
            return jsonify(public_model_record(model_store.read(model_id))), 200
        except TrainingRunValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except ModelNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except ModelRecordError:
            return jsonify({"error": "stored model metadata is unavailable"}), 500

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
        "MODEL_CATALOGUE_STORAGE_ROOT",
        os.getenv(
            "MODEL_CATALOGUE_STORAGE_ROOT",
            str(Path(app.config["TRAINING_RUN_STORAGE_ROOT"]).parent / "models"),
        ),
    )
    app.config.setdefault(
        "API_DATA_CATALOGUE_URL",
        os.getenv("API_DATA_CATALOGUE_URL", "http://127.0.0.1:5001"),
    )
    app.config.setdefault("CATALOGUE_CONNECT_TIMEOUT_SECONDS", 5.0)
    app.config.setdefault("CATALOGUE_READ_TIMEOUT_SECONDS", 60.0)
    app.config.setdefault("CATALOGUE_MAX_BUNDLE_BYTES", 250 * 1024 * 1024)
