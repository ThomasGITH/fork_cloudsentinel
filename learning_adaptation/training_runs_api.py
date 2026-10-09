"""HTTP API and orchestration service for generic multi-detector training runs."""

from __future__ import annotations

import math
import os
from pathlib import Path
import csv
from typing import Any, Callable
import uuid
from copy import deepcopy

from flask import Blueprint, current_app, jsonify, request
from werkzeug.exceptions import BadRequest

from detectors import registry


def _validate_metrics_partition_v1(snapshot: dict[str, Any]) -> None:
    """Validate the shared v1 envelope once before any child is dispatched."""

    directory = Path(snapshot["directory"]).resolve()
    shapes: dict[str, tuple[int, int]] = {}
    for logical_name in ("train", "test", "labels"):
        path = (directory / f"{logical_name}.csv").resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise TrainingRunValidationError(
                f"metrics-partition-v1 is missing {logical_name} data"
            )
        try:
            with path.open("r", encoding="utf-8", newline="") as source:
                rows = list(csv.reader(source))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise TrainingRunValidationError(
                f"invalid {logical_name} CSV: {exc}"
            ) from exc
        if not rows or any(not row for row in rows):
            raise TrainingRunValidationError(f"{logical_name} CSV is empty or malformed")
        width = len(rows[0])
        if width < 1 or any(len(row) != width for row in rows):
            raise TrainingRunValidationError(f"{logical_name} CSV is not rectangular")
        shapes[logical_name] = (len(rows), width)
    if shapes["train"][1] != shapes["test"][1]:
        raise TrainingRunValidationError(
            "train and test data must have the same feature count"
        )
    if shapes["labels"][1] != 1 or shapes["labels"][0] != shapes["test"][0]:
        raise TrainingRunValidationError(
            "metrics-partition-v1 labels must contain one value per test observation"
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
    from training_run_storage import (
        DatasetSnapshotStore,
        TrainingRunNotFoundError,
        TrainingRunStore,
        TrainingRunValidationError,
        new_identifier,
        validate_identifier,
    )


def validate_request_payload(
    payload: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], str | None, str | None]:
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

    manifests = {
        item["id"]: item for item in registry.scan_detectors()["detectors"]
    }
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
        try:
            descriptor = registry.get_descriptor(detector_id)
        except (registry.UnknownDetectorError, registry.ManifestValidationError) as exc:
            raise TrainingRunValidationError(
                f"detector {detector_id!r} changed during request validation"
            ) from exc
        manifest = descriptor.manifest
        training_capability = manifest.get("capabilities", {}).get("training", {})
        if not training_capability.get("enabled"):
            raise TrainingRunValidationError(
                f"detector {detector_id!r} does not declare training capability"
            )
        if training_capability.get("protocol") not in registry.SUPPORTED_TRAINING_PROTOCOLS:
            raise TrainingRunValidationError(
                f"detector {detector_id!r} uses an unsupported training protocol"
            )
        if training_capability.get("input_profile") not in registry.SUPPORTED_INPUT_PROFILES:
            raise TrainingRunValidationError(
                f"detector {detector_id!r} uses an unsupported input profile"
            )
        if descriptor.source_class == "external" and descriptor.runtime_status.get(
            "training_runtime_status"
        ) != "ready":
            raise TrainingRunValidationError(
                f"external detector {detector_id!r} is not ready in the training runtime"
            )
        try:
            parameters = registry.validate_training_parameters(
                detector_id, detector_request.get("parameters"), manifest
            )
        except registry.ParameterValidationError as exc:
            raise TrainingRunValidationError(str(exc)) from exc
        validated.append(
            {
                "detector_id": detector_id,
                "parameters": parameters,
                "plugin_reference": descriptor.reference().to_dict(),
            }
        )

    client_request_id = payload.get("client_request_id")
    if client_request_id is not None and (
        not isinstance(client_request_id, str)
        or not client_request_id.strip()
        or len(client_request_id) > 200
    ):
        raise TrainingRunValidationError(
            "client_request_id must be a non-empty string of at most 200 characters"
        )
    run_name = payload.get("run_name")
    if run_name is not None and (
        not isinstance(run_name, str)
        or not run_name.strip()
        or len(run_name.strip()) > 200
    ):
        raise TrainingRunValidationError(
            "run_name must be a non-empty string of at most 200 characters"
        )
    return normalized_dataset, validated, client_request_id, (
        run_name.strip() if run_name is not None else None
    )


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


def create_training_runs_blueprint(
    training_task: Any,
    status_reader: Callable[[str], Any],
) -> Blueprint:
    blueprint = Blueprint("training_runs", __name__)

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
            dataset, detector_requests, client_request_id, run_name = validate_request_payload(
                payload
            )
            snapshot_store, run_store, _model_store = stores()
        except (BadRequest, TrainingRunValidationError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

        run_id = new_identifier("run")
        snapshot = None
        downloaded_source = None
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
            _validate_metrics_partition_v1(snapshot)
        except (TrainingRunValidationError, CatalogueBundleError, ValueError) as exc:
            if snapshot is not None:
                snapshot_store.remove(snapshot["snapshot_id"])
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
        dispatch_children = []
        created_at = utc_now()
        for detector_request in detector_requests:
            detector_id = detector_request["detector_id"]
            model_id = f"{run_id}-{detector_id}-{uuid.uuid4().hex[:8]}"
            index = len(child_records)
            record = {
                "child_id": f"{run_id}-{detector_id}",
                "detector_id": detector_id,
                "model_id": model_id,
                "task_id": None,
                "status": "queued",
                "parameters": detector_request["parameters"],
                "snapshot_id": snapshot["snapshot_id"],
                "plugin": detector_request["plugin_reference"],
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
            child_records.append(record)
            dispatch_children.append(
                {
                    "record_index": index,
                    "context_payload": {
                        "run_id": run_id,
                        "detector_id": detector_id,
                        "model_id": model_id,
                        "snapshot_id": snapshot["snapshot_id"],
                        "plugin": detector_request["plugin_reference"],
                    },
                }
            )

        run = {
            "schema_version": 2 if dataset["source"] == "catalogue" else 1,
            "run_id": run_id,
            "status": "queued",
            "created_at": created_at,
            "started_at": None,
            "updated_at": created_at,
            "completed_at": None,
            "client_request_id": client_request_id,
            "run_name": run_name,
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
        for child in dispatch_children:
            index = child["record_index"]
            try:
                task_id = training_task.apply_async(
                    args=[child["context_payload"]]
                ).id
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

    @blueprint.get("/internal/models/<model_id>/live-input-recipe")
    def get_model_live_input_recipe(model_id: str):
        """Return a pinned recipe to trusted server-side monitoring callers."""
        try:
            validate_identifier(model_id, "model_id")
            _snapshot_store, _run_store, model_store = stores()
            record = model_store.read(model_id)
        except TrainingRunValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except ModelNotFoundError:
            return jsonify({"error": "unknown model"}), 404
        inference = record.get("inference", {})
        live = record.get("live_monitoring", {})
        recipe = live.get("recipe") if isinstance(live, dict) else None
        if (
            record.get("status") != "available"
            or inference.get("status") != "ready"
            or inference.get("contract") != "cloudsentinel.inference/v1"
            or live.get("status") != "ready"
            or not isinstance(recipe, dict)
        ):
            return jsonify({"error": "model is not ready for live monitoring"}), 409
        return jsonify(
            {
                "model_id": record["model_id"],
                "detector_id": record["detector_id"],
                "detector_version": record["detector_version"],
                "artifact_manifest_sha256": inference.get(
                    "artifact_manifest_sha256"
                ),
                "feature_identity": deepcopy(record["feature_identity"]),
                "recipe": deepcopy(recipe),
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
