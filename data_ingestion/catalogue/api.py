"""Flask API for the local file-backed Data Catalogue."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any
import uuid

from flask import Blueprint, current_app, jsonify, request
from werkzeug.exceptions import BadRequest

from .legacy import LegacyDatasetProjector
from .fetching import (
    CatalogueFetchError,
    FetchLimits,
    fail_fetch_attempt,
    read_preview,
    start_fetch_attempt,
    validate_fetch_request,
)
from .models import build_draft_dataset
from .lifecycle import (
    CatalogueConflictError,
    add_incident,
    add_row_labels,
    calculate_usability,
    create_dataset_version,
    set_partition,
    update_dataset_metadata,
)
from .repository import (
    DatasetAlreadyExistsError,
    DatasetNotFoundError,
    FileCatalogueRepository,
)
from .validation import (
    ANOMALY_SCENARIOS,
    CatalogueValidationError,
    VERSION_STATUSES,
    WORKLOAD_CHARACTERISTICS,
    WORKLOAD_INTENSITIES,
    validate_identifier,
    validate_version,
)


def _positive_integer(value: str | None, field: str, default: int, maximum: int | None = None) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CatalogueValidationError(f"{field} must be an integer") from exc
    if parsed < 1 or (maximum is not None and parsed > maximum):
        suffix = f" between 1 and {maximum}" if maximum else " positive"
        raise CatalogueValidationError(f"{field} must be{suffix}")
    return parsed


def _boolean_filter(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise CatalogueValidationError("labels_available must be true or false")


def _enum_filter(value: str | None, field: str, choices: set[str]) -> str | None:
    if value is not None and value not in choices:
        raise CatalogueValidationError(
            f"{field} must be one of: {', '.join(sorted(choices))}"
        )
    return value


def _summary(record: dict[str, Any], version: dict[str, Any]) -> dict[str, Any]:
    source = version["source"]
    schema = version["technical_schema"]
    return {
        "dataset_id": record["dataset_id"],
        "version": record["latest_version"],
        "display_name": record["display_name"],
        "description": record["description"],
        "purpose": record["purpose"],
        "status": record["status"],
        "created_at": record["created_at"],
        "workload_context": record["workload_context"],
        "labels_available": record["ground_truth"]["labels_available"],
        "modality": schema.get("modality"),
        "observation_count": schema.get("observation_count", 0),
        "feature_count": schema.get("feature_count", 0),
        "application": source.get("application"),
        "namespace": source.get("namespace"),
        "start_time": source.get("start_time"),
        "end_time": source.get("end_time"),
        "partition_mode": version["partition"]["mode"],
        "validation_warning_count": len(version["validation"].get("warnings", [])),
        **calculate_usability(version),
    }


def _detail_with_usability(detail: dict[str, Any]) -> dict[str, Any]:
    return {**detail, **calculate_usability(detail["version"])}


def create_catalogue_blueprint() -> Blueprint:
    blueprint = Blueprint("data_catalogue", __name__)

    def components() -> tuple[FileCatalogueRepository, LegacyDatasetProjector]:
        repository = FileCatalogueRepository(current_app.config["CATALOGUE_STORAGE_ROOT"])
        projector = LegacyDatasetProjector(
            current_app.config["LEGACY_DATASETS_ROOT"], repository
        )
        return repository, projector

    @blueprint.get("/datasets")
    def list_datasets():
        try:
            page = _positive_integer(request.args.get("page"), "page", 1)
            page_size = _positive_integer(
                request.args.get("page_size"), "page_size", 20, maximum=100
            )
            status = _enum_filter(request.args.get("status"), "status", VERSION_STATUSES)
            intensity = _enum_filter(
                request.args.get("workload_intensity"),
                "workload_intensity",
                WORKLOAD_INTENSITIES,
            )
            characteristic = _enum_filter(
                request.args.get("dominant_workload_characteristic"),
                "dominant_workload_characteristic",
                WORKLOAD_CHARACTERISTICS,
            )
            scenario = _enum_filter(
                request.args.get("anomaly_scenario"),
                "anomaly_scenario",
                ANOMALY_SCENARIOS,
            )
            labels_available = _boolean_filter(request.args.get("labels_available"))
            modality = request.args.get("modality")
            purpose = request.args.get("purpose")
            search = request.args.get("search", "").strip().lower()

            repository, projector = components()
            projector.project_all()
            summaries = []
            for record in repository.list_records():
                version = repository.read_version(record["dataset_id"], record["latest_version"])
                item = _summary(record, version)
                context = item["workload_context"]
                searchable = " ".join(
                    str(value or "")
                    for value in (
                        item["dataset_id"],
                        item["display_name"],
                        item["description"],
                        item["application"],
                        item["namespace"],
                    )
                ).lower()
                if search and search not in searchable:
                    continue
                if status and item["status"] != status:
                    continue
                if modality and item["modality"] != modality:
                    continue
                if intensity and context.get("workload_intensity") != intensity:
                    continue
                if characteristic and context.get("dominant_workload_characteristic") != characteristic:
                    continue
                if scenario and context.get("anomaly_scenario") != scenario:
                    continue
                if labels_available is not None and item["labels_available"] != labels_available:
                    continue
                if purpose and purpose not in item["purpose"]:
                    continue
                summaries.append(item)

            summaries.sort(key=lambda item: (item["display_name"].lower(), item["dataset_id"]))
            total = len(summaries)
            start = (page - 1) * page_size
            return jsonify(
                {
                    "items": summaries[start : start + page_size],
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "pages": math.ceil(total / page_size) if total else 0,
                }
            )
        except CatalogueValidationError as exc:
            return jsonify({"error": str(exc)}), 400

    @blueprint.get("/datasets/<dataset_id>")
    def get_dataset(dataset_id: str):
        try:
            validate_identifier(dataset_id)
            repository, projector = components()
            projector.project_all()
            return jsonify(_detail_with_usability(repository.read_detail(dataset_id))), 200
        except CatalogueValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except DatasetNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @blueprint.post("/datasets")
    def create_dataset():
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415
        try:
            payload = request.get_json(silent=False)
            record, version = build_draft_dataset(payload)
            repository, projector = components()
            projector.project_all()
            repository.create(record, version)
            return jsonify(_detail_with_usability(repository.read_detail(record["dataset_id"]))), 201
        except (BadRequest, CatalogueValidationError) as exc:
            return jsonify({"error": str(exc)}), 400
        except DatasetAlreadyExistsError as exc:
            return jsonify({"error": str(exc)}), 409

    @blueprint.get("/datasets/<dataset_id>/versions/<int:version>")
    def get_dataset_version(dataset_id: str, version: int):
        try:
            validate_identifier(dataset_id)
            validate_version(version)
            repository, _projector = components()
            stored = repository.read_version(dataset_id, version)
            return jsonify({**stored, **calculate_usability(stored)})
        except CatalogueValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except DatasetNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @blueprint.post("/datasets/<dataset_id>/versions")
    def create_version(dataset_id: str):
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415
        try:
            repository, _projector = components()
            detail = create_dataset_version(
                repository, dataset_id, request.get_json(silent=False)
            )
            return jsonify(_detail_with_usability(detail)), 201
        except (BadRequest, CatalogueValidationError) as exc:
            return jsonify({"error": str(exc)}), 400
        except DatasetNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except DatasetAlreadyExistsError as exc:
            return jsonify({"error": str(exc)}), 409

    @blueprint.patch("/datasets/<dataset_id>")
    def patch_dataset(dataset_id: str):
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415
        try:
            repository, _projector = components()
            detail = update_dataset_metadata(
                repository, dataset_id, request.get_json(silent=False)
            )
            return jsonify(_detail_with_usability(detail))
        except (BadRequest, CatalogueValidationError) as exc:
            return jsonify({"error": str(exc)}), 400
        except DatasetNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @blueprint.post("/datasets/<dataset_id>/versions/<int:version>/incidents")
    def create_incident(dataset_id: str, version: int):
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415
        try:
            repository, _projector = components()
            incident = add_incident(
                repository, dataset_id, version, request.get_json(silent=False)
            )
            return jsonify(incident), 201
        except (BadRequest, CatalogueValidationError) as exc:
            return jsonify({"error": str(exc)}), 400
        except DatasetNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @blueprint.post("/datasets/<dataset_id>/versions/<int:version>/labels")
    def create_labels(dataset_id: str, version: int):
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415
        try:
            repository, _projector = components()
            labels = add_row_labels(
                repository, dataset_id, version, request.get_json(silent=False)
            )
            return jsonify(labels), 201
        except (BadRequest, CatalogueValidationError) as exc:
            return jsonify({"error": str(exc)}), 400
        except CatalogueConflictError as exc:
            return jsonify({"error": str(exc)}), 409
        except DatasetNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @blueprint.post("/datasets/<dataset_id>/versions/<int:version>/partitions")
    def create_partition(dataset_id: str, version: int):
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415
        try:
            repository, _projector = components()
            partition = set_partition(
                repository, dataset_id, version, request.get_json(silent=False)
            )
            return jsonify(partition), 201
        except (BadRequest, CatalogueValidationError) as exc:
            return jsonify({"error": str(exc)}), 400
        except DatasetNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @blueprint.post("/datasets/<dataset_id>/versions/<int:version>/fetch")
    def fetch_dataset(dataset_id: str, version: int):
        try:
            validate_identifier(dataset_id)
            validate_version(version)
            repository, _projector = components()
            stored_version = repository.read_version(dataset_id, version)
            executions = validate_fetch_request(stored_version, dict(current_app.config))
            dispatcher = current_app.extensions.get("catalogue_fetch_dispatch")
            if dispatcher is None:
                return jsonify({"error": "catalogue fetch dispatcher is unavailable"}), 503
            attempt_id = f"attempt-{uuid.uuid4().hex}"
            task_id = f"catalogue-fetch-{uuid.uuid4().hex}"
            start_fetch_attempt(
                repository,
                dataset_id,
                version,
                attempt_id,
                task_id,
                len(executions),
            )
            try:
                dispatcher(dataset_id, version, attempt_id, task_id)
            except Exception as exc:
                fail_fetch_attempt(repository, dataset_id, version, attempt_id, exc)
                return jsonify({"error": "catalogue fetch task could not be dispatched"}), 503
            return (
                jsonify(
                    {
                        "dataset_id": dataset_id,
                        "version": version,
                        "status": "fetching",
                        "fetch_task_id": task_id,
                        "attempt_id": attempt_id,
                    }
                ),
                202,
            )
        except CatalogueValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except DatasetNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @blueprint.get("/datasets/<dataset_id>/fetch-status")
    def fetch_status(dataset_id: str):
        try:
            validate_identifier(dataset_id)
            raw_version = request.args.get("version")
            if raw_version is None:
                raise CatalogueValidationError("version query parameter is required")
            try:
                version_number = int(raw_version)
            except ValueError as exc:
                raise CatalogueValidationError("version must be a positive integer") from exc
            validate_version(version_number)
            repository, _projector = components()
            version = repository.read_version(dataset_id, version_number)
            fetch = version.get("fetch") or {}
            return jsonify(
                {
                    "dataset_id": dataset_id,
                    "version": version_number,
                    "status": version["status"],
                    "phase": fetch.get("phase", version["status"]),
                    "fetch_task_id": fetch.get("task_id"),
                    "attempt_id": fetch.get("attempt_id"),
                    "progress": fetch.get("progress", {"completed": 0, "total": 0}),
                    "warnings": fetch.get("warnings", []),
                    "error": fetch.get("error"),
                }
            )
        except CatalogueValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except DatasetNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @blueprint.get("/datasets/<dataset_id>/preview")
    def preview_dataset(dataset_id: str):
        try:
            validate_identifier(dataset_id)
            raw_version = request.args.get("version")
            if raw_version is None:
                raise CatalogueValidationError("version query parameter is required")
            try:
                version_number = int(raw_version)
            except ValueError as exc:
                raise CatalogueValidationError("version must be a positive integer") from exc
            validate_version(version_number)
            maximum = FetchLimits.from_config(dict(current_app.config)).maximum_preview_rows
            limit = _positive_integer(request.args.get("limit"), "limit", min(20, maximum), maximum)
            repository, _projector = components()
            return jsonify(read_preview(repository, dataset_id, version_number, limit))
        except CatalogueValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except DatasetNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except CatalogueFetchError as exc:
            return jsonify({"error": str(exc)}), 409

    return blueprint


def configure_catalogue_defaults(app: Any) -> None:
    app.config.setdefault(
        "CATALOGUE_STORAGE_ROOT",
        os.getenv(
            "CATALOGUE_STORAGE_ROOT",
            str(Path(app.root_path) / "catalogue_storage"),
        ),
    )
    app.config.setdefault(
        "CATALOGUE_PROMETHEUS_SOURCES",
        {
            "cluster-default": os.getenv(
                "CATALOGUE_CLUSTER_DEFAULT_PROMETHEUS_URL",
                "http://prometheus-server.monitoring.svc.cluster.local:80",
            )
        },
    )
    defaults = {
        "CATALOGUE_MAX_QUERY_COUNT": 25,
        "CATALOGUE_MAX_QUERY_LENGTH": 10000,
        "CATALOGUE_MAX_WINDOW_SECONDS": 31 * 24 * 60 * 60,
        "CATALOGUE_MIN_SAMPLING_INTERVAL_SECONDS": 5,
        "CATALOGUE_MAX_THEORETICAL_SAMPLES": 2_000_000,
        "CATALOGUE_MAX_SERIES": 1000,
        "CATALOGUE_MAX_RESPONSE_BYTES": 25 * 1024 * 1024,
        "CATALOGUE_MAX_PREVIEW_ROWS": 100,
        "CATALOGUE_CONNECT_TIMEOUT_SECONDS": 5.0,
        "CATALOGUE_READ_TIMEOUT_SECONDS": 30.0,
        "CATALOGUE_MAX_RETRIES": 2,
    }
    for key, value in defaults.items():
        app.config.setdefault(key, value)
    app.config.setdefault(
        "LEGACY_DATASETS_ROOT",
        os.getenv(
            "LEGACY_DATASETS_ROOT",
            str(Path(app.root_path).parent / "learning_adaptation" / "datasets"),
        ),
    )
