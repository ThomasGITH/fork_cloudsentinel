"""Flask API for the local file-backed Data Catalogue."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, jsonify, request
from werkzeug.exceptions import BadRequest

from .legacy import LegacyDatasetProjector
from .models import build_draft_dataset
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
    }


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
            return jsonify(repository.read_detail(dataset_id)), 200
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
            return jsonify(repository.read_detail(record["dataset_id"])), 201
        except (BadRequest, CatalogueValidationError) as exc:
            return jsonify({"error": str(exc)}), 400
        except DatasetAlreadyExistsError as exc:
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
        "LEGACY_DATASETS_ROOT",
        os.getenv(
            "LEGACY_DATASETS_ROOT",
            str(Path(app.root_path).parent / "learning_adaptation" / "datasets"),
        ),
    )
