"""Public API for persisted anomaly-detection comparison runs."""

from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
from typing import Any, Callable
import uuid

import requests
from flask import Blueprint, current_app, jsonify, request
from werkzeug.exceptions import BadRequest

try:
    from learning_adaptation.comparison_storage import (
        ComparisonConflictError,
        ComparisonNotFoundError,
        ComparisonStore,
    )
    from learning_adaptation.comparison_robustness import aggregate_robustness
    from learning_adaptation.model_catalogue import ModelCatalogueStore, ModelNotFoundError
    from learning_adaptation.training_lifecycle import (
        safe_failure_summary,
        stored_evaluation_projection,
        utc_now,
    )
    from learning_adaptation.training_run_storage import validate_identifier
except ModuleNotFoundError:
    from comparison_storage import ComparisonConflictError, ComparisonNotFoundError, ComparisonStore
    from comparison_robustness import aggregate_robustness
    from model_catalogue import ModelCatalogueStore, ModelNotFoundError
    from training_lifecycle import safe_failure_summary, stored_evaluation_projection, utc_now
    from training_run_storage import validate_identifier


class ComparisonValidationError(ValueError):
    pass


class ComparisonCompatibilityError(ComparisonValidationError):
    pass


def _positive(value: str | None, field: str, default: int, maximum: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonValidationError(f"{field} must be an integer") from exc
    if parsed < 1 or parsed > maximum:
        raise ComparisonValidationError(f"{field} must be between 1 and {maximum}")
    return parsed


def _boolean_query(value: str | None, field: str) -> bool:
    if value in (None, "", "false", "0"):
        return False
    if value in {"true", "1"}:
        return True
    raise ComparisonValidationError(f"{field} must be true or false")


def _catalogue_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    getter = current_app.config.get("COMPARISON_CATALOGUE_GET") or requests.get
    try:
        response = getter(
            f"{current_app.config['API_DATA_CATALOGUE_URL'].rstrip('/')}{path}",
            params=params,
            timeout=(5.0, 30.0),
        )
    except requests.RequestException as exc:
        raise ComparisonValidationError("Data Catalogue is temporarily unavailable") from exc
    if response.status_code == 404:
        raise ComparisonValidationError("evaluation dataset was not found")
    if response.status_code >= 400:
        raise ComparisonValidationError("evaluation dataset could not be validated")
    try:
        payload = response.json()
    except ValueError as exc:
        raise ComparisonValidationError("Data Catalogue returned invalid metadata") from exc
    if not isinstance(payload, dict):
        raise ComparisonValidationError("Data Catalogue returned invalid metadata")
    return payload


def _dataset_context(dataset_id: str, version_number: int | None = None) -> dict[str, Any]:
    validate_identifier(dataset_id, "dataset_id")
    detail = _catalogue_get(f"/datasets/{dataset_id}")
    if version_number is None:
        version_number = detail.get("latest_version") or detail.get("version", {}).get("version")
    if isinstance(version_number, bool) or not isinstance(version_number, int) or version_number < 1:
        raise ComparisonValidationError("dataset version must be a positive integer")
    version = _catalogue_get(f"/datasets/{dataset_id}/versions/{version_number}")
    partition = version.get("partition") or {}
    schema = version.get("technical_schema") or {}
    if version.get("status") != "available":
        raise ComparisonValidationError("evaluation dataset version must be available")
    if partition.get("mode") in {None, "none"} or not partition.get("partition_id"):
        raise ComparisonValidationError("evaluation dataset requires an explicit partition")
    order = schema.get("feature_order") or []
    reference = partition.get("feature_order_reference") or {}
    feature_hash = reference.get("sha256")
    if not order or not feature_hash:
        raise ComparisonValidationError("evaluation dataset feature identity is incomplete")
    return {
        "dataset_id": dataset_id,
        "display_name": detail.get("display_name") or dataset_id,
        "version": version_number,
        "partition_id": partition["partition_id"],
        "partition_checksum": partition.get("checksum"),
        "status": version["status"],
        "modality": schema.get("modality") or "metrics",
        "feature_order": order,
        "feature_order_sha256": feature_hash,
        "observation_count": (partition.get("test") or {}).get(
            "observation_count", (partition.get("test") or {}).get("row_count")
        ),
        "start_time": (partition.get("test") or {}).get("start_time") or version.get("source", {}).get("start_time"),
        "end_time": (partition.get("test") or {}).get("end_time") or version.get("source", {}).get("end_time"),
        "workload_context": version.get("workload_context") or detail.get("workload_context", {}),
        "ground_truth_available": bool((version.get("ground_truth") or {}).get("labels_available")),
        "known_incident_windows": deepcopy(version.get("incident_context") or []),
    }


def _model_compatibility(record: dict[str, Any], dataset: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    inference = record.get("inference") or {}
    if record.get("status") != "available" or inference.get("status") != "ready":
        reasons.append("Model artifact is not inference-ready")
    if inference.get("contract") != "cloudsentinel.inference/v1":
        reasons.append("Model uses an unsupported inference contract")
    if not inference.get("artifact_id") or not inference.get("artifact_manifest_sha256"):
        reasons.append("Model artifact metadata is incomplete")
    identity = record.get("feature_identity") or {}
    if identity.get("feature_order_sha256") != dataset["feature_order_sha256"]:
        reasons.append("Feature identity does not match this evaluation dataset")
    model_modality = (
        record.get("modality")
        or (record.get("model_metadata") or {}).get("modality")
        or "metrics"
    )
    if model_modality != dataset.get("modality"):
        reasons.append(
            f"Requires {str(model_modality).title()} data"
        )
    return not reasons, reasons


def _selected_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    inference = record["inference"]
    plugin = record.get("plugin") or {}
    return {
        "model_id": record["model_id"],
        "display_name": record.get("model_metadata", {}).get("display_name") or record["model_id"],
        "detector_id": record["detector_id"],
        "detector_version": record["detector_version"],
        "model_created_at": record["created_at"],
        "training_run_id": record["training_run_id"],
        "training_dataset": deepcopy(record.get("dataset", {})),
        "stored_evaluation": stored_evaluation_projection(record),
        "live_monitoring": {
            key: deepcopy(record.get("live_monitoring", {}).get(key))
            for key in (
                "status",
                "contract",
                "recipe_sha256",
                "feature_order_sha256",
                "sampling_interval_seconds",
                "reason",
            )
            if record.get("live_monitoring", {}).get(key) is not None
        },
        "feature_identity": deepcopy(record["feature_identity"]),
        "artifact_identity": {
            "artifact_id": inference["artifact_id"],
            "artifact_manifest_sha256": inference["artifact_manifest_sha256"],
            "contract": inference["contract"],
            "artifact_format": inference["artifact_format"],
        },
        "plugin": {
            key: plugin.get(key)
            for key in ("source", "detector_id", "detector_version", "runtime_profile")
            if plugin.get(key) is not None
        },
    }


def public_comparison(record: dict[str, Any], *, summary: bool = False) -> dict[str, Any]:
    allowed = {
        "schema_version", "comparison_id", "name", "analysis_type", "configuration_version",
        "status", "created_at", "started_at", "updated_at", "completed_at",
        "evaluation_dataset", "modality", "feature_identity", "ground_truth_available",
        "known_incident_window", "known_incident_windows", "result_reference", "safe_failure_summary",
        "deterministic_summary", "results",
    }
    value = {key: deepcopy(item) for key, item in record.items() if key in allowed}
    selected = []
    for item in record.get("selected_models", []):
        selected.append({
            key: deepcopy(item.get(key))
            for key in (
                "model_id", "display_name", "detector_id", "detector_version",
                "model_created_at", "training_run_id", "training_dataset", "stored_evaluation",
                "feature_identity", "artifact_identity", "plugin",
                "live_monitoring",
            )
        })
    value["selected_models"] = selected
    value["model_count"] = len(selected)
    if summary:
        value.pop("results", None)
        value.pop("feature_identity", None)
        winner = None
        labelled = [
            item for item in record.get("results", [])
            if item.get("status") == "completed" and item.get("metrics", {}).get("f1_score") is not None
        ]
        if labelled:
            highest = max(item["metrics"]["f1_score"] for item in labelled)
            winners = [item for item in labelled if item["metrics"]["f1_score"] == highest]
            if len(winners) == 1:
                winner = winners[0]["display_name"]
        value["f1_winner"] = winner
    return value


def create_comparisons_blueprint(comparison_task: Any) -> Blueprint:
    blueprint = Blueprint("comparisons", __name__, url_prefix="/api/comparisons")

    def stores() -> tuple[ComparisonStore, ModelCatalogueStore]:
        return (
            ComparisonStore(current_app.config["COMPARISON_STORAGE_ROOT"]),
            ModelCatalogueStore(current_app.config["MODEL_CATALOGUE_STORAGE_ROOT"]),
        )

    @blueprint.get("")
    def list_comparisons():
        try:
            page = _positive(request.args.get("page"), "page", 1, 1_000_000)
            page_size = _positive(request.args.get("page_size"), "page_size", 20, 100)
            status = request.args.get("status")
            if status and status not in {"queued", "running", "completed", "partial", "failed"}:
                raise ComparisonValidationError("unsupported comparison status")
            modality = request.args.get("modality")
            workload = request.args.get("workload")
            related_model_id = request.args.get("model_id")
            related_artifact_id = request.args.get("artifact_id")
            related_artifact_hash = request.args.get("artifact_manifest_sha256")
            if any((related_model_id, related_artifact_id, related_artifact_hash)) and not all(
                (related_model_id, related_artifact_id, related_artifact_hash)
            ):
                raise ComparisonValidationError(
                    "related artifact filtering requires model_id, artifact_id and artifact_manifest_sha256"
                )
            if related_model_id:
                validate_identifier(related_model_id, "model_id")
                validate_identifier(related_artifact_id, "artifact_id")
                if len(related_artifact_hash) != 64 or any(
                    character not in "0123456789abcdef"
                    for character in related_artifact_hash.lower()
                ):
                    raise ComparisonValidationError(
                        "artifact_manifest_sha256 must be a SHA-256 digest"
                    )
            search = request.args.get("search", "").strip().lower()
            store, _models = stores()
            items = []
            skipped = 0
            for comparison_id in store.identifiers():
                try:
                    item = store.read(comparison_id)
                except Exception:
                    skipped += 1
                    continue
                context = item.get("evaluation_dataset", {}).get("workload_context", {})
                haystack = " ".join((item.get("name", ""), comparison_id, item.get("evaluation_dataset", {}).get("display_name", ""))).lower()
                if search and search not in haystack:
                    continue
                if status and item.get("status") != status:
                    continue
                if modality and item.get("modality") != modality:
                    continue
                if workload and workload not in context.values():
                    continue
                if related_model_id:
                    related = any(
                        selected.get("model_id") == related_model_id
                        and (selected.get("artifact_identity") or {}).get("artifact_id")
                        == related_artifact_id
                        and (selected.get("artifact_identity") or {}).get(
                            "artifact_manifest_sha256"
                        )
                        == related_artifact_hash
                        for selected in item.get("selected_models", [])
                        if isinstance(selected, dict)
                    )
                    if not related:
                        continue
                items.append(public_comparison(item, summary=True))
            items.sort(key=lambda item: (item.get("created_at", ""), item["comparison_id"]), reverse=True)
            total = len(items)
            start = (page - 1) * page_size
            return jsonify({
                "items": items[start:start + page_size], "page": page, "page_size": page_size,
                "total": total, "pages": math.ceil(total / page_size) if total else 0,
                "skipped_corrupt_records": skipped,
            })
        except (ComparisonValidationError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @blueprint.get("/evaluation-datasets")
    def evaluation_datasets():
        try:
            payload = _catalogue_get("/datasets", {"page": 1, "page_size": 100, "status": "available"})
            items = []
            for summary in payload.get("items", []):
                try:
                    item = _dataset_context(summary["dataset_id"], summary.get("version"))
                except (ComparisonValidationError, ValueError):
                    continue
                items.append(item)
            items.sort(key=lambda item: (not item["ground_truth_available"], item["display_name"].lower()))
            return jsonify({"items": items, "total": len(items)})
        except ComparisonValidationError as exc:
            return jsonify({"error": str(exc)}), 503

    @blueprint.get("/compatible-models")
    def compatible_models():
        try:
            dataset_id = request.args.get("dataset_id")
            version = request.args.get("version")
            parsed_version = int(version) if version is not None else None
            dataset = _dataset_context(dataset_id, parsed_version)
            partition_id = request.args.get("partition_id")
            if partition_id and partition_id != dataset["partition_id"]:
                raise ComparisonValidationError("selected partition is not active for this version")
            _store, model_store = stores()
            records, _corrupt = model_store.records()
            items = []
            for record in records:
                compatible, reasons = _model_compatibility(record, dataset)
                snapshot = _selected_snapshot(record) if compatible else {
                    "model_id": record["model_id"],
                    "display_name": record.get("model_metadata", {}).get("display_name") or record["model_id"],
                    "detector_id": record["detector_id"],
                    "detector_version": record["detector_version"],
                    "model_created_at": record["created_at"],
                    "training_dataset": deepcopy(record.get("dataset", {})),
                    "stored_evaluation": stored_evaluation_projection(record),
                }
                snapshot.update(compatible=compatible, compatibility_reasons=reasons)
                items.append(snapshot)
            items.sort(key=lambda item: (not item["compatible"], item["display_name"].lower()))
            return jsonify({"dataset": dataset, "items": items, "total": len(items)})
        except (ComparisonValidationError, ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @blueprint.post("")
    def create_comparison():
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415
        try:
            payload = request.get_json(silent=False)
            if not isinstance(payload, dict):
                raise ComparisonValidationError("request body must be an object")
            name = payload.get("name")
            if not isinstance(name, str) or not name.strip() or len(name.strip()) > 200:
                raise ComparisonValidationError("name must contain 1 to 200 characters")
            requested_dataset = payload.get("evaluation_dataset")
            if not isinstance(requested_dataset, dict):
                raise ComparisonValidationError("evaluation_dataset must be an object")
            dataset = _dataset_context(requested_dataset.get("dataset_id"), requested_dataset.get("version"))
            if requested_dataset.get("partition_id") != dataset["partition_id"]:
                raise ComparisonValidationError("evaluation partition is unknown or no longer active")
            model_ids = payload.get("model_ids")
            if not isinstance(model_ids, list) or len(model_ids) < 2:
                raise ComparisonValidationError("at least two model IDs are required")
            if len(model_ids) > 10 or len(set(model_ids)) != len(model_ids):
                raise ComparisonValidationError("model IDs must be unique and limited to 10")
            client_request_id = payload.get("client_request_id")
            if not isinstance(client_request_id, str) or not client_request_id or len(client_request_id) > 200:
                raise ComparisonValidationError("client_request_id is required and must be at most 200 characters")
            store, model_store = stores()
            duplicate = store.find_client_request(client_request_id)
            if duplicate is not None:
                return jsonify(public_comparison(duplicate)), 200
            selected = []
            for model_id in model_ids:
                validate_identifier(model_id, "model_id")
                try:
                    model = model_store.read(model_id)
                except ModelNotFoundError as exc:
                    raise ComparisonValidationError(f"unknown Saved Model: {model_id!r}") from exc
                compatible, reasons = _model_compatibility(model, dataset)
                if not compatible:
                    raise ComparisonCompatibilityError(
                        f"Saved Model {model_id!r} is incompatible: {'; '.join(reasons)}"
                    )
                selected.append(_selected_snapshot(model))
            created = utc_now()
            comparison_id = f"comparison-{uuid.uuid4().hex}"
            record = {
                "schema_version": 1,
                "comparison_id": comparison_id,
                "name": name.strip(),
                "analysis_type": "anomaly_detection",
                "configuration_version": "cloudsentinel.comparison/v1",
                "status": "queued",
                "created_at": created,
                "started_at": None,
                "updated_at": created,
                "completed_at": None,
                "client_request_id": client_request_id,
                "evaluation_dataset": dataset,
                "modality": dataset["modality"],
                "feature_identity": {
                    "feature_order": dataset["feature_order"],
                    "sha256": dataset["feature_order_sha256"],
                },
                "ground_truth_available": dataset["ground_truth_available"],
                "known_incident_window": (dataset["known_incident_windows"] or [None])[0],
                "known_incident_windows": deepcopy(dataset["known_incident_windows"]),
                "selected_models": selected,
                "results": [
                    {
                        "model_id": item["model_id"],
                        "display_name": item["display_name"],
                        "detector_id": item["detector_id"],
                        "detector_version": item["detector_version"],
                        "artifact_identity": deepcopy(item["artifact_identity"]),
                        "status": "queued",
                        "started_at": None,
                        "completed_at": None,
                        "safe_error_summary": None,
                    }
                    for item in selected
                ],
                "result_reference": None,
                "safe_failure_summary": None,
                "deterministic_summary": None,
            }
            stored, created_new = store.create_idempotent(record)
            if not created_new:
                return jsonify(public_comparison(stored)), 200
            task_id = f"comparison-task-{uuid.uuid4().hex}"
            try:
                comparison_task.apply_async(args=[comparison_id], task_id=task_id)
            except Exception as exc:
                store.update(comparison_id, lambda item: item.update(
                    status="failed", updated_at=utc_now(), completed_at=utc_now(),
                    safe_failure_summary="comparison task could not be dispatched",
                ))
                return jsonify({"error": "comparison task could not be dispatched"}), 503
            return jsonify(public_comparison(store.read(comparison_id))), 202
        except ComparisonCompatibilityError as exc:
            return jsonify({"error": str(exc)}), 422
        except (BadRequest, ComparisonValidationError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @blueprint.get("/<comparison_id>")
    def get_comparison(comparison_id: str):
        try:
            store, _models = stores()
            return jsonify(public_comparison(store.read(comparison_id)))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except ComparisonNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @blueprint.get("/<comparison_id>/robustness")
    def comparison_robustness(comparison_id: str):
        try:
            store, _models = stores()
            current = store.read(comparison_id)
            filters = {
                "workload": request.args.get("workload", "").strip(),
                "scenario": request.args.get("scenario", "").strip(),
                "labelled_only": _boolean_query(
                    request.args.get("labelled_only"), "labelled_only"
                ),
                "shared_only": _boolean_query(
                    request.args.get("shared_only"), "shared_only"
                ),
            }
            filters = {key: value for key, value in filters.items() if value}
            if filters.get("workload") not in {
                None, "Low", "Normal", "High", "Variable"
            }:
                raise ComparisonValidationError("unsupported workload filter")
            if filters.get("scenario") not in {
                None,
                "Normal operation",
                "CPU stress",
                "Memory stress",
                "Network delay",
                "Unknown",
            }:
                raise ComparisonValidationError("unsupported scenario filter")
            return jsonify(
                aggregate_robustness(
                    store,
                    current,
                    filters=filters,
                    maximum_contexts_per_model=current_app.config.get(
                        "COMPARISON_ROBUSTNESS_MAX_CONTEXTS_PER_MODEL", 200
                    ),
                )
            )
        except ComparisonNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except (ComparisonValidationError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @blueprint.get("/<comparison_id>/status")
    def comparison_status(comparison_id: str):
        try:
            store, _models = stores()
            record = store.read(comparison_id)
            return jsonify({
                "comparison_id": record["comparison_id"],
                "status": record["status"],
                "updated_at": record.get("updated_at"),
                "completed_at": record.get("completed_at"),
                "safe_failure_summary": record.get("safe_failure_summary"),
                "results": [
                    {
                        key: item.get(key)
                        for key in ("model_id", "display_name", "status", "safe_error_summary")
                    }
                    for item in record.get("results", [])
                ],
            })
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except ComparisonNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    return blueprint


def configure_comparison_defaults(app: Any) -> None:
    app.config.setdefault(
        "COMPARISON_STORAGE_ROOT",
        os.getenv(
            "COMPARISON_STORAGE_ROOT",
            str(Path(app.config["TRAINING_RUN_STORAGE_ROOT"]).parent / "comparisons"),
        ),
    )
    app.config.setdefault(
        "API_GENERIC_ANOMALY_DETECTION_URL",
        os.getenv("API_GENERIC_ANOMALY_DETECTION_URL", "http://127.0.0.1:5015"),
    )
    app.config.setdefault("COMPARISON_MAX_BUNDLE_BYTES", 250 * 1024 * 1024)
    app.config.setdefault("COMPARISON_MAX_TIMELINE_POINTS", 1000)
    robustness_limit = int(
        os.getenv("COMPARISON_ROBUSTNESS_MAX_CONTEXTS_PER_MODEL", "200")
    )
    if robustness_limit < 1 or robustness_limit > 1_000:
        raise ValueError(
            "COMPARISON_ROBUSTNESS_MAX_CONTEXTS_PER_MODEL must be between 1 and 1000"
        )
    app.config.setdefault(
        "COMPARISON_ROBUSTNESS_MAX_CONTEXTS_PER_MODEL", robustness_limit
    )
