"""File-backed, artifact-independent metadata records for trained models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from learning_adaptation.training_run_storage import (
        TrainingRunValidationError,
        atomic_write_json,
        validate_identifier,
    )
except ModuleNotFoundError:  # Service image copies modules beside app.py.
    from training_run_storage import (
        TrainingRunValidationError,
        atomic_write_json,
        validate_identifier,
    )


class ModelNotFoundError(LookupError):
    """Raised when a model metadata record does not exist."""


class ModelRecordError(ValueError):
    """Raised when model metadata is unsafe or incomplete."""


MODEL_STATUSES = {"available"}
PROMOTION_STATUSES = {
    "not_promoted",
    "promoted",
    "promotion_failed",
    "not_applicable",
}


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelRecordError(f"{field} must be an object")
    return value


def validate_model_record(record: Any) -> dict[str, Any]:
    record = _object(record, "model record")
    required = {
        "schema_version",
        "model_id",
        "detector_id",
        "detector_version",
        "status",
        "created_at",
        "updated_at",
        "training_run_id",
        "dataset",
        "snapshot",
        "feature_identity",
        "training_parameters",
        "evaluation",
        "promotion",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ModelRecordError("model record is missing: " + ", ".join(missing))
    if record["schema_version"] != 1:
        raise ModelRecordError("model record schema_version must be 1")
    for field in ("model_id", "detector_id", "training_run_id"):
        try:
            validate_identifier(record[field], field)
        except TrainingRunValidationError as exc:
            raise ModelRecordError(str(exc)) from exc
    if not isinstance(record["detector_version"], str) or not record["detector_version"]:
        raise ModelRecordError("detector_version must be a non-empty string")
    if record["status"] not in MODEL_STATUSES:
        raise ModelRecordError("unsupported model status")
    for field in ("created_at", "updated_at"):
        if not isinstance(record[field], str) or not record[field]:
            raise ModelRecordError(f"{field} must be a non-empty timestamp")
    for field in (
        "dataset",
        "snapshot",
        "feature_identity",
        "training_parameters",
        "evaluation",
        "promotion",
    ):
        _object(record[field], field)
    if "artifact" in record:
        artifact = _object(record["artifact"], "artifact")
        if not isinstance(artifact.get("format"), str) or not artifact["format"]:
            raise ModelRecordError("artifact.format must be a non-empty string")
        safe_artifact_reference = artifact.get("safe_reference")
        if safe_artifact_reference is not None:
            try:
                validate_identifier(safe_artifact_reference, "artifact.safe_reference")
            except TrainingRunValidationError as exc:
                raise ModelRecordError(str(exc)) from exc
    if "model_metadata" in record:
        _object(record["model_metadata"], "model_metadata")
    if "plugin" in record:
        plugin = _object(record["plugin"], "plugin")
        required_plugin_fields = {
            "source",
            "detector_id",
            "detector_version",
            "package_sha256",
            "manifest_sha256",
            "runtime_profile",
        }
        if not required_plugin_fields.issubset(plugin):
            raise ModelRecordError("plugin reference is incomplete")
        if plugin["source"] not in {"builtin", "external"}:
            raise ModelRecordError("plugin source is invalid")
        if plugin["detector_id"] != record["detector_id"] or plugin[
            "detector_version"
        ] != record["detector_version"]:
            raise ModelRecordError("plugin reference does not match model detector")
        for field in ("package_sha256", "manifest_sha256"):
            value = plugin[field]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ModelRecordError(f"plugin.{field} is invalid")
        if not isinstance(plugin["runtime_profile"], str) or not plugin[
            "runtime_profile"
        ]:
            raise ModelRecordError("plugin.runtime_profile is invalid")
    if record["promotion"].get("status") not in PROMOTION_STATUSES:
        raise ModelRecordError("unsupported promotion status")
    safe_reference = record["promotion"].get("safe_reference")
    if safe_reference is not None:
        try:
            validate_identifier(safe_reference, "promotion.safe_reference")
        except TrainingRunValidationError as exc:
            raise ModelRecordError(str(exc)) from exc
    return record


class ModelCatalogueStore:
    """Persist safe model metadata without depending on runtime artifact paths."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def path(self, model_id: str) -> Path:
        try:
            model_id = validate_identifier(model_id, "model_id")
        except TrainingRunValidationError as exc:
            raise ModelRecordError(str(exc)) from exc
        path = (self.root / f"{model_id}.json").resolve()
        if not path.is_relative_to(self.root):
            raise ModelRecordError("model_id resolves outside the model catalogue")
        return path

    def write(self, record: dict[str, Any]) -> dict[str, Any]:
        record = validate_model_record(record)
        path = self.path(record["model_id"])
        if path.is_file():
            existing = self.read(record["model_id"])
            if existing != record:
                raise ModelRecordError(
                    f"model record {record['model_id']!r} already exists with different metadata"
                )
            return existing
        atomic_write_json(path, record)
        return record

    def replace(self, record: dict[str, Any]) -> dict[str, Any]:
        record = validate_model_record(record)
        atomic_write_json(self.path(record["model_id"]), record)
        return record

    def read(self, model_id: str) -> dict[str, Any]:
        path = self.path(model_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ModelNotFoundError(f"unknown model: {model_id!r}") from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelRecordError(f"model record {model_id!r} is invalid: {exc}") from exc
        return validate_model_record(value)

    def records(self) -> tuple[list[dict[str, Any]], int]:
        if not self.root.is_dir():
            return [], 0
        records = []
        corrupt = 0
        for path in sorted(self.root.glob("*.json")):
            try:
                records.append(self.read(path.stem))
            except (ModelNotFoundError, ModelRecordError):
                corrupt += 1
        return records, corrupt
