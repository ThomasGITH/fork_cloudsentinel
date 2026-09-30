"""Resolve safe model records, verified artifacts and pinned adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from detectors import registry
from detectors.contracts import ModelLoadContext
from detectors.model_artifacts import ModelArtifactError, ModelArtifactStore
from learning_adaptation.model_catalogue import (
    ModelCatalogueStore,
    ModelNotFoundError,
    ModelRecordError,
)


class ModelNotReadyError(RuntimeError):
    pass


class GenericModelRepository:
    def __init__(self, model_root: str | Path, artifact_root: str | Path):
        self.models = ModelCatalogueStore(model_root)
        self.artifacts = ModelArtifactStore(artifact_root)

    def record(self, model_id: str) -> dict[str, Any]:
        return self.models.read(model_id)

    def prepare(self, model_id: str) -> tuple[dict[str, Any], dict[str, Any], Any, ModelLoadContext]:
        record = self.record(model_id)
        inference = record.get("inference") or {"status": "legacy_external"}
        if inference.get("status") != "ready":
            raise ModelNotReadyError(
                f"model is not ready for generic inference ({inference.get('status')})"
            )
        verified = self.artifacts.verify(model_id)
        if verified["manifest_sha256"] != inference.get("artifact_manifest_sha256"):
            raise ModelArtifactError("artifact manifest checksum mismatch")
        manifest = verified["manifest"]
        if (
            manifest.get("detector_id") != record["detector_id"]
            or manifest.get("detector_version") != record["detector_version"]
            or manifest.get("artifact_format") != inference.get("artifact_format")
        ):
            raise ModelArtifactError("artifact and model catalogue metadata disagree")
        adapter = registry.get_inference_adapter(
            record["detector_id"], plugin_reference=record.get("plugin")
        )
        context = ModelLoadContext(
            model_id=model_id,
            detector_id=record["detector_id"],
            detector_version=record["detector_version"],
            artifact_directory=verified["directory"] / "files",
            artifact_manifest=manifest,
            model_record=record,
            feature_identity=record["feature_identity"],
        )
        return record, verified, adapter, context
