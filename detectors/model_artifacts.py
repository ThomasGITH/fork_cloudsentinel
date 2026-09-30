"""Immutable, checksummed model artifacts shared by training and inference."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from .contracts import ArtifactResult


class ModelArtifactError(ValueError):
    """An artifact cannot be published or verified safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


class ModelArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _directory(self, model_id: str) -> Path:
        if not isinstance(model_id, str) or not model_id or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in model_id
        ) or model_id in {".", ".."}:
            raise ModelArtifactError("model_id is unsafe")
        path = (self.root / model_id).resolve()
        if not path.is_relative_to(self.root):
            raise ModelArtifactError("model artifact resolves outside storage")
        return path

    def publish(
        self,
        model_id: str,
        artifact: ArtifactResult,
        *,
        detector_id: str,
        detector_version: str,
        artifact_format: str,
        feature_identity: Mapping[str, Any],
        plugin_reference: Mapping[str, Any] | None,
        inference_contract: str,
    ) -> dict[str, Any]:
        source = artifact.directory.resolve()
        if not source.is_dir() or source.is_symlink():
            raise ModelArtifactError("source artifact directory is unavailable")
        if not artifact.required_files:
            raise ModelArtifactError("artifact has no required files")
        final = self._directory(model_id)
        self.root.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{model_id}.", dir=self.root))
        try:
            files_directory = stage / "files"
            files_directory.mkdir()
            files = []
            for filename in artifact.required_files:
                if not isinstance(filename, str) or Path(filename).name != filename:
                    raise ModelArtifactError("artifact filename is unsafe")
                source_candidate = source / filename
                if source_candidate.is_symlink():
                    raise ModelArtifactError(
                        f"required artifact file {filename!r} is unavailable"
                    )
                source_file = source_candidate.resolve()
                if (
                    not source_file.is_relative_to(source)
                    or not source_file.is_file()
                ):
                    raise ModelArtifactError(
                        f"required artifact file {filename!r} is unavailable"
                    )
                destination = files_directory / filename
                shutil.copyfile(source_file, destination)
                files.append(
                    {
                        "name": filename,
                        "sha256": sha256_file(destination),
                        "bytes": destination.stat().st_size,
                    }
                )
            manifest = {
                "schema_version": 1,
                "artifact_id": f"artifact_{model_id}",
                "model_id": model_id,
                "detector_id": detector_id,
                "detector_version": detector_version,
                "artifact_format": artifact_format,
                "inference_contract": inference_contract,
                "feature_identity": dict(feature_identity),
                "plugin": dict(plugin_reference) if plugin_reference else None,
                "files": files,
            }
            manifest_path = stage / "artifact_manifest.json"
            manifest_path.write_bytes(_canonical_json(manifest) + b"\n")
            manifest_sha256 = sha256_file(manifest_path)
            if final.exists():
                existing = self.verify(model_id)
                if existing["manifest_sha256"] != manifest_sha256:
                    raise ModelArtifactError(
                        "immutable model artifact already exists with different content"
                    )
                return existing
            os.replace(stage, final)
            for path in final.rglob("*"):
                path.chmod(0o555 if path.is_dir() else 0o444)
            final.chmod(0o555)
            return {
                "artifact_id": manifest["artifact_id"],
                "manifest_sha256": manifest_sha256,
                "manifest": manifest,
                "directory": final,
            }
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)

    def verify(self, model_id: str) -> dict[str, Any]:
        directory = self._directory(model_id)
        manifest_path = directory / "artifact_manifest.json"
        if manifest_path.is_symlink():
            raise ModelArtifactError("artifact manifest is unavailable")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelArtifactError("artifact manifest is unavailable") from exc
        if not isinstance(manifest, dict) or manifest.get("model_id") != model_id:
            raise ModelArtifactError("artifact manifest identity mismatch")
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ModelArtifactError("artifact manifest contains no files")
        for record in files:
            if not isinstance(record, dict) or Path(str(record.get("name"))).name != record.get("name"):
                raise ModelArtifactError("artifact manifest filename is unsafe")
            candidate = directory / "files" / record["name"]
            if candidate.is_symlink():
                raise ModelArtifactError("required artifact file is unavailable")
            path = candidate.resolve()
            if (
                not path.is_relative_to((directory / "files").resolve())
                or not path.is_file()
            ):
                raise ModelArtifactError("required artifact file is unavailable")
            if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
                raise ModelArtifactError("artifact file integrity mismatch")
        return {
            "artifact_id": manifest.get("artifact_id"),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest": manifest,
            "directory": directory,
        }
