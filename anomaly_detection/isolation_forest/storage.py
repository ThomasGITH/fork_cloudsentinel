"""Safe local artifact staging and validation for Isolation Forest models."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, BinaryIO

from detectors.isolation_forest.training_request import (
    IsolationForestRequestError,
    validate_model_id,
)


DETECTOR_ID = "isolation-forest"
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "model_metadata.json"
EVALUATION_FILENAME = "model_evaluation.json"
ARTIFACT_FILENAMES = (MODEL_FILENAME, METADATA_FILENAME, EVALUATION_FILENAME)
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class ArtifactValidationError(ValueError):
    """Raised when uploaded or stored artifacts violate the service contract."""


class ArtifactConflictError(ArtifactValidationError):
    """Raised when an active model ID already refers to different content."""


def safe_model_id(value: Any) -> str:
    try:
        return validate_model_id(value)
    except IsolationForestRequestError as exc:
        raise ArtifactValidationError(str(exc)) from exc


def validate_checksum(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ArtifactValidationError(
            f"{field_name} must be a 64-character hexadecimal SHA-256 digest"
        )
    return value.lower()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_and_hash(source: BinaryIO, destination: Path) -> str:
    digest = hashlib.sha256()
    with destination.open("xb") as target:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            target.write(block)
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path, artifact_name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"{artifact_name} must contain valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{artifact_name} must contain a JSON object")
    return value


def validate_metadata(metadata: Any, model_id: str) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ArtifactValidationError("model_metadata.json must contain a JSON object")
    if metadata.get("schema_version") != 1:
        raise ArtifactValidationError("metadata schema_version must be 1")
    if metadata.get("detector_id") != DETECTOR_ID:
        raise ArtifactValidationError(
            "metadata detector_id must be 'isolation-forest'"
        )
    if metadata.get("model_id") != model_id:
        raise ArtifactValidationError(
            "metadata model_id must match the requested model_id"
        )
    if metadata.get("artifact_format") != "joblib":
        raise ArtifactValidationError("metadata artifact_format must be 'joblib'")

    n_features = metadata.get("n_features")
    if not isinstance(n_features, int) or isinstance(n_features, bool) or n_features < 1:
        raise ArtifactValidationError("metadata n_features must be a positive integer")
    if metadata.get("feature_identity") != "positional":
        raise ArtifactValidationError(
            "metadata feature_identity must be 'positional'"
        )

    preprocessing = metadata.get("preprocessing")
    if not isinstance(preprocessing, (str, dict)) or not preprocessing:
        raise ArtifactValidationError(
            "metadata preprocessing must describe the persisted preprocessing"
        )
    if isinstance(preprocessing, str) and "StandardScaler" not in preprocessing:
        raise ArtifactValidationError(
            "metadata preprocessing must identify the persisted StandardScaler"
        )

    score_semantics = metadata.get("score_semantics")
    if not isinstance(score_semantics, dict):
        raise ArtifactValidationError("metadata score_semantics must be an object")
    if score_semantics.get("higher_is_more_anomalous") is not True:
        raise ArtifactValidationError(
            "metadata score_semantics must declare higher_is_more_anomalous as true"
        )
    threshold = score_semantics.get("binary_threshold")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or float(threshold) != 0.0
    ):
        raise ArtifactValidationError(
            "metadata score_semantics binary_threshold must be 0.0"
        )
    return metadata


def model_directory(model_storage_root: str | os.PathLike[str], model_id: str) -> Path:
    model_id = safe_model_id(model_id)
    return Path(model_storage_root) / DETECTOR_ID / model_id


def validate_active_artifacts(
    model_storage_root: str | os.PathLike[str], model_id: str
) -> tuple[Path, dict[str, Any]]:
    directory = model_directory(model_storage_root, model_id)
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError(f"unknown Isolation Forest model: {model_id}")
    for filename in ARTIFACT_FILENAMES:
        path = directory / filename
        if not path.is_file() or path.is_symlink():
            raise ArtifactValidationError(
                f"stored model is incomplete: missing {filename}"
            )
    metadata = validate_metadata(
        read_json_object(directory / METADATA_FILENAME, METADATA_FILENAME),
        model_id,
    )
    read_json_object(directory / EVALUATION_FILENAME, EVALUATION_FILENAME)
    return directory, metadata


def _existing_hashes(directory: Path) -> dict[str, str] | None:
    if not directory.exists():
        return None
    if not directory.is_dir() or directory.is_symlink():
        raise ArtifactConflictError("the requested model_id already exists in an unsafe form")
    hashes: dict[str, str] = {}
    for filename in ARTIFACT_FILENAMES:
        path = directory / filename
        if not path.is_file() or path.is_symlink():
            raise ArtifactConflictError(
                "the requested model_id already exists with incomplete artifacts"
            )
        hashes[filename] = file_sha256(path)
    return hashes


def install_artifacts(
    model_storage_root: str | os.PathLike[str],
    model_id: str,
    uploads: dict[str, BinaryIO],
    expected_hashes: dict[str, str],
) -> bool:
    """Validate and atomically install artifacts; return True for an idempotent replay."""
    model_id = safe_model_id(model_id)
    detector_root = Path(model_storage_root) / DETECTOR_ID
    staging_root = detector_root / ".staging"
    detector_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    final_directory = detector_root / model_id
    staging_directory = Path(
        tempfile.mkdtemp(prefix=f"{model_id}-", dir=staging_root)
    )

    try:
        actual_hashes: dict[str, str] = {}
        for filename in ARTIFACT_FILENAMES:
            actual_hashes[filename] = copy_and_hash(
                uploads[filename], staging_directory / filename
            )
            if actual_hashes[filename] != expected_hashes[filename]:
                raise ArtifactValidationError(
                    f"SHA-256 checksum mismatch for {filename}"
                )

        metadata = read_json_object(
            staging_directory / METADATA_FILENAME, METADATA_FILENAME
        )
        validate_metadata(metadata, model_id)
        read_json_object(staging_directory / EVALUATION_FILENAME, EVALUATION_FILENAME)

        existing_hashes = _existing_hashes(final_directory)
        if existing_hashes is not None:
            if existing_hashes == actual_hashes:
                return True
            raise ArtifactConflictError(
                "the requested model_id already exists with different artifacts"
            )

        try:
            staging_directory.rename(final_directory)
        except OSError:
            if not final_directory.exists():
                raise
            existing_hashes = _existing_hashes(final_directory)
            if existing_hashes == actual_hashes:
                return True
            raise ArtifactConflictError(
                "the requested model_id was installed concurrently with different artifacts"
            )
        return False
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory)
        try:
            staging_root.rmdir()
        except OSError:
            pass
