"""Path-safe, atomic filesystem repository for Data Catalogue metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .validation import CatalogueValidationError, validate_identifier, validate_version


class DatasetNotFoundError(LookupError):
    """Raised when a dataset record or version does not exist."""


class DatasetAlreadyExistsError(FileExistsError):
    """Raised when a dataset identifier has already been registered."""


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


class FileCatalogueRepository:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.datasets_root = (self.root / "datasets").resolve()

    def _dataset_directory(self, dataset_id: str) -> Path:
        dataset_id = validate_identifier(dataset_id)
        path = (self.datasets_root / dataset_id).resolve()
        if not path.is_relative_to(self.datasets_root):
            raise CatalogueValidationError("dataset_id resolves outside catalogue storage")
        return path

    def _version_path(self, dataset_id: str, version: int) -> Path:
        version = validate_version(version)
        directory = self._dataset_directory(dataset_id)
        path = (directory / "versions" / f"{version:06d}.json").resolve()
        if not path.is_relative_to(directory):
            raise CatalogueValidationError("version resolves outside dataset storage")
        return path

    def _safe_child(self, dataset_id: str, *parts: str) -> Path:
        directory = self._dataset_directory(dataset_id)
        path = directory.joinpath(*parts).resolve()
        if not path.is_relative_to(directory):
            raise CatalogueValidationError("catalogue path resolves outside dataset storage")
        return path

    def artifact_directory(self, dataset_id: str, version: int) -> Path:
        version = validate_version(version)
        return self._safe_child(dataset_id, "artifacts", f"{version:06d}")

    def staging_directory(self, dataset_id: str, version: int, attempt_id: str) -> Path:
        version = validate_version(version)
        attempt_id = validate_identifier(attempt_id, "attempt_id")
        return self._safe_child(
            dataset_id, ".staging", f"{version:06d}-{attempt_id}"
        )

    def attempt_path(self, dataset_id: str, version: int, attempt_id: str) -> Path:
        version = validate_version(version)
        attempt_id = validate_identifier(attempt_id, "attempt_id")
        return self._safe_child(
            dataset_id, "fetch_attempts", f"{version:06d}", f"{attempt_id}.json"
        )

    def create(self, record: dict[str, Any], version: dict[str, Any]) -> None:
        dataset_id = validate_identifier(record.get("dataset_id"))
        if version.get("dataset_id") != dataset_id:
            raise CatalogueValidationError("dataset and version identities do not match")
        validate_version(version.get("version"))
        self.datasets_root.mkdir(parents=True, exist_ok=True)
        final_directory = self._dataset_directory(dataset_id)
        if final_directory.exists():
            raise DatasetAlreadyExistsError(f"dataset already exists: {dataset_id!r}")
        staging_directory = Path(
            tempfile.mkdtemp(prefix=f".{dataset_id}.", dir=self.datasets_root)
        )
        try:
            atomic_write_json(staging_directory / "dataset.json", record)
            atomic_write_json(
                staging_directory / "versions" / f"{version['version']:06d}.json",
                version,
            )
            os.replace(staging_directory, final_directory)
        except Exception:
            shutil.rmtree(staging_directory, ignore_errors=True)
            raise

    def read_record(self, dataset_id: str) -> dict[str, Any]:
        path = self._dataset_directory(dataset_id) / "dataset.json"
        return self._read_json(path, dataset_id)

    def read_version(self, dataset_id: str, version: int) -> dict[str, Any]:
        return self._read_json(self._version_path(dataset_id, version), dataset_id)

    def read_detail(self, dataset_id: str) -> dict[str, Any]:
        record = self.read_record(dataset_id)
        version = self.read_version(dataset_id, record["latest_version"])
        return {**record, "version": version}

    def write_attempt(
        self, dataset_id: str, version: int, attempt_id: str, value: dict[str, Any]
    ) -> None:
        atomic_write_json(self.attempt_path(dataset_id, version, attempt_id), value)

    def read_attempt(self, dataset_id: str, version: int, attempt_id: str) -> dict[str, Any]:
        return self._read_json(
            self.attempt_path(dataset_id, version, attempt_id), dataset_id
        )

    def update_version(self, dataset_id: str, version_number: int, version: dict[str, Any]) -> None:
        if version.get("dataset_id") != validate_identifier(dataset_id):
            raise CatalogueValidationError("dataset and version identities do not match")
        if version.get("version") != validate_version(version_number):
            raise CatalogueValidationError("version identities do not match")
        atomic_write_json(self._version_path(dataset_id, version_number), version)

    def update_record(self, dataset_id: str, record: dict[str, Any]) -> None:
        if record.get("dataset_id") != validate_identifier(dataset_id):
            raise CatalogueValidationError("dataset record identity does not match")
        atomic_write_json(self._dataset_directory(dataset_id) / "dataset.json", record)

    def promote_staging(self, dataset_id: str, version: int, attempt_id: str) -> Path:
        staging = self.staging_directory(dataset_id, version, attempt_id)
        active = self.artifact_directory(dataset_id, version)
        if not staging.is_dir():
            raise RuntimeError("catalogue staging directory is missing")
        if active.exists():
            raise RuntimeError("catalogue version already has active artifacts")
        active.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, active)
        return active

    def list_records(self) -> list[dict[str, Any]]:
        if not self.datasets_root.is_dir():
            return []
        records = []
        for path in sorted(self.datasets_root.iterdir()):
            resolved = path.resolve()
            if not resolved.is_relative_to(self.datasets_root) or not resolved.is_dir():
                continue
            try:
                records.append(self.read_record(path.name))
            except (CatalogueValidationError, DatasetNotFoundError):
                continue
        return records

    @staticmethod
    def _read_json(path: Path, dataset_id: str) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DatasetNotFoundError(f"unknown dataset: {dataset_id!r}") from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"could not read dataset {dataset_id!r}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"dataset {dataset_id!r} metadata is not a JSON object")
        return value
