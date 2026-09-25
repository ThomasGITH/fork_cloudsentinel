"""Local filesystem storage for immutable dataset snapshots and training runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
import uuid


SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DATASET_FILENAMES = {
    "train": "train_{dataset_id}",
    "test": "test_{dataset_id}",
    "labels": "test_label_{dataset_id}",
}


class TrainingRunValidationError(ValueError):
    """Raised when a training-run request or local dataset is invalid."""


class TrainingRunNotFoundError(LookupError):
    """Raised when a requested training run does not exist."""


def validate_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrainingRunValidationError(f"{field} must be a non-empty string")
    if value in {".", ".."} or not SAFE_IDENTIFIER.fullmatch(value):
        raise TrainingRunValidationError(
            f"{field} may contain only letters, digits, '-', '_' and '.' and may not contain paths"
        )
    return value


def new_identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


class DatasetSnapshotStore:
    """Create immutable local copies of repository datasets."""

    def __init__(self, snapshot_root: str | Path, dataset_root: str | Path):
        self.snapshot_root = Path(snapshot_root).resolve()
        self.dataset_root = Path(dataset_root).resolve()

    def _dataset_directory(self, dataset_id: str) -> Path:
        dataset_id = validate_identifier(dataset_id, "dataset.dataset_id")
        directory = (self.dataset_root / dataset_id).resolve()
        if not directory.is_relative_to(self.dataset_root) or not directory.is_dir():
            raise TrainingRunValidationError(
                f"unknown existing dataset: {dataset_id!r}"
            )
        return directory

    def validate_source(self, dataset_id: str) -> dict[str, Any]:
        directory = self._dataset_directory(dataset_id)
        source_files: dict[str, Path] = {}
        for logical_name, template in DATASET_FILENAMES.items():
            path = (directory / template.format(dataset_id=dataset_id)).resolve()
            if not path.is_relative_to(directory) or not path.is_file():
                raise TrainingRunValidationError(
                    f"dataset {dataset_id!r} is missing {logical_name} data"
                )
            source_files[logical_name] = path

        details_path = (directory / "details.json").resolve()
        if not details_path.is_relative_to(directory) or not details_path.is_file():
            raise TrainingRunValidationError(
                f"dataset {dataset_id!r} is missing details.json"
            )
        try:
            details = json.loads(details_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TrainingRunValidationError(
                f"dataset {dataset_id!r} has invalid details.json: {exc}"
            ) from exc
        if not isinstance(details, dict):
            raise TrainingRunValidationError("dataset details must be a JSON object")
        return {"directory": directory, "files": source_files, "details": details}

    def create(self, dataset_id: str, source: dict[str, Any] | None = None) -> dict[str, Any]:
        source = source or self.validate_source(dataset_id)
        snapshot_id = new_identifier("snapshot")
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        final_directory = self.snapshot_root / snapshot_id
        staging_directory = Path(
            tempfile.mkdtemp(prefix=f".{snapshot_id}.", dir=self.snapshot_root)
        )
        try:
            file_metadata = {}
            for logical_name, destination_name in (
                ("train", "train.csv"),
                ("test", "test.csv"),
                ("labels", "labels.csv"),
            ):
                destination = staging_directory / destination_name
                shutil.copyfile(source["files"][logical_name], destination)
                file_metadata[logical_name] = {
                    "filename": destination_name,
                    "sha256": sha256_file(destination),
                    "bytes": destination.stat().st_size,
                }

            combined_digest = hashlib.sha256()
            for logical_name in ("train", "test", "labels"):
                combined_digest.update(file_metadata[logical_name]["sha256"].encode("ascii"))

            metadata = {
                "schema_version": 1,
                "snapshot_id": snapshot_id,
                "source": {"type": "existing", "dataset_id": dataset_id},
                "sha256": combined_digest.hexdigest(),
                "files": file_metadata,
                "dataset_details": source["details"],
            }
            atomic_write_json(staging_directory / "snapshot.json", metadata)
            os.replace(staging_directory, final_directory)
            for path in final_directory.iterdir():
                if path.is_file():
                    path.chmod(0o444)
            return {**metadata, "directory": final_directory}
        except Exception:
            shutil.rmtree(staging_directory, ignore_errors=True)
            raise

    def remove(self, snapshot_id: str) -> None:
        snapshot_id = validate_identifier(snapshot_id, "snapshot_id")
        directory = (self.snapshot_root / snapshot_id).resolve()
        if directory.is_relative_to(self.snapshot_root):
            shutil.rmtree(directory, ignore_errors=True)


class TrainingRunStore:
    """Atomically persist compact run metadata and detector child inputs."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def directory(self, run_id: str) -> Path:
        run_id = validate_identifier(run_id, "run_id")
        directory = (self.root / run_id).resolve()
        if not directory.is_relative_to(self.root):
            raise TrainingRunValidationError("run_id resolves outside the run storage root")
        return directory

    def child_input_directory(self, run_id: str, detector_id: str) -> Path:
        detector_id = validate_identifier(detector_id, "detector_id")
        directory = self.directory(run_id) / "children" / detector_id / "input"
        directory.mkdir(parents=True, exist_ok=False)
        return directory

    def write(self, run: dict[str, Any]) -> None:
        atomic_write_json(self.directory(run["run_id"]) / "run.json", run)

    def read(self, run_id: str) -> dict[str, Any]:
        path = self.directory(run_id) / "run.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise TrainingRunNotFoundError(f"unknown training run: {run_id!r}") from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"could not read training run {run_id!r}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"training run {run_id!r} is not a JSON object")
        return value

    def remove(self, run_id: str) -> None:
        directory = self.directory(run_id)
        shutil.rmtree(directory, ignore_errors=True)
