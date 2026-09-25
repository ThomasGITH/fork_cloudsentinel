"""Atomic, compact result storage for Isolation Forest detections."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


RESULT_FILENAME = "isolation_forest_results.json"
SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ResultStorageError(RuntimeError):
    """Raised when a detection result cannot be stored safely."""


class ResultConflictError(ResultStorageError):
    """Raised when a task iteration has already been recorded."""


def validate_task_id(value: Any) -> str:
    if not isinstance(value, str) or not SAFE_TASK_ID.fullmatch(value):
        raise ValueError(
            "task_id must contain 1-128 letters, digits, dots, underscores, or hyphens"
        )
    if value in {".", ".."}:
        raise ValueError("task_id must identify a result directory")
    return value


def result_path(results_root: str | os.PathLike[str], task_id: str) -> Path:
    root = Path(results_root).resolve()
    task_directory = (root / validate_task_id(task_id)).resolve()
    if task_directory.parent != root:
        raise ResultStorageError(
            "task_id resolves outside IF_RESULTS_STORAGE_ROOT"
        )
    return task_directory / RESULT_FILENAME


def _read_existing(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink():
        raise ResultStorageError("Isolation Forest result path is unsafe")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResultStorageError(
            "existing Isolation Forest result file is not valid JSON"
        ) from exc
    if not isinstance(document, dict) or not isinstance(document.get("results"), dict):
        raise ResultStorageError("existing Isolation Forest result file is invalid")
    return document


def ensure_iteration_available(
    results_root: str | os.PathLike[str], task_id: str, iteration: str
) -> None:
    document = _read_existing(result_path(results_root, task_id))
    if document is not None and iteration in document["results"]:
        raise ResultConflictError(
            f"iteration {iteration!r} already exists for task_id {task_id!r}"
        )


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(document, temporary, indent=2, sort_keys=True)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as exc:
        raise ResultStorageError(
            f"could not atomically store Isolation Forest result: {exc}"
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def append_result(
    results_root: str | os.PathLike[str],
    *,
    task_id: str,
    model_id: str,
    start_time: int | float,
    containers: list[str],
    metrics: list[str],
    step: int | float,
    crca_threshold: int | float,
    iteration: str,
    result: dict[str, Any],
) -> Path:
    """Atomically append one new iteration, rejecting duplicate iteration keys."""
    path = result_path(results_root, task_id)
    document = _read_existing(path)
    if document is None:
        document = {
            "detector_id": "isolation-forest",
            "model": model_id,
            "start_time": start_time,
            "containers": containers,
            "metrics": metrics,
            "step": step,
            "crca_threshold": crca_threshold,
            "results": {},
        }
    else:
        expected_identity = {
            "detector_id": "isolation-forest",
            "model": model_id,
            "containers": containers,
            "metrics": metrics,
            "step": step,
            "crca_threshold": crca_threshold,
        }
        mismatches = [
            key for key, value in expected_identity.items() if document.get(key) != value
        ]
        if mismatches:
            raise ResultConflictError(
                "task result identity does not match existing fields: "
                + ", ".join(mismatches)
            )
    if iteration in document["results"]:
        raise ResultConflictError(
            f"iteration {iteration!r} already exists for task_id {task_id!r}"
        )
    document["results"][iteration] = result
    _atomic_write(path, document)
    return path


def update_result(
    results_root: str | os.PathLike[str],
    *,
    task_id: str,
    iteration: str,
    result: dict[str, Any],
) -> Path:
    """Atomically replace one existing iteration after its RCA attempt."""
    path = result_path(results_root, task_id)
    document = _read_existing(path)
    if document is None or iteration not in document["results"]:
        raise ResultStorageError(
            f"iteration {iteration!r} does not exist for task_id {task_id!r}"
        )
    document["results"][iteration] = result
    _atomic_write(path, document)
    return path
