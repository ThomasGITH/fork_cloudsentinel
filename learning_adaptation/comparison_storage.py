"""Atomic file-backed persistence for immutable comparison runs."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import threading
from typing import Any, Callable

try:
    from learning_adaptation.training_run_storage import (
        atomic_write_json,
        validate_identifier,
    )
except ModuleNotFoundError:
    from training_run_storage import atomic_write_json, validate_identifier


class ComparisonNotFoundError(LookupError):
    pass


class ComparisonConflictError(RuntimeError):
    pass


class ComparisonStore:
    _locks_guard = threading.Lock()
    _locks: dict[str, threading.RLock] = {}

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _path(self, comparison_id: str) -> Path:
        comparison_id = validate_identifier(comparison_id, "comparison_id")
        path = (self.root / comparison_id / "comparison.json").resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("comparison_id resolves outside comparison storage")
        return path

    @classmethod
    def _lock(cls, key: str) -> threading.RLock:
        with cls._locks_guard:
            return cls._locks.setdefault(key, threading.RLock())

    def create(self, record: dict[str, Any]) -> dict[str, Any]:
        path = self._path(record["comparison_id"])
        with self._lock(str(path)):
            if path.exists():
                raise ComparisonConflictError("comparison already exists")
            atomic_write_json(path, record)
        return deepcopy(record)

    def create_idempotent(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        """Create once per client request within the single API process."""

        request_id = record.get("client_request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("client_request_id is required")
        with self._lock(str(self.root / ".client-request-index")):
            existing = self.find_client_request(request_id)
            if existing is not None:
                return existing, False
            return self.create(record), True

    def read(self, comparison_id: str) -> dict[str, Any]:
        path = self._path(comparison_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ComparisonNotFoundError(f"unknown comparison: {comparison_id!r}") from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("stored comparison is unavailable") from exc
        if not isinstance(value, dict):
            raise ValueError("stored comparison is invalid")
        return value

    def update(
        self, comparison_id: str, mutate: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        path = self._path(comparison_id)
        with self._lock(str(path)):
            value = self.read(comparison_id)
            if value.get("status") in {"completed", "partial", "failed"}:
                raise ComparisonConflictError("terminal comparison records are immutable")
            mutate(value)
            atomic_write_json(path, value)
            return deepcopy(value)

    def identifiers(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            item.name
            for item in self.root.iterdir()
            if item.is_dir() and (item / "comparison.json").is_file()
        )

    def find_client_request(self, client_request_id: str) -> dict[str, Any] | None:
        for comparison_id in self.identifiers():
            try:
                record = self.read(comparison_id)
            except (ComparisonNotFoundError, ValueError):
                continue
            if record.get("client_request_id") == client_request_id:
                return record
        return None
