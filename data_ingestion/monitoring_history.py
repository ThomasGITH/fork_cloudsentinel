"""Durable, file-backed history for generic live-monitoring sessions."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Iterator


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ACTIVE_STATUSES = {"queued", "collecting", "warmup", "active"}
TERMINAL_STATUSES = {"stopped", "failed", "dispatch_failed"}


class MonitoringHistoryError(RuntimeError):
    """Safe monitoring-history failure."""


class MonitoringSessionNotFound(MonitoringHistoryError):
    pass


class MonitoringSessionStopped(MonitoringHistoryError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(canonical_bytes(value))
            target.write(b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MonitoringHistoryError("monitoring history record is unreadable") from exc
    if not isinstance(value, dict):
        raise MonitoringHistoryError("monitoring history record is invalid")
    return value


class MonitoringHistoryStore:
    """Atomic single-node store shared by the Data Ingestion API and worker."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_observations: int = 5000,
        max_features: int = 1000,
    ):
        self.root = Path(root).resolve()
        self.sessions = self.root / "sessions"
        self.max_observations = max_observations
        self.max_features = max_features

    def _validate_id(self, value: str, name: str) -> str:
        if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
            raise MonitoringHistoryError(f"{name} is invalid")
        return value

    def _session_dir(self, session_id: str) -> Path:
        return self.sessions / self._validate_id(session_id, "session_id")

    def _publish_blob(self, kind: str, payload: dict[str, Any]) -> tuple[str, str]:
        if kind not in {"input", "result", "rca_input", "rca_result"}:
            raise MonitoringHistoryError("monitoring artifact kind is invalid")
        digest = canonical_sha256(payload)
        directory = self.root / "artifacts" / kind
        final = directory / f"sha256_{digest}.json"
        if not final.exists():
            _atomic_json(final, payload)
        elif canonical_sha256(_read_json(final)) != digest:
            raise MonitoringHistoryError("monitoring artifact checksum mismatch")
        return f"{kind}:sha256:{digest}", digest

    def read_artifact(self, reference: str) -> dict[str, Any]:
        try:
            kind, algorithm, digest = reference.split(":", 2)
        except (AttributeError, ValueError) as exc:
            raise MonitoringHistoryError("monitoring artifact reference is invalid") from exc
        if (
            kind not in {"input", "result", "rca_input", "rca_result"}
            or algorithm != "sha256"
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise MonitoringHistoryError("monitoring artifact reference is invalid")
        value = _read_json(self.root / "artifacts" / kind / f"sha256_{digest}.json")
        if canonical_sha256(value) != digest:
            raise MonitoringHistoryError("monitoring artifact checksum mismatch")
        return value

    @contextmanager
    def _locked(self, session_id: str) -> Iterator[None]:
        directory = self._session_dir(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        lock_path = directory / ".lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def create_session(self, record: dict[str, Any], recipe: dict[str, Any]) -> dict[str, Any]:
        session_id = self._validate_id(record.get("session_id"), "session_id")
        directory = self._session_dir(session_id)
        with self._locked(session_id):
            if (directory / "session.json").exists():
                raise MonitoringHistoryError("monitoring session already exists")
            recipe_copy = deepcopy(recipe)
            recipe_sha256 = canonical_sha256(
                {key: value for key, value in recipe_copy.items() if key != "recipe_sha256"}
            )
            if recipe_copy.get("recipe_sha256") != recipe_sha256:
                raise MonitoringHistoryError("monitoring recipe checksum mismatch")
            _atomic_json(directory / "recipe.json", recipe_copy)
            stored = deepcopy(record)
            stored.update(
                schema_version=1,
                recipe_reference="recipe.json",
                recipe_snapshot_sha256=canonical_sha256(recipe_copy),
                cycle_count=0,
                prediction_watermark=None,
            )
            _atomic_json(directory / "session.json", stored)
            return deepcopy(stored)

    def read_session(self, session_id: str) -> dict[str, Any]:
        path = self._session_dir(session_id) / "session.json"
        if not path.is_file():
            raise MonitoringSessionNotFound("unknown monitoring session")
        return _read_json(path)

    def read_recipe(self, session_id: str) -> dict[str, Any]:
        record = self.read_session(session_id)
        path = self._session_dir(session_id) / str(record.get("recipe_reference"))
        recipe = _read_json(path)
        if canonical_sha256(recipe) != record.get("recipe_snapshot_sha256"):
            raise MonitoringHistoryError("monitoring recipe snapshot checksum mismatch")
        return recipe

    def update_session(
        self, session_id: str, update: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        with self._locked(session_id):
            record = self.read_session(session_id)
            update(record)
            record["updated_at"] = utc_now()
            _atomic_json(self._session_dir(session_id) / "session.json", record)
            return deepcopy(record)

    def stop_session(self, session_id: str, stopped_at: str) -> dict[str, Any]:
        def stop(record: dict[str, Any]) -> None:
            record["status"] = "stopped"
            record["stopped_at"] = stopped_at

        return self.update_session(session_id, stop)

    def fail_session(self, session_id: str, summary: str, *, status: str = "failed") -> dict[str, Any]:
        safe = str(summary)[:500]

        def fail(record: dict[str, Any]) -> None:
            if record.get("status") != "stopped":
                record["status"] = status
                record["error"] = {"message": safe, "type": "monitoring_error"}
                record["completed_at"] = utc_now()

        return self.update_session(session_id, fail)

    def commit_cycle(
        self,
        session_id: str,
        cycle_id: str,
        *,
        input_payload: dict[str, Any],
        result_payload: dict[str, Any],
        cycle: dict[str, Any],
        next_watermark: str | None,
        session_status: str,
    ) -> dict[str, Any]:
        self._validate_id(cycle_id, "cycle_id")
        rows = input_payload.get("matrix")
        timestamps = input_payload.get("timestamps")
        features = input_payload.get("feature_order")
        if (
            not isinstance(rows, list)
            or not isinstance(timestamps, list)
            or len(rows) != len(timestamps)
            or len(rows) > self.max_observations
            or not isinstance(features, list)
            or len(features) > self.max_features
            or any(not isinstance(row, list) or len(row) != len(features) for row in rows)
        ):
            raise MonitoringHistoryError("monitoring input artifact exceeds safe bounds")
        directory = self._session_dir(session_id)
        cycles = directory / "cycles"
        final = cycles / cycle_id
        with self._locked(session_id):
            session = self.read_session(session_id)
            if session.get("status") == "stopped":
                raise MonitoringSessionStopped("monitoring session is stopped")
            if final.exists():
                existing = _read_json(final / "cycle.json")
                if (
                    existing.get("cycle_id") != cycle_id
                    or existing.get("input_sha256") != canonical_sha256(input_payload)
                    or existing.get("result_sha256") != canonical_sha256(result_payload)
                    or any(existing.get(key) != value for key, value in cycle.items())
                    or existing.get("prediction_watermark_after") != next_watermark
                    or existing.get("session_status_after") != session_status
                ):
                    raise MonitoringHistoryError("monitoring cycle identity conflict")
                self._apply_cycle_to_session(
                    session,
                    existing,
                    existing.get("prediction_watermark_after"),
                    existing.get("session_status_after"),
                )
                _atomic_json(directory / "session.json", session)
                return existing
            cycles.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=f".{cycle_id}.", dir=cycles))
            try:
                input_record = deepcopy(input_payload)
                result_record = deepcopy(result_payload)
                input_reference, input_sha256 = self._publish_blob("input", input_record)
                result_reference, result_sha256 = self._publish_blob("result", result_record)
                committed = deepcopy(cycle)
                committed.update(
                    schema_version=1,
                    cycle_id=cycle_id,
                    session_id=session_id,
                    input_reference=input_reference,
                    input_sha256=input_sha256,
                    result_reference=result_reference,
                    result_sha256=result_sha256,
                    prediction_watermark_after=next_watermark,
                    session_status_after=session_status,
                    committed_at=utc_now(),
                )
                _atomic_json(stage / "cycle.json", committed)
                os.replace(stage, final)
            except Exception:
                shutil.rmtree(stage, ignore_errors=True)
                raise
            self._apply_cycle_to_session(
                session, committed, next_watermark, session_status
            )
            _atomic_json(directory / "session.json", session)
            return deepcopy(committed)

    @staticmethod
    def _apply_cycle_to_session(
        session: dict[str, Any],
        cycle: dict[str, Any],
        next_watermark: str | None,
        status: str,
    ) -> None:
        session["status"] = status
        session["last_cycle_id"] = cycle["cycle_id"]
        if cycle.get("status") != "failed":
            session["last_successful_update"] = cycle.get("committed_at")
        session["cycle_count"] = int(session.get("cycle_count", 0)) + (
            0 if session.get("last_committed_cycle_id") == cycle["cycle_id"] else 1
        )
        session["last_committed_cycle_id"] = cycle["cycle_id"]
        session["latest_cycle"] = {
            key: deepcopy(cycle.get(key))
            for key in (
                "cycle_id",
                "status",
                "artifact_identity",
                "query_window",
                "inference_window",
                "prediction_window",
                "prediction_count",
                "anomaly_count",
                "anomaly_percentage",
                "warmup_observations",
                "warnings",
                "safe_error",
                "started_at",
                "completed_at",
                "committed_at",
            )
            if cycle.get(key) is not None
        }
        if next_watermark is not None:
            current = session.get("prediction_watermark")
            if current is None or next_watermark > current:
                session["prediction_watermark"] = next_watermark
        session["updated_at"] = utc_now()
        session.pop("error", None)

    def _cycle_records(self, session_id: str) -> list[dict[str, Any]]:
        cycles = self._session_dir(session_id) / "cycles"
        if not cycles.is_dir():
            return []
        records = []
        for path in cycles.iterdir():
            if not path.is_dir() or path.name.startswith("."):
                continue
            try:
                records.append(_read_json(path / "cycle.json"))
            except MonitoringHistoryError:
                continue
        records.sort(key=lambda item: item.get("committed_at", ""), reverse=True)
        return records

    def read_cycle(self, session_id: str, cycle_id: str) -> dict[str, Any]:
        self._validate_id(cycle_id, "cycle_id")
        path = self._session_dir(session_id) / "cycles" / cycle_id / "cycle.json"
        if not path.is_file():
            raise MonitoringHistoryError("unknown monitoring cycle")
        return _read_json(path)

    def pending_rca_cycles(self, session_id: str, *, limit: int = 100) -> list[str]:
        self.read_session(session_id)
        if limit < 1 or limit > 100:
            raise MonitoringHistoryError("RCA reconciliation limit is invalid")
        records = [
            item for item in reversed(self._cycle_records(session_id))
            if self.read_rca_event(session_id, item["cycle_id"]) is None
        ]
        return [item["cycle_id"] for item in records[:limit]]

    def _rca_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "rca_runs"

    def _rca_event_path(self, session_id: str, cycle_id: str) -> Path:
        self._validate_id(cycle_id, "cycle_id")
        return self._session_dir(session_id) / "rca_events" / f"{cycle_id}.json"

    def read_rca_event(self, session_id: str, cycle_id: str) -> dict[str, Any] | None:
        path = self._rca_event_path(session_id, cycle_id)
        return _read_json(path) if path.is_file() else None

    def record_rca_event(
        self,
        session_id: str,
        cycle_id: str,
        event: dict[str, Any],
        *,
        rca_state: dict[str, Any],
    ) -> dict[str, Any]:
        with self._locked(session_id):
            self.read_cycle(session_id, cycle_id)
            path = self._rca_event_path(session_id, cycle_id)
            if path.exists():
                return _read_json(path)
            stored = {
                "schema_version": 1,
                "cycle_id": cycle_id,
                "session_id": session_id,
                "recorded_at": utc_now(),
                **deepcopy(event),
            }
            _atomic_json(path, stored)
            session = self.read_session(session_id)
            session["rca"] = deepcopy(rca_state)
            session["updated_at"] = utc_now()
            _atomic_json(self._session_dir(session_id) / "session.json", session)
            return stored

    def create_rca_run(
        self,
        session_id: str,
        cycle_id: str,
        run: dict[str, Any],
        *,
        input_payload: dict[str, Any] | None,
        event: dict[str, Any],
        rca_state: dict[str, Any],
    ) -> dict[str, Any]:
        run_id = self._validate_id(run.get("run_id"), "run_id")
        directory = self._rca_dir(session_id) / run_id
        with self._locked(session_id):
            self.read_cycle(session_id, cycle_id)
            existing_event = self.read_rca_event(session_id, cycle_id)
            if existing_event is not None and existing_event.get("run_id") not in {None, run_id}:
                raise MonitoringHistoryError("monitoring cycle already has an RCA run")
            if directory.exists():
                existing = _read_json(directory / "run.json")
                if existing.get("cycle_id") != cycle_id:
                    raise MonitoringHistoryError("RCA run identity conflict")
                event_path = self._rca_event_path(session_id, cycle_id)
                if not event_path.exists():
                    _atomic_json(
                        event_path,
                        {
                            "schema_version": 1,
                            "cycle_id": cycle_id,
                            "session_id": session_id,
                            "recorded_at": utc_now(),
                            "run_id": run_id,
                            **deepcopy(event),
                        },
                    )
                session = self.read_session(session_id)
                session["rca"] = deepcopy(rca_state)
                session["updated_at"] = utc_now()
                _atomic_json(self._session_dir(session_id) / "session.json", session)
                return existing
            stored = deepcopy(run)
            stored.update(schema_version=1, session_id=session_id, cycle_id=cycle_id)
            if input_payload is not None:
                reference, digest = self._publish_blob("rca_input", input_payload)
                stored["input_reference"] = reference
                stored["input_sha256"] = digest
            stage_parent = self._rca_dir(session_id)
            stage_parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=stage_parent))
            try:
                _atomic_json(stage / "run.json", stored)
                os.replace(stage, directory)
            except Exception:
                shutil.rmtree(stage, ignore_errors=True)
                raise
            stored_event = {
                "schema_version": 1,
                "cycle_id": cycle_id,
                "session_id": session_id,
                "recorded_at": utc_now(),
                "run_id": run_id,
                **deepcopy(event),
            }
            _atomic_json(self._rca_event_path(session_id, cycle_id), stored_event)
            session = self.read_session(session_id)
            session["rca"] = deepcopy(rca_state)
            session["updated_at"] = utc_now()
            _atomic_json(self._session_dir(session_id) / "session.json", session)
            return deepcopy(stored)

    def read_rca_run(self, session_id: str, run_id: str) -> dict[str, Any]:
        self._validate_id(run_id, "run_id")
        path = self._rca_dir(session_id) / run_id / "run.json"
        if not path.is_file():
            raise MonitoringHistoryError("unknown monitoring RCA run")
        return _read_json(path)

    def update_rca_run(
        self, session_id: str, run_id: str, update: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        with self._locked(session_id):
            record = self.read_rca_run(session_id, run_id)
            update(record)
            record["updated_at"] = utc_now()
            _atomic_json(self._rca_dir(session_id) / run_id / "run.json", record)
            return deepcopy(record)

    def claim_rca_run(self, session_id: str, run_id: str) -> dict[str, Any] | None:
        claimed = None

        def claim(record: dict[str, Any]) -> None:
            nonlocal claimed
            if record.get("status") != "queued":
                return
            record["status"] = "running"
            record["started_at"] = utc_now()
            claimed = deepcopy(record)

        self.update_rca_run(session_id, run_id, claim)
        return claimed

    def complete_rca_run(
        self, session_id: str, run_id: str, result_payload: dict[str, Any]
    ) -> dict[str, Any]:
        reference, digest = self._publish_blob("rca_result", result_payload)

        def complete(record: dict[str, Any]) -> None:
            if record.get("status") == "completed":
                if record.get("result_sha256") != digest:
                    raise MonitoringHistoryError("RCA result identity conflict")
                return
            if record.get("status") not in {"running", "queued"}:
                raise MonitoringHistoryError("RCA run is already terminal")
            record.update(
                status="completed",
                completed_at=utc_now(),
                result_reference=reference,
                result_sha256=digest,
                ranking_count=len(result_payload.get("ranking", [])),
                graph_count=len(result_payload.get("graphs", [])),
            )
            record.pop("failure", None)

        return self.update_rca_run(session_id, run_id, complete)

    def fail_rca_run(self, session_id: str, run_id: str, message: str) -> dict[str, Any]:
        def fail(record: dict[str, Any]) -> None:
            if record.get("status") == "completed":
                return
            record.update(
                status="failed",
                completed_at=utc_now(),
                failure={"type": "rca_execution_error", "message": str(message)[:500]},
            )

        return self.update_rca_run(session_id, run_id, fail)

    def list_rca_runs(
        self, session_id: str, *, page: int = 1, page_size: int = 20
    ) -> dict[str, Any]:
        self.read_session(session_id)
        page, page_size = self._page(page, page_size)
        records = []
        directory = self._rca_dir(session_id)
        if directory.is_dir():
            for path in directory.iterdir():
                if not path.is_dir() or path.name.startswith("."):
                    continue
                try:
                    records.append(_read_json(path / "run.json"))
                except MonitoringHistoryError:
                    continue
        records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        start = (page - 1) * page_size
        return {
            "items": [self.public_rca_run(item) for item in records[start : start + page_size]],
            "page": page,
            "page_size": page_size,
            "total": len(records),
            "pages": math.ceil(len(records) / page_size) if records else 0,
            "truncated": start + page_size < len(records),
        }

    def rca_run_detail(self, session_id: str, run_id: str) -> dict[str, Any]:
        record = self.read_rca_run(session_id, run_id)
        result = None
        if record.get("status") == "completed" and record.get("result_reference"):
            stored = self.read_artifact(record["result_reference"])
            result = {
                "ranking": deepcopy(stored.get("ranking", [])[:100]),
                "graphs": deepcopy(stored.get("graphs", [])[:3]),
                "truncated": len(stored.get("ranking", [])) > 100,
            }
        return {"run": self.public_rca_run(record), "result": result}

    @staticmethod
    def public_rca_run(record: dict[str, Any]) -> dict[str, Any]:
        allowed = (
            "run_id", "session_id", "cycle_id", "status", "created_at",
            "started_at", "updated_at", "completed_at", "trigger_percentage",
            "threshold_percentage", "episode_id", "cooldown_until", "trigger_window",
            "context_window", "analysis_window", "observation_count", "feature_count",
            "feature_identity", "input_sha256", "preprocessing", "algorithm",
            "model_identity", "recipe_identity",
            "result_sha256", "ranking_count", "graph_count", "failure",
            "skip_reason_code", "skip_reason",
            "dispatch_status", "dispatch_error",
        )
        return {key: deepcopy(record[key]) for key in allowed if key in record}

    def list_cycles(
        self, session_id: str, *, page: int = 1, page_size: int = 50
    ) -> dict[str, Any]:
        self.read_session(session_id)
        page, page_size = self._page(page, page_size)
        records = self._cycle_records(session_id)
        start = (page - 1) * page_size
        return {
            "items": [self.public_cycle(item) for item in records[start : start + page_size]],
            "page": page,
            "page_size": page_size,
            "total": len(records),
            "pages": math.ceil(len(records) / page_size) if records else 0,
            "truncated": start + page_size < len(records),
        }

    def list_sessions(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        status: str | None = None,
        search: str | None = None,
    ) -> dict[str, Any]:
        page, page_size = self._page(page, page_size)
        records = []
        if self.sessions.is_dir():
            for path in self.sessions.iterdir():
                if not path.is_dir() or path.name.startswith("."):
                    continue
                try:
                    record = _read_json(path / "session.json")
                except MonitoringHistoryError:
                    continue
                if status and record.get("status") != status:
                    continue
                haystack = f"{record.get('display_name', '')} {record.get('model_id', '')}".lower()
                if search and search.lower() not in haystack:
                    continue
                records.append(record)
        records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        start = (page - 1) * page_size
        return {
            "items": [self.public_session(item) for item in records[start : start + page_size]],
            "page": page,
            "page_size": page_size,
            "total": len(records),
            "pages": math.ceil(len(records) / page_size) if records else 0,
        }

    def session_detail(
        self, session_id: str, *, cycle_page: int = 1, cycle_page_size: int = 50
    ) -> dict[str, Any]:
        record = self.read_session(session_id)
        cycles = self.list_cycles(
            session_id, page=cycle_page, page_size=cycle_page_size
        )
        for cycle in cycles["items"]:
            event = self.read_rca_event(session_id, cycle["cycle_id"])
            if event is not None:
                cycle["rca_trigger"] = {
                    key: deepcopy(event[key])
                    for key in (
                        "status",
                        "reason_code",
                        "reason",
                        "threshold_percentage",
                        "episode_state",
                        "run_id",
                    )
                    if key in event
                }
        return {
            "session": self.public_session(record),
            "cycles": cycles,
            "rca_runs": self.list_rca_runs(session_id, page=1, page_size=20),
            "provenance": {
                "artifact_id": record.get("artifact_id"),
                "artifact_manifest_sha256": record.get("artifact_manifest_sha256"),
                "inference_contract": record.get("inference_contract"),
                "artifact_format": record.get("artifact_format"),
                "recipe_contract": record.get("recipe_contract"),
                "recipe_sha256": record.get("recipe_sha256"),
                "feature_order_sha256": record.get("feature_order_sha256"),
                "sampling_interval_seconds": record.get("sampling_interval_seconds"),
            },
        }

    @staticmethod
    def _page(page: int, page_size: int) -> tuple[int, int]:
        try:
            page = int(page)
            page_size = int(page_size)
        except (TypeError, ValueError) as exc:
            raise MonitoringHistoryError("pagination is invalid") from exc
        if page < 1 or page_size < 1 or page_size > 100:
            raise MonitoringHistoryError("pagination is outside safe limits")
        return page, page_size

    @staticmethod
    def public_session(record: dict[str, Any]) -> dict[str, Any]:
        allowed = (
            "session_id",
            "display_name",
            "model_id",
            "detector_id",
            "detector_version",
            "artifact_id",
            "inference_contract",
            "artifact_format",
            "status",
            "created_at",
            "started_at",
            "updated_at",
            "stopped_at",
            "completed_at",
            "last_successful_update",
            "window_seconds",
            "poll_interval_seconds",
            "sampling_interval_seconds",
            "buffered_observations",
            "warmup_observations",
            "cycle_count",
            "latest_cycle",
            "warnings",
            "error",
            "rca",
        )
        return {key: deepcopy(record[key]) for key in allowed if key in record}

    @staticmethod
    def public_cycle(record: dict[str, Any]) -> dict[str, Any]:
        allowed = (
            "cycle_id",
            "status",
            "artifact_identity",
            "query_window",
            "inference_window",
            "prediction_window",
            "started_at",
            "completed_at",
            "committed_at",
            "prediction_count",
            "anomaly_count",
            "anomaly_percentage",
            "warmup_observations",
            "warnings",
            "safe_error",
            "input_sha256",
            "result_sha256",
        )
        return {key: deepcopy(record[key]) for key in allowed if key in record}
