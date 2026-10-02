"""Verified evaluation bundles and generic-runtime client for comparisons."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable
from urllib.parse import quote
import zipfile

import requests


class ComparisonEvaluationError(RuntimeError):
    """Safe comparison transport, integrity, or runtime failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_hash(order: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(order, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _safe_response_error(response: Any, fallback: str) -> str:
    try:
        payload = response.json()
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        message = payload["error"]
        if "://" not in message and "/app/" not in message:
            return " ".join(message.split())[:500]
    return fallback


class EvaluationBundleClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: tuple[float, float] = (5.0, 60.0),
        maximum_bytes: int = 250 * 1024 * 1024,
        request_get: Callable[..., Any] = requests.get,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.maximum_bytes = maximum_bytes
        self.request_get = request_get

    def download(self, dataset: dict[str, Any]) -> dict[str, Any]:
        temporary = Path(tempfile.mkdtemp(prefix="comparison-evaluation-download-"))
        archive_path = temporary / "evaluation-bundle.zip"
        response = None
        url = (
            f"{self.base_url}/datasets/{quote(dataset['dataset_id'], safe='')}/versions/"
            f"{dataset['version']}/partitions/{quote(dataset['partition_id'], safe='')}/"
            "evaluation-bundle"
        )
        try:
            try:
                response = self.request_get(url, stream=True, timeout=self.timeout)
            except requests.RequestException as exc:
                raise ComparisonEvaluationError(
                    "Data Catalogue evaluation bundle request failed"
                ) from exc
            if response.status_code != 200:
                raise ComparisonEvaluationError(
                    _safe_response_error(
                        response,
                        f"Data Catalogue evaluation bundle returned HTTP {response.status_code}",
                    )
                )
            size = 0
            with archive_path.open("wb") as target:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > self.maximum_bytes:
                        raise ComparisonEvaluationError(
                            "evaluation bundle exceeds the configured size limit"
                        )
                    target.write(chunk)
            return verify_evaluation_bundle(
                archive_path, dataset, maximum_bytes=self.maximum_bytes
            )
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()


def verify_evaluation_bundle(
    archive_path: Path,
    dataset: dict[str, Any],
    *,
    maximum_bytes: int = 250 * 1024 * 1024,
) -> dict[str, Any]:
    extraction = archive_path.parent / "verified"
    extraction.mkdir()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            names = {item.filename for item in infos}
            allowed = {"manifest.json", "matrix.csv", "labels.csv", "timestamps.csv"}
            if not {"manifest.json", "matrix.csv"}.issubset(names) or not names.issubset(allowed):
                raise ComparisonEvaluationError("evaluation bundle has invalid contents")
            if len(names) != len(infos) or sum(item.file_size for item in infos) > maximum_bytes:
                raise ComparisonEvaluationError("evaluation bundle exceeds safe limits")
            for info in infos:
                if info.is_dir() or Path(info.filename).name != info.filename:
                    raise ComparisonEvaluationError("evaluation bundle contains unsafe paths")
                if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ComparisonEvaluationError("evaluation bundle contains a symlink")
                with archive.open(info) as source, (extraction / info.filename).open("wb") as target:
                    shutil.copyfileobj(source, target)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ComparisonEvaluationError("evaluation bundle is corrupt") from exc
    try:
        manifest = json.loads((extraction / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonEvaluationError("evaluation manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("profile") != "metrics-evaluation/v1":
        raise ComparisonEvaluationError("evaluation bundle profile is unsupported")
    for field, expected in (
        ("dataset_id", dataset["dataset_id"]),
        ("dataset_version", dataset["version"]),
        ("partition_id", dataset["partition_id"]),
    ):
        if manifest.get(field) != expected:
            raise ComparisonEvaluationError(f"evaluation manifest {field} mismatch")
    order = manifest.get("feature_order")
    if (
        not isinstance(order, list)
        or not order
        or any(not isinstance(item, str) or not item for item in order)
        or len(set(order)) != len(order)
        or manifest.get("feature_order_sha256") != _feature_hash(order)
    ):
        raise ComparisonEvaluationError("evaluation feature identity is invalid")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != names - {"manifest.json"}:
        raise ComparisonEvaluationError("evaluation file manifest is invalid")
    for filename, record in files.items():
        path = extraction / filename
        if (
            not isinstance(record, dict)
            or record.get("sha256") != _sha256(path)
            or record.get("bytes") != path.stat().st_size
        ):
            raise ComparisonEvaluationError(f"evaluation {filename} checksum mismatch")
    with (extraction / "matrix.csv").open("r", encoding="utf-8", newline="") as source:
        matrix = list(csv.reader(source))
    if not matrix or any(len(row) != len(order) for row in matrix):
        raise ComparisonEvaluationError("evaluation matrix shape is invalid")
    labels = None
    if "labels.csv" in names:
        with (extraction / "labels.csv").open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.reader(source))
        if len(rows) != len(matrix) or any(len(row) != 1 or row[0] not in {"0", "1"} for row in rows):
            raise ComparisonEvaluationError("evaluation labels are invalid")
        labels = [int(row[0]) for row in rows]
    timestamps = None
    if "timestamps.csv" in names:
        with (extraction / "timestamps.csv").open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.reader(source))
        if len(rows) != len(matrix) or any(len(row) != 1 or not row[0] for row in rows):
            raise ComparisonEvaluationError("evaluation timestamps are invalid")
        timestamps = [row[0] for row in rows]
    counts = manifest.get("counts", {})
    if counts.get("observations") != len(matrix) or counts.get("features") != len(order):
        raise ComparisonEvaluationError("evaluation counts do not match bundle contents")
    if bool(labels) != bool(manifest.get("ground_truth_available")):
        raise ComparisonEvaluationError("evaluation ground-truth declaration is inconsistent")
    return {
        "temporary_root": archive_path.parent,
        "matrix_path": extraction / "matrix.csv",
        "labels": labels,
        "timestamps": timestamps,
        "manifest": manifest,
    }


class GenericEvaluationClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: tuple[float, float] = (5.0, 120.0),
        maximum_response_bytes: int = 20 * 1024 * 1024,
        request_post: Callable[..., Any] = requests.post,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.maximum_response_bytes = maximum_response_bytes
        self.request_post = request_post

    def evaluate(
        self,
        model_id: str,
        matrix_path: Path,
        feature_order: list[str],
        feature_order_sha256: str,
        timestamps: list[str] | None,
    ) -> dict[str, Any]:
        try:
            activated = self.request_post(
                f"{self.base_url}/models/{quote(model_id, safe='')}/activate",
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ComparisonEvaluationError("generic model activation failed") from exc
        if activated.status_code != 200:
            raise ComparisonEvaluationError(
                _safe_response_error(activated, "generic model activation was rejected")
            )
        metadata = {
            "model_id": model_id,
            "feature_order": feature_order,
            "feature_order_sha256": feature_order_sha256,
            "timestamps": timestamps or [],
            "context": {},
        }
        try:
            with matrix_path.open("rb") as matrix:
                response = self.request_post(
                    f"{self.base_url}/internal/evaluate",
                    files={"matrix": ("matrix.csv", matrix, "text/csv")},
                    data={"metadata": json.dumps(metadata, separators=(",", ":"))},
                    timeout=self.timeout,
                )
        except (OSError, requests.RequestException) as exc:
            raise ComparisonEvaluationError("generic model evaluation failed") from exc
        if response.status_code != 200:
            raise ComparisonEvaluationError(
                _safe_response_error(response, "generic model evaluation was rejected")
            )
        content_length = (getattr(response, "headers", {}) or {}).get("Content-Length")
        if content_length:
            try:
                if int(content_length) > self.maximum_response_bytes:
                    raise ComparisonEvaluationError(
                        "generic evaluation response exceeds the configured limit"
                    )
            except ValueError:
                pass
        try:
            payload = response.json()
        except ValueError as exc:
            raise ComparisonEvaluationError("generic runtime returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ComparisonEvaluationError("generic runtime returned an invalid result")
        try:
            response_size = len(
                json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
            )
        except (TypeError, ValueError) as exc:
            raise ComparisonEvaluationError(
                "generic runtime returned an invalid result"
            ) from exc
        if response_size > self.maximum_response_bytes:
            raise ComparisonEvaluationError(
                "generic evaluation response exceeds the configured limit"
            )
        return payload
