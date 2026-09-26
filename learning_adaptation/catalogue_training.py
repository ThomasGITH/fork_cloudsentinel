"""Download and verify immutable Data Catalogue training bundles."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable
import zipfile

import requests


EXPECTED_FILES = {"manifest.json", "train.csv", "test.csv", "labels.csv"}


class CatalogueBundleError(ValueError):
    """Raised for central catalogue transport or integrity failures."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_order_hash(value: list[str]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _safe_error(response: Any) -> str:
    try:
        payload = response.json()
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"][:500]
    text = getattr(response, "text", "")
    return text.strip()[:500] if isinstance(text, str) and text.strip() else "no response body"


class CatalogueBundleClient:
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
        temporary = Path(tempfile.mkdtemp(prefix="catalogue-training-download-"))
        archive_path = temporary / "training-bundle.zip"
        response = None
        url = (
            f"{self.base_url}/datasets/{dataset['dataset_id']}/versions/"
            f"{dataset['version']}/partitions/{dataset['partition_id']}/training-bundle"
        )
        try:
            try:
                response = self.request_get(url, stream=True, timeout=self.timeout)
            except requests.RequestException as exc:
                raise CatalogueBundleError(
                    f"Data Catalogue training bundle request failed: {exc}"
                ) from exc
            if response.status_code != 200:
                raise CatalogueBundleError(
                    "Data Catalogue training bundle returned "
                    f"HTTP {response.status_code}: {_safe_error(response)}"
                )
            size = 0
            with archive_path.open("wb") as target:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > self.maximum_bytes:
                        raise CatalogueBundleError(
                            "Data Catalogue training bundle exceeds the configured size limit"
                        )
                    target.write(chunk)
            return verify_bundle(
                archive_path, dataset, maximum_bytes=self.maximum_bytes
            )
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()


def verify_bundle(
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
            if names != EXPECTED_FILES or len(infos) != len(EXPECTED_FILES):
                raise CatalogueBundleError("Data Catalogue training bundle has invalid contents")
            if sum(item.file_size for item in infos) > maximum_bytes:
                raise CatalogueBundleError(
                    "Data Catalogue training bundle expands beyond the safe size limit"
                )
            for info in infos:
                if info.is_dir() or Path(info.filename).name != info.filename:
                    raise CatalogueBundleError("Data Catalogue training bundle contains unsafe paths")
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise CatalogueBundleError("Data Catalogue training bundle contains a symlink")
                destination = extraction / info.filename
                with archive.open(info) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
    except (zipfile.BadZipFile, OSError) as exc:
        raise CatalogueBundleError(f"Data Catalogue training bundle is corrupt: {exc}") from exc
    try:
        manifest = json.loads((extraction / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CatalogueBundleError("Data Catalogue training manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise CatalogueBundleError("Data Catalogue training manifest schema is unsupported")
    identities = {
        "dataset_id": dataset["dataset_id"],
        "dataset_version": dataset["version"],
        "partition_id": dataset["partition_id"],
    }
    for field, expected in identities.items():
        if manifest.get(field) != expected:
            raise CatalogueBundleError(f"Data Catalogue training manifest {field} mismatch")
    feature_order = manifest.get("feature_order")
    if (
        not isinstance(feature_order, list)
        or not feature_order
        or any(not isinstance(item, str) or not item for item in feature_order)
        or manifest.get("feature_order_sha256") != _feature_order_hash(feature_order)
    ):
        raise CatalogueBundleError("Data Catalogue feature-order checksum mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != EXPECTED_FILES - {"manifest.json"}:
        raise CatalogueBundleError("Data Catalogue training manifest file list is invalid")
    for filename, record in files.items():
        path = extraction / filename
        if not isinstance(record, dict) or record.get("sha256") != _sha256(path):
            raise CatalogueBundleError(f"Data Catalogue {filename} checksum mismatch")
        if record.get("bytes") != path.stat().st_size:
            raise CatalogueBundleError(f"Data Catalogue {filename} size mismatch")
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or counts.get("labels") != counts.get("test"):
        raise CatalogueBundleError("Data Catalogue labels do not cover the test partition")
    checksum = manifest.get("partition_checksum")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise CatalogueBundleError("Data Catalogue partition checksum is missing")
    actual_counts = {}
    for filename in ("train.csv", "test.csv", "labels.csv"):
        with (extraction / filename).open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.reader(source))
        if not rows or any(not row for row in rows):
            raise CatalogueBundleError(f"Data Catalogue {filename} is empty or invalid")
        actual_counts[filename] = len(rows)
        expected_width = 1 if filename == "labels.csv" else len(feature_order)
        if any(len(row) != expected_width for row in rows):
            raise CatalogueBundleError(f"Data Catalogue {filename} shape mismatch")
    if (
        actual_counts["train.csv"] != counts.get("train")
        or actual_counts["test.csv"] != counts.get("test")
        or actual_counts["labels.csv"] != counts.get("labels")
        or counts.get("features") != len(feature_order)
    ):
        raise CatalogueBundleError("Data Catalogue training manifest count mismatch")
    return {
        "directory": extraction,
        "temporary_root": archive_path.parent,
        "files": {
            "train": extraction / "train.csv",
            "test": extraction / "test.csv",
            "labels": extraction / "labels.csv",
        },
        "details": manifest.get("dataset_details", {}),
        "manifest": manifest,
    }


def configured_client(config: dict[str, Any]) -> CatalogueBundleClient:
    return CatalogueBundleClient(
        config["API_DATA_CATALOGUE_URL"],
        timeout=(
            float(config["CATALOGUE_CONNECT_TIMEOUT_SECONDS"]),
            float(config["CATALOGUE_READ_TIMEOUT_SECONDS"]),
        ),
        maximum_bytes=int(config["CATALOGUE_MAX_BUNDLE_BYTES"]),
    )
