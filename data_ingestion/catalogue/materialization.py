"""Materialize exact catalogue partitions as checksummed training bundles."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any
import zipfile

from .repository import FileCatalogueRepository
from .validation import (
    CatalogueValidationError,
    partition_checksum,
    validate_identifier,
    validate_timestamp,
    validate_version,
)


class CatalogueMaterializationError(RuntimeError):
    """Raised when immutable catalogue data cannot be materialized safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_order_hash(feature_order: list[str]) -> str:
    encoded = json.dumps(
        feature_order, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_relative(root: Path, reference: str, field: str) -> Path:
    path = (root / reference).resolve()
    if not path.is_relative_to(root.resolve()):
        raise CatalogueMaterializationError(f"{field} resolves outside its storage root")
    if not path.is_file() or path.is_symlink():
        raise CatalogueMaterializationError(f"{field} artifact is missing")
    return path


def _partition_definition(
    repository: FileCatalogueRepository,
    version: dict[str, Any],
    partition_id: str,
) -> dict[str, Any]:
    partition_id = validate_identifier(partition_id, "partition_id")
    current = version.get("partition") or {}
    if current.get("partition_id") == partition_id:
        artifact = current.get("artifact")
        if artifact:
            path = _safe_relative(repository.root, artifact, "partition")
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = dict(current)
    else:
        match = next(
            (
                item
                for item in version.get("partition_history", [])
                if item.get("partition_id") == partition_id
            ),
            None,
        )
        if match is None:
            raise CatalogueMaterializationError(
                f"unknown partition_id {partition_id!r} for this dataset version"
            )
        path = _safe_relative(repository.root, match["artifact"], "partition")
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("partition_id") != partition_id:
        raise CatalogueMaterializationError("partition artifact identity mismatch")
    checksum_value = {
        key: item
        for key, item in value.items()
        if key not in {"checksum", "partition_id", "created_at", "artifact"}
    }
    expected = partition_checksum(checksum_value)
    if value.get("checksum") != expected:
        raise CatalogueMaterializationError("partition checksum mismatch")
    return value


def _read_matrix(path: Path, name: str, *, allow_missing: bool) -> list[list[str]]:
    rows: list[list[str]] = []
    width: int | None = None
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            for row_number, row in enumerate(csv.reader(source), start=1):
                if not row:
                    raise CatalogueMaterializationError(
                        f"{name} contains an empty row at row {row_number}"
                    )
                if width is None:
                    width = len(row)
                elif len(row) != width:
                    raise CatalogueMaterializationError(f"{name} is not rectangular")
                for raw in row:
                    if raw.strip() == "":
                        if not allow_missing:
                            raise CatalogueMaterializationError(
                                f"{name} contains a missing value"
                            )
                        continue
                    try:
                        float(raw)
                    except ValueError as exc:
                        raise CatalogueMaterializationError(
                            f"{name} contains a non-numeric value"
                        ) from exc
                rows.append(row)
    except (OSError, UnicodeError) as exc:
        raise CatalogueMaterializationError(f"could not read {name}: {exc}") from exc
    if not rows or not width:
        raise CatalogueMaterializationError(f"{name} is empty")
    return rows


def _read_labels(path: Path, *, header: bool) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.reader(source))
    except (OSError, UnicodeError) as exc:
        raise CatalogueMaterializationError(f"could not read labels: {exc}") from exc
    if header:
        if not rows or rows[0] != ["label"]:
            raise CatalogueMaterializationError("label artifact has an invalid header")
        rows = rows[1:]
    labels = []
    for row in rows:
        if len(row) != 1:
            raise CatalogueMaterializationError("labels must contain exactly one column")
        value = float(row[0])
        if value not in {0.0, 1.0}:
            raise CatalogueMaterializationError("labels must contain only 0 and 1")
        labels.append(str(int(value)))
    if not labels:
        raise CatalogueMaterializationError("labels are empty")
    return labels


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        csv.writer(target).writerows(rows)


def _verify_known_artifact(
    path: Path,
    expected_sha256: str | None,
    field: str,
) -> str:
    actual = _sha256(path)
    if not expected_sha256 or actual != expected_sha256:
        raise CatalogueMaterializationError(f"{field} checksum mismatch")
    return actual


def _legacy_inputs(
    legacy_root: Path,
    version: dict[str, Any],
    partition: dict[str, Any],
) -> tuple[list[list[str]], list[list[str]], list[str], dict[str, str]]:
    values: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for name in ("train", "test", "labels"):
        part = partition.get(name)
        if not isinstance(part, dict):
            raise CatalogueMaterializationError(
                "selected partition has no labels required by current training flows"
            )
        path = _safe_relative(legacy_root, part["artifact"], f"partition.{name}")
        record = version.get("artifacts", {}).get(name, {})
        if record.get("path") != part["artifact"]:
            raise CatalogueMaterializationError(f"partition.{name} is not a known version artifact")
        hashes[name] = _verify_known_artifact(path, record.get("sha256"), name)
        values[name] = path
    train = _read_matrix(values["train"], "train", allow_missing=True)
    test = _read_matrix(values["test"], "test", allow_missing=True)
    labels = _read_labels(values["labels"], header=False)
    return train, test, labels, hashes


def _time_range_inputs(
    repository: FileCatalogueRepository,
    version: dict[str, Any],
    partition: dict[str, Any],
) -> tuple[list[list[str]], list[list[str]], list[str], dict[str, str]]:
    active = repository.artifact_directory(version["dataset_id"], version["version"])
    observations = active / "canonical" / "observations.csv"
    timestamps = active / "canonical" / "timestamps.csv"
    records = version.get("artifacts", {})
    hashes = {
        "canonical/observations.csv": _verify_known_artifact(
            observations,
            records.get("canonical/observations.csv", {}).get("sha256"),
            "canonical observations",
        ),
        "canonical/timestamps.csv": _verify_known_artifact(
            timestamps,
            records.get("canonical/timestamps.csv", {}).get("sha256"),
            "canonical timestamps",
        ),
    }
    feature_order = version["technical_schema"].get("feature_order", [])
    train_range = [
        validate_timestamp(partition["train"][field], f"partition.train.{field}")[1]
        for field in ("start_time", "end_time")
    ]
    test_range = [
        validate_timestamp(partition["test"][field], f"partition.test.{field}")[1]
        for field in ("start_time", "end_time")
    ]
    train: list[list[str]] = []
    test: list[list[str]] = []
    test_indices: list[int] = []
    try:
        with observations.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != ["timestamp", *feature_order]:
                raise CatalogueMaterializationError(
                    "canonical observations do not match the stored feature order"
                )
            for index, row in enumerate(reader):
                parsed = validate_timestamp(row["timestamp"], "observation timestamp")[1]
                values = [row[name] for name in feature_order]
                for raw in values:
                    if raw != "":
                        try:
                            float(raw)
                        except ValueError as exc:
                            raise CatalogueMaterializationError(
                                "canonical observations contain a non-numeric value"
                            ) from exc
                if train_range[0] <= parsed <= train_range[1]:
                    train.append(values)
                elif test_range[0] <= parsed <= test_range[1]:
                    test.append(values)
                    test_indices.append(index)
    except (OSError, UnicodeError) as exc:
        raise CatalogueMaterializationError(
            f"could not read canonical observations: {exc}"
        ) from exc

    label_source = version.get("ground_truth", {}).get("active_label_source")
    if partition.get("labels_source") != "active_row_labels" or not label_source:
        raise CatalogueMaterializationError(
            "selected partition has no labels required by current training flows"
        )
    label_path = _safe_relative(repository.root, label_source["artifact"], "labels")
    hashes["labels/row_labels.csv"] = _verify_known_artifact(
        label_path, label_source.get("sha256"), "labels"
    )
    all_labels = _read_labels(label_path, header=True)
    if len(all_labels) != version["technical_schema"].get("observation_count"):
        raise CatalogueMaterializationError("labels do not cover all observations")
    labels = [all_labels[index] for index in test_indices]
    return train, test, labels, hashes


def materialize_training_bundle(
    repository: FileCatalogueRepository,
    legacy_root: str | Path,
    dataset_id: str,
    version_number: int,
    partition_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Create a temporary ZIP for an exact, immutable catalogue partition."""
    dataset_id = validate_identifier(dataset_id)
    version_number = validate_version(version_number)
    partition_id = validate_identifier(partition_id, "partition_id")
    version = repository.read_version(dataset_id, version_number)
    if version.get("status") != "available":
        raise CatalogueMaterializationError("dataset version must be available")
    partition = _partition_definition(repository, version, partition_id)
    if partition.get("mode") == "none":
        raise CatalogueMaterializationError("selected partition has mode none")

    feature_order = version.get("technical_schema", {}).get("feature_order", [])
    if not feature_order or len(set(feature_order)) != len(feature_order):
        raise CatalogueMaterializationError("dataset feature order is missing or invalid")
    feature_hash = _feature_order_hash(feature_order)
    reference = partition.get("feature_order_reference")
    if reference and (
        reference.get("feature_count") != len(feature_order)
        or reference.get("sha256") != feature_hash
    ):
        raise CatalogueMaterializationError("feature-order checksum mismatch")

    if partition["mode"] == "predefined":
        train, test, labels, source_hashes = _legacy_inputs(
            Path(legacy_root).resolve(), version, partition
        )
        label_source = {"type": "predefined", "sha256": source_hashes["labels"]}
    elif partition["mode"] == "time_range":
        train, test, labels, source_hashes = _time_range_inputs(
            repository, version, partition
        )
        active = version["ground_truth"]["active_label_source"]
        label_source = {"type": active["type"], "sha256": active["sha256"]}
    else:
        raise CatalogueMaterializationError("unsupported partition mode")

    if len(train) < 2 or len(test) < 2:
        raise CatalogueMaterializationError(
            "partition must contain at least two train and two test observations"
        )
    feature_count = len(feature_order)
    if any(len(row) != feature_count for row in [*train, *test]):
        raise CatalogueMaterializationError("partition feature count mismatch")
    if len(labels) != len(test):
        raise CatalogueMaterializationError("labels do not cover the test partition")

    temporary_directory = Path(tempfile.mkdtemp(prefix="catalogue-training-bundle-"))
    try:
        files = {
            "train.csv": train,
            "test.csv": test,
            "labels.csv": [[item] for item in labels],
        }
        file_records = {}
        for filename, rows in files.items():
            path = temporary_directory / filename
            _write_csv(path, rows)
            file_records[filename] = {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        provenance_record = version.get("artifacts", {}).get("provenance.json", {})
        manifest = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "dataset_version": version_number,
            "partition_id": partition_id,
            "partition_checksum": partition["checksum"],
            "partition_mode": partition["mode"],
            "feature_order": feature_order,
            "feature_order_sha256": feature_hash,
            "label_source": label_source,
            "counts": {
                "train": len(train),
                "test": len(test),
                "labels": len(labels),
                "features": feature_count,
            },
            "source_artifact_checksums": source_hashes,
            "provenance_reference": {
                "artifact": provenance_record.get("path"),
                "sha256": provenance_record.get("sha256"),
            },
            "dataset_details": {
                "feature_order": feature_order,
                "step_size": version.get("source", {}).get("sampling_interval_seconds"),
                "data_entries": len(test),
                "catalogue_dataset_id": dataset_id,
                "catalogue_version": version_number,
                "partition_id": partition_id,
            },
            "files": file_records,
        }
        manifest_path = temporary_directory / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        bundle_path = temporary_directory / "training-bundle.zip"
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for filename in ("manifest.json", "train.csv", "test.csv", "labels.csv"):
                archive.write(temporary_directory / filename, arcname=filename)
        return bundle_path, manifest
    except Exception:
        import shutil

        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
