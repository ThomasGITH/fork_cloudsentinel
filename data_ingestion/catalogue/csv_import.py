"""Validated CSV ingestion into immutable catalogue artefacts."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median
from typing import Any
import uuid

from .lifecycle import add_row_labels, calculate_usability, set_partition
from .models import build_draft_dataset, utc_now, version_summary
from .repository import FileCatalogueRepository, atomic_write_bytes, atomic_write_json
from .validation import CatalogueValidationError, validate_timestamp


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_OBSERVATIONS = 200_000
MAX_FEATURES = 100


def _csv_bytes(header: list[str], rows: list[list[Any]]) -> bytes:
    target = io.StringIO(newline="")
    writer = csv.writer(target)
    writer.writerow(header)
    writer.writerows(rows)
    return target.getvalue().encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse(content: bytes, timestamp_column: str, label_column: str | None) -> dict[str, Any]:
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise CatalogueValidationError("CSV must be a non-empty file no larger than 25 MiB")
    try:
        rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    except (UnicodeError, csv.Error) as exc:
        raise CatalogueValidationError("CSV must be valid UTF-8 with a readable header") from exc
    if len(rows) < 5:
        raise CatalogueValidationError("CSV needs a header and at least four observations")
    header = [value.strip() for value in rows[0]]
    if not header or any(not value for value in header) or len(set(header)) != len(header):
        raise CatalogueValidationError("CSV column names must be non-empty and unique")
    if timestamp_column not in header:
        raise CatalogueValidationError("The configured timestamp column was not found")
    if label_column and label_column not in header:
        raise CatalogueValidationError("The configured label column was not found")
    if label_column == timestamp_column:
        raise CatalogueValidationError("Timestamp and label columns must be different")
    features = [value for value in header if value not in {timestamp_column, label_column}]
    if not features or len(features) > MAX_FEATURES:
        raise CatalogueValidationError("CSV must contain between 1 and 100 numeric feature columns")
    raw_rows = rows[1:]
    if len(raw_rows) > MAX_OBSERVATIONS:
        raise CatalogueValidationError("CSV exceeds the 200,000-observation limit")
    timestamp_index = header.index(timestamp_column)
    label_index = header.index(label_column) if label_column else None
    feature_indices = [header.index(value) for value in features]
    timestamps: list[str] = []
    parsed_times: list[datetime] = []
    matrix: list[list[float]] = []
    labels: list[int] = []
    for number, row in enumerate(raw_rows, start=2):
        if len(row) != len(header):
            raise CatalogueValidationError(f"CSV row {number} does not match the header width")
        timestamp, parsed = validate_timestamp(row[timestamp_index].strip(), f"CSV row {number} timestamp")
        if parsed_times and parsed <= parsed_times[-1]:
            raise CatalogueValidationError("CSV timestamps must be strictly increasing")
        parsed_times.append(parsed)
        timestamps.append(timestamp)
        values = []
        for index in feature_indices:
            try:
                value = float(row[index])
            except ValueError as exc:
                raise CatalogueValidationError(f"CSV row {number} contains a non-numeric feature") from exc
            if not math.isfinite(value):
                raise CatalogueValidationError(f"CSV row {number} contains a non-finite feature")
            values.append(value)
        matrix.append(values)
        if label_index is not None:
            raw_label = row[label_index].strip()
            if raw_label not in {"0", "1"}:
                raise CatalogueValidationError("Label values must use 0 = normal and 1 = anomaly")
            labels.append(int(raw_label))
    intervals = [
        (right - left).total_seconds() for left, right in zip(parsed_times, parsed_times[1:])
    ]
    return {
        "features": features,
        "timestamps": timestamps,
        "parsed_times": parsed_times,
        "matrix": matrix,
        "labels": labels,
        "sampling_interval_seconds": max(1, int(round(median(intervals)))),
    }


def import_csv_dataset(
    repository: FileCatalogueRepository,
    metadata: dict[str, Any],
    content: bytes,
    filename: str,
) -> dict[str, Any]:
    timestamp_column = str(metadata.get("timestamp_column") or "timestamp").strip()
    label_column = str(metadata.get("label_column") or "").strip() or None
    split_percent = metadata.get("train_split_percent", 70)
    if isinstance(split_percent, bool):
        raise CatalogueValidationError("Training split must be a percentage")
    try:
        split_percent = int(split_percent)
    except (TypeError, ValueError) as exc:
        raise CatalogueValidationError("Training split must be a whole-number percentage") from exc
    if split_percent < 20 or split_percent > 80:
        raise CatalogueValidationError("Training split must be between 20 and 80 percent")
    parsed = _parse(content, timestamp_column, label_column)
    count = len(parsed["timestamps"])
    split_index = max(2, min(count - 2, round(count * split_percent / 100)))
    workload = metadata.get("workload_context")
    payload = {
        "display_name": metadata.get("display_name"),
        "description": metadata.get("description", ""),
        "purpose": metadata.get("purpose"),
        "source": {
            "type": "prometheus",
            "prometheus_source_id": "cluster-default",
            "application": metadata.get("application") or "uploaded-dataset",
            "namespace": metadata.get("namespace") or "external",
            "requested_targets": {"services": [], "pods": [], "label_selectors": []},
            "start_time": parsed["timestamps"][0],
            "end_time": parsed["timestamps"][-1],
            "sampling_interval_seconds": parsed["sampling_interval_seconds"],
            "queries": [{
                "query_id": "uploaded_features",
                "display_name": "Uploaded feature columns",
                "promql_template": "uploaded_csv",
                "required": True,
                "execution_mode": "single",
            }],
        },
        "workload_context": workload,
        "ground_truth": {"labels_available": False, "label_semantics": {}, "known_incident_windows": []},
        "partition": {"mode": "none"},
    }
    record, version = build_draft_dataset(payload)
    dataset_id = record["dataset_id"]
    imported_at = utc_now()
    version["status"] = "available"
    version["source"] = {
        "type": "uploaded_csv",
        "filename": Path(filename).name[:200],
        "application": payload["source"]["application"],
        "namespace": payload["source"]["namespace"],
        "start_time": parsed["timestamps"][0],
        "end_time": parsed["timestamps"][-1],
        "sampling_interval_seconds": parsed["sampling_interval_seconds"],
        "requested_targets": {"services": [], "pods": [], "label_selectors": []},
        "resolved_targets": {"services": [], "pods": [], "label_selectors": []},
        "fetch_timestamp": imported_at,
        "queries": [],
    }
    version["technical_schema"] = {
        "schema_version": 1,
        "modality": "metrics",
        "timestamp_column": "timestamp",
        "timestamp_timezone": "UTC",
        "observation_count": count,
        "feature_count": len(parsed["features"]),
        "feature_identity": "named",
        "feature_order": parsed["features"],
        "missing_values": {"total": 0, "by_feature": {name: 0 for name in parsed["features"]}},
        "normalization": "none",
        "imputation": "none",
    }
    version["provenance"] = {
        "configuration_frozen": True,
        "origin": "browser_csv_upload",
        "original_file_sha256": _sha(content),
        "imported_at": imported_at,
        "timestamp_column": timestamp_column,
        "label_column": label_column,
        "resolved_queries": [],
        "query_results": [],
        "fetch_errors": [],
        "query_warnings": [],
    }
    version["validation"] = {"status": "passed", "warnings": [], "errors": []}
    record["status"] = "available"
    record["technical_schema"] = version["technical_schema"]
    record["versions"] = [version_summary(version)]
    repository.create(record, version)

    attempt_id = f"import-{uuid.uuid4().hex}"
    staging = repository.staging_directory(dataset_id, 1, attempt_id)
    observations = _csv_bytes(
        ["timestamp", *parsed["features"]],
        [[timestamp, *row] for timestamp, row in zip(parsed["timestamps"], parsed["matrix"])],
    )
    matrix = _csv_bytes(parsed["features"], parsed["matrix"])
    timestamps = _csv_bytes(["timestamp"], [[item] for item in parsed["timestamps"]])
    files = {
        "canonical/observations.csv": observations,
        "canonical/feature_matrix.csv": matrix,
        "canonical/timestamps.csv": timestamps,
    }
    for relative, value in files.items():
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(target, value)
    atomic_write_json(staging / "schema.json", version["technical_schema"])
    atomic_write_json(staging / "provenance.json", version["provenance"])
    atomic_write_json(staging / "validation.json", version["validation"])
    checksums = {
        str(path.relative_to(staging)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    }
    atomic_write_json(staging / "checksums.json", checksums)
    active = repository.promote_staging(dataset_id, 1, attempt_id)
    version = repository.read_version(dataset_id, 1)
    version["checksums"] = checksums
    version["artifacts"] = {
        relative: {
            "path": str((active / relative).relative_to(repository.root)),
            "sha256": digest,
            "bytes": (active / relative).stat().st_size,
        }
        for relative, digest in checksums.items()
    }
    repository.update_version(dataset_id, 1, version)
    if parsed["labels"]:
        add_row_labels(repository, dataset_id, 1, {"source": "row_level", "labels": parsed["labels"]})
    set_partition(
        repository,
        dataset_id,
        1,
        {
            "mode": "time_range",
            "train": {
                "start_time": parsed["timestamps"][0],
                "end_time": parsed["timestamps"][split_index - 1],
            },
            "test": {
                "start_time": parsed["timestamps"][split_index],
                "end_time": parsed["timestamps"][-1],
            },
            "gap_seconds": max(
                0,
                int((parsed["parsed_times"][split_index] - parsed["parsed_times"][split_index - 1]).total_seconds()),
            ),
            "labeled_evaluation": bool(parsed["labels"]),
        },
    )
    detail = repository.read_detail(dataset_id)
    return {**detail, **calculate_usability(detail["version"])}
