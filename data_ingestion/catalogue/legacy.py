"""Read-only projection of legacy learning-adaptation datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .models import utc_now, version_summary
from .repository import DatasetAlreadyExistsError, FileCatalogueRepository
from .validation import CatalogueValidationError, validate_identifier, validate_partition


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_facts(path: Path, name: str) -> dict[str, Any]:
    rows = 0
    columns = None
    values = set()
    with path.open("r", encoding="utf-8", newline="") as source:
        for row_number, row in enumerate(csv.reader(source), start=1):
            if not row or any(item.strip() == "" for item in row):
                raise ValueError(f"{name} contains an empty value at row {row_number}")
            if columns is None:
                columns = len(row)
            elif len(row) != columns:
                raise ValueError(f"{name} is not rectangular at row {row_number}")
            for item in row:
                number = float(item)
                if not math.isfinite(number):
                    raise ValueError(f"{name} contains a non-finite value at row {row_number}")
                if name == "labels":
                    values.add(number)
            rows += 1
    if rows == 0 or not columns:
        raise ValueError(f"{name} is empty")
    return {
        "row_count": rows,
        "feature_count": columns,
        "values": sorted(values),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


class LegacyDatasetProjector:
    def __init__(self, legacy_root: str | Path, repository: FileCatalogueRepository):
        self.legacy_root = Path(legacy_root).resolve()
        self.repository = repository

    def project_all(self) -> None:
        if not self.legacy_root.is_dir():
            return
        for candidate in sorted(self.legacy_root.iterdir()):
            resolved = candidate.resolve()
            if not resolved.is_relative_to(self.legacy_root) or not resolved.is_dir():
                continue
            try:
                dataset_id = validate_identifier(candidate.name)
            except CatalogueValidationError:
                continue
            try:
                self.repository.read_record(dataset_id)
                continue
            except LookupError:
                pass
            projected = self._project(candidate, dataset_id)
            if projected is None:
                continue
            try:
                self.repository.create(*projected)
            except DatasetAlreadyExistsError:
                continue

    def _safe_file(self, directory: Path, filename: str) -> Path | None:
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            return None
        return path

    def _project(
        self, directory: Path, dataset_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        file_names = {
            "train": f"train_{dataset_id}",
            "test": f"test_{dataset_id}",
            "labels": f"test_label_{dataset_id}",
        }
        files = {
            name: self._safe_file(directory, filename)
            for name, filename in file_names.items()
        }
        if any(path is None for path in files.values()):
            return None

        warnings = []
        errors = []
        facts = {}
        for name, path in files.items():
            try:
                facts[name] = _csv_facts(path, name)
            except (OSError, UnicodeError, ValueError) as exc:
                errors.append(str(exc))

        details_path = self._safe_file(directory, "details.json")
        details = None
        if details_path is None:
            warnings.append("details.json is missing; legacy metadata and feature names are incomplete")
        else:
            try:
                details = json.loads(details_path.read_text(encoding="utf-8"))
                if not isinstance(details, dict):
                    raise ValueError("details.json root is not an object")
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                warnings.append(f"details.json is invalid: {exc}")
                details = None

        if not errors:
            if facts["train"]["feature_count"] != facts["test"]["feature_count"]:
                errors.append("train and test feature counts differ")
            if facts["labels"]["feature_count"] != 1:
                errors.append("labels must contain exactly one column")
            if facts["labels"]["row_count"] != facts["test"]["row_count"]:
                errors.append("label row count differs from test row count")
            if not set(facts["labels"]["values"]).issubset({0.0, 1.0}):
                errors.append("labels contain values other than 0 and 1")

        train_rows = facts.get("train", {}).get("row_count", 0)
        test_rows = facts.get("test", {}).get("row_count", 0)
        feature_count = facts.get("train", {}).get("feature_count", 0)
        feature_order = [f"feature_{index}" for index in range(feature_count)]
        feature_identity = "positional"
        display_name = dataset_id
        sampling_interval = None
        if details is not None:
            display_name = details.get("dataset") or dataset_id
            containers = details.get("containers")
            metrics = details.get("metrics")
            if isinstance(containers, list) and isinstance(metrics, list):
                proposed = [
                    f"{container}_{metric}"
                    for container in containers
                    for metric in metrics
                ]
                if len(proposed) == feature_count and len(set(proposed)) == len(proposed):
                    feature_order = proposed
                    feature_identity = "named"
                else:
                    warnings.append(
                        "details.json containers and metrics do not define the actual feature count uniquely"
                    )
            else:
                warnings.append("details.json containers or metrics are missing or invalid")
            declared_entries = details.get("data_entries")
            if declared_entries is not None and declared_entries != test_rows:
                warnings.append(
                    f"details.json data_entries={declared_entries} differs from actual test row count {test_rows}"
                )
            step = details.get("step_size")
            if isinstance(step, int) and not isinstance(step, bool) and step > 0:
                sampling_interval = step

        created_at = utc_now()
        labels_available = not errors and "labels" in facts
        status = "failed" if errors else ("available" if details is not None else "draft")
        technical_schema = {
            "modality": "metrics",
            "timestamp_column": None,
            "timestamp_timezone": None,
            "observation_count": train_rows + test_rows,
            "train_observation_count": train_rows,
            "test_observation_count": test_rows,
            "feature_count": feature_count,
            "feature_identity": feature_identity,
            "feature_order": feature_order,
            "missing_values": {"total": 0, "by_feature": {}},
            "normalization": "unknown",
            "imputation": "unknown",
        }
        artifact_records = {}
        checksums = {}
        for name, path in files.items():
            if name not in facts:
                continue
            relative_path = str(path.relative_to(self.legacy_root))
            artifact_records[name] = {
                "storage": "legacy",
                "path": relative_path,
                "sha256": facts[name]["sha256"],
                "bytes": facts[name]["bytes"],
            }
            checksums[name] = facts[name]["sha256"]

        partition_value = {"mode": "none"}
        if not errors:
            partition_value = {
                "mode": "predefined",
                "train": {"artifact": artifact_records["train"]["path"], "row_count": train_rows},
                "test": {"artifact": artifact_records["test"]["path"], "row_count": test_rows},
                "labels": {
                    "artifact": artifact_records["labels"]["path"],
                    "row_count": facts["labels"]["row_count"],
                    "semantics": {"0": "normal", "1": "anomaly"},
                },
            }
        partition = validate_partition(partition_value)
        ground_truth = {
            "labels_available": labels_available,
            "label_semantics": {"0": "normal", "1": "anomaly"} if labels_available else {},
            "known_incident_windows": [],
        }
        version = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "version": 1,
            "status": status,
            "created_at": created_at,
            "workload_context": {
                "workload_intensity": None,
                "dominant_workload_characteristic": None,
                "anomaly_scenario": "Unknown",
            },
            "source": {
                "type": "legacy",
                "prometheus_source_id": None,
                "application": None,
                "namespace": None,
                "requested_targets": {"services": [], "pods": [], "label_selectors": []},
                "resolved_targets": {"services": [], "pods": [], "label_selectors": []},
                "start_time": None,
                "end_time": None,
                "sampling_interval_seconds": sampling_interval,
                "queries": [],
                "fetch_timestamp": None,
            },
            "provenance": {
                "configuration_frozen": True,
                "origin": "learning_adaptation/datasets",
                "prometheus_provenance": "unknown",
                "legacy_details": details,
                "resolved_queries": [],
                "query_results": [],
                "fetch_errors": [],
                "query_warnings": [],
            },
            "technical_schema": technical_schema,
            "artifacts": artifact_records,
            "checksums": checksums,
            "ground_truth": ground_truth,
            "incident_context": [],
            "partition": partition,
            "validation": {
                "status": "failed" if errors else ("passed_with_warnings" if warnings else "passed"),
                "warnings": warnings,
                "errors": errors,
            },
        }
        record = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "display_name": display_name,
            "description": "",
            "purpose": ["training", "evaluation"],
            "status": status,
            "created_at": created_at,
            "created_by": None,
            "metadata_revision": 1,
            "latest_version": 1,
            "workload_context": {
                "workload_intensity": None,
                "dominant_workload_characteristic": None,
                "anomaly_scenario": "Unknown",
            },
            "ground_truth": {
                "labels_available": labels_available,
                "known_incident_count": 0,
            },
            "technical_schema": technical_schema,
            "versions": [version_summary(version)],
        }
        return record, version
