"""Dataset versioning, annotations, labels, partitions, and usability."""

from __future__ import annotations

import copy
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

from .models import utc_now, validate_source, version_summary
from .repository import (
    FileCatalogueRepository,
    atomic_write_bytes,
    atomic_write_json,
)
from .validation import (
    ANOMALY_SCENARIOS,
    CatalogueValidationError,
    partition_checksum,
    require_object,
    require_string,
    validate_annotations,
    validate_identifier,
    validate_partition,
    validate_purpose,
    validate_tags,
    validate_timestamp,
    validate_version,
    validate_workload_context,
)


LABEL_SEMANTICS = {"0": "normal", "1": "anomaly"}


class CatalogueConflictError(RuntimeError):
    """Raised when an immutable version overlay would be overwritten."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_request(source: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(source[key])
        for key in (
            "type",
            "prometheus_source_id",
            "application",
            "namespace",
            "requested_targets",
            "start_time",
            "end_time",
            "sampling_interval_seconds",
            "queries",
        )
        if key in source
    }


def _new_empty_schema() -> dict[str, Any]:
    return {
        "modality": "metrics",
        "timestamp_column": "timestamp",
        "timestamp_timezone": "UTC",
        "observation_count": 0,
        "feature_count": 0,
        "feature_identity": "named",
        "feature_order": [],
        "missing_values": {"total": 0, "by_feature": {}},
        "normalization": "none",
        "imputation": "none",
    }


def create_dataset_version(
    repository: FileCatalogueRepository, dataset_id: str, payload: Any
) -> dict[str, Any]:
    payload = require_object(payload, "request body")
    dataset_id = validate_identifier(dataset_id)
    record = repository.read_record(dataset_id)
    copy_from = payload.get("copy_from_version", record["latest_version"])
    copy_from = validate_version(copy_from)
    base = repository.read_version(dataset_id, copy_from)

    allowed = {
        "copy_from_version",
        "source",
        "workload_context",
        "partition",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise CatalogueValidationError(
            f"unsupported version fields: {', '.join(sorted(unknown))}"
        )
    source_request = _source_request(base["source"])
    source_updates = payload.get("source", {})
    if not isinstance(source_updates, dict):
        raise CatalogueValidationError("source must be an object")
    source_request.update(copy.deepcopy(source_updates))
    source, source_start, source_end = validate_source(source_request)

    partition_input = payload.get("partition", {"mode": "none"})
    partition = validate_partition(partition_input)
    if partition["mode"] == "time_range":
        for name in ("train", "test"):
            _value, start = validate_timestamp(
                partition[name]["start_time"], f"partition.{name}.start_time"
            )
            _value, end = validate_timestamp(
                partition[name]["end_time"], f"partition.{name}.end_time"
            )
            if start < source_start or end > source_end:
                raise CatalogueValidationError(
                    f"partition.{name} range must be inside the source time window"
                )
    workload_context = validate_workload_context(
        payload.get("workload_context", record["workload_context"])
    )
    number = max([item["version"] for item in record.get("versions", [])] or [0]) + 1
    created_at = utc_now()
    version = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "version": number,
        "status": "draft",
        "created_at": created_at,
        "copied_from_version": copy_from,
        "workload_context": workload_context,
        "source": source,
        "provenance": {
            "configuration_frozen": False,
            "copied_from_version": copy_from,
            "resolved_queries": [],
            "query_results": [],
            "fetch_errors": [],
            "query_warnings": [],
        },
        "technical_schema": _new_empty_schema(),
        "artifacts": {},
        "checksums": {},
        "ground_truth": {
            "labels_available": False,
            "label_semantics": {},
            "known_incident_windows": [],
            "active_label_source": None,
        },
        "incident_context": [],
        "partition": partition,
        "partition_history": [],
        "validation": {"status": "not_run", "warnings": [], "errors": []},
    }
    record = copy.deepcopy(record)
    record["latest_version"] = number
    record["status"] = "draft"
    record["technical_schema"] = version["technical_schema"]
    record["ground_truth"] = {
        "labels_available": False,
        "known_incident_count": 0,
    }
    if record.get("workload_context") != workload_context:
        record["workload_context"] = workload_context
        revision = int(record.get("metadata_revision", 1)) + 1
        record["metadata_revision"] = revision
        record.setdefault("metadata_history", []).append(
            {
                "revision": revision,
                "updated_at": created_at,
                "changed_fields": ["workload_context"],
                "reason": f"dataset version {number} created",
            }
        )
    record.setdefault("versions", []).append(version_summary(version))
    repository.create_version(dataset_id, record, version)
    return repository.read_detail(dataset_id)


def update_dataset_metadata(
    repository: FileCatalogueRepository, dataset_id: str, payload: Any
) -> dict[str, Any]:
    payload = require_object(payload, "request body")
    allowed = {
        "display_name",
        "description",
        "purpose",
        "tags",
        "workload_context",
        "ground_truth_annotations",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise CatalogueValidationError(
            "metadata update may contain only display_name, description, purpose, tags, "
            "workload_context and ground_truth_annotations"
        )
    if not payload:
        raise CatalogueValidationError("metadata update must contain at least one field")
    record = repository.read_record(dataset_id)
    changed = []
    if "display_name" in payload:
        record["display_name"] = require_string(payload["display_name"], "display_name")
        changed.append("display_name")
    if "description" in payload:
        record["description"] = require_string(
            payload["description"], "description", allow_empty=True
        )
        changed.append("description")
    if "purpose" in payload:
        record["purpose"] = validate_purpose(payload["purpose"])
        changed.append("purpose")
    if "tags" in payload:
        record["tags"] = validate_tags(payload["tags"])
        changed.append("tags")
    if "workload_context" in payload:
        record["workload_context"] = validate_workload_context(payload["workload_context"])
        changed.append("workload_context")
    if "ground_truth_annotations" in payload:
        annotations = validate_annotations(payload["ground_truth_annotations"])
        now = utc_now()
        record["ground_truth_annotations"] = [
            {
                "annotation_id": f"annotation-{uuid.uuid4().hex}",
                **item,
                "created_at": now,
            }
            for item in annotations
        ]
        changed.append("ground_truth_annotations")
    revision = int(record.get("metadata_revision", 1)) + 1
    record["metadata_revision"] = revision
    record.setdefault("metadata_history", []).append(
        {"revision": revision, "updated_at": utc_now(), "changed_fields": sorted(changed)}
    )
    repository.update_record(dataset_id, record)
    return repository.read_detail(dataset_id)


def _validate_string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise CatalogueValidationError(f"{field} must be a list of non-empty strings")
    return [item.strip() for item in value]


def add_incident(
    repository: FileCatalogueRepository,
    dataset_id: str,
    version_number: int,
    payload: Any,
) -> dict[str, Any]:
    payload = require_object(payload, "request body")
    allowed = {
        "incident_id",
        "start_time",
        "end_time",
        "scenario",
        "affected_services",
        "affected_metrics",
        "annotation",
        "source",
        "allow_overlap",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise CatalogueValidationError(f"unsupported incident fields: {', '.join(sorted(unknown))}")
    version = repository.read_version(dataset_id, version_number)
    if version["status"] != "available":
        raise CatalogueValidationError("incidents can be added only to available versions")
    incident_id = validate_identifier(
        payload.get("incident_id") or f"incident-{uuid.uuid4().hex}", "incident_id"
    )
    start_value, start = validate_timestamp(payload.get("start_time"), "start_time")
    end_value, end = validate_timestamp(payload.get("end_time"), "end_time")
    if start >= end:
        raise CatalogueValidationError("incident start_time must be before end_time")
    source_start_value = version["source"].get("start_time")
    source_end_value = version["source"].get("end_time")
    if not source_start_value or not source_end_value:
        raise CatalogueValidationError("dataset version has no timestamped source window")
    _value, source_start = validate_timestamp(source_start_value, "source.start_time")
    _value, source_end = validate_timestamp(source_end_value, "source.end_time")
    if start < source_start or end > source_end:
        raise CatalogueValidationError("incident range must be inside the dataset time window")
    scenario = payload.get("scenario")
    if scenario not in ANOMALY_SCENARIOS:
        raise CatalogueValidationError(
            f"scenario must be one of: {', '.join(sorted(ANOMALY_SCENARIOS))}"
        )
    allow_overlap = payload.get("allow_overlap", False)
    if not isinstance(allow_overlap, bool):
        raise CatalogueValidationError("allow_overlap must be boolean")
    incidents = copy.deepcopy(version.get("incident_context", []))
    if any(item.get("incident_id") == incident_id for item in incidents):
        raise CatalogueValidationError(f"duplicate incident_id: {incident_id!r}")
    if not allow_overlap:
        for item in incidents:
            _value, existing_start = validate_timestamp(item["start_time"], "incident.start_time")
            _value, existing_end = validate_timestamp(item["end_time"], "incident.end_time")
            if start < existing_end and end > existing_start:
                raise CatalogueValidationError("incident windows must not overlap")
    incident = {
        "incident_id": incident_id,
        "start_time": start_value,
        "end_time": end_value,
        "scenario": scenario,
        "affected_services": _validate_string_list(
            payload.get("affected_services"), "affected_services"
        ),
        "affected_metrics": _validate_string_list(
            payload.get("affected_metrics"), "affected_metrics"
        ),
        "annotation": require_string(payload.get("annotation", ""), "annotation", allow_empty=True),
        "source": require_string(payload.get("source", "manual"), "source"),
        "created_at": utc_now(),
    }
    incidents.append(incident)
    overlay = repository.overlay_directory(dataset_id, version_number)
    incident_path = overlay / "incidents.json"
    incident_document = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "version": version_number,
        "incidents": incidents,
    }
    atomic_write_json(incident_path, incident_document)
    incident_bytes = incident_path.read_bytes()
    version["incident_context"] = incidents
    version.setdefault("ground_truth", {})["known_incident_windows"] = incidents
    version["ground_truth"]["incident_artifact"] = {
        "path": str(incident_path.relative_to(repository.root)),
        "sha256": _sha256_bytes(incident_bytes),
    }
    repository.update_version(dataset_id, version_number, version)
    _sync_ground_truth_summary(repository, version)
    return incident


def _timestamps(repository: FileCatalogueRepository, version: dict[str, Any]) -> list[str]:
    path = (
        repository.artifact_directory(version["dataset_id"], version["version"])
        / "canonical"
        / "timestamps.csv"
    )
    if not path.is_file():
        raise CatalogueValidationError("dataset version has no canonical timestamp artifact")
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "timestamp" not in reader.fieldnames:
            raise CatalogueValidationError("timestamp artifact has an invalid schema")
        return [row["timestamp"] for row in reader]


def _sync_ground_truth_summary(
    repository: FileCatalogueRepository, version: dict[str, Any]
) -> None:
    record = repository.read_record(version["dataset_id"])
    record["versions"] = [
        version_summary(version) if item["version"] == version["version"] else item
        for item in record["versions"]
    ]
    if record["latest_version"] == version["version"]:
        record["ground_truth"] = {
            "labels_available": version.get("ground_truth", {}).get("labels_available", False),
            "known_incident_count": len(version.get("incident_context", [])),
        }
    repository.update_record(version["dataset_id"], record)


def add_row_labels(
    repository: FileCatalogueRepository,
    dataset_id: str,
    version_number: int,
    payload: Any,
) -> dict[str, Any]:
    payload = require_object(payload, "request body")
    version = repository.read_version(dataset_id, version_number)
    if version["status"] != "available":
        raise CatalogueValidationError("labels can be added only to available versions")
    ground_truth = version.setdefault("ground_truth", {})
    existing_partition_labels = version.get("partition", {}).get("labels")
    if ground_truth.get("active_label_source") or existing_partition_labels:
        raise CatalogueConflictError("row-level labels already exist and cannot be overwritten")
    source_type = payload.get("source")
    observation_count = version["technical_schema"].get("observation_count", 0)
    if source_type == "row_level":
        labels = payload.get("labels")
        if not isinstance(labels, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item not in {0, 1}
            for item in labels
        ):
            raise CatalogueValidationError("labels must be a list containing only 0 and 1")
        derivation = None
    elif source_type == "incident_windows":
        incidents = version.get("incident_context", [])
        if not incidents:
            raise CatalogueValidationError("incident-derived labels require incident windows")
        timestamps = _timestamps(repository, version)
        labels = []
        parsed_incidents = [
            (
                validate_timestamp(item["start_time"], "incident.start_time")[1],
                validate_timestamp(item["end_time"], "incident.end_time")[1],
            )
            for item in incidents
        ]
        for timestamp in timestamps:
            _value, parsed = validate_timestamp(timestamp, "timestamp")
            labels.append(
                1 if any(start <= parsed <= end for start, end in parsed_incidents) else 0
            )
        derivation = {
            "method": "timestamp_within_incident_window_inclusive",
            "incident_count": len(incidents),
            "derived_at": utc_now(),
        }
    else:
        raise CatalogueValidationError("label source must be row_level or incident_windows")
    if len(labels) != observation_count:
        raise CatalogueValidationError(
            f"label count {len(labels)} must equal observation count {observation_count}"
        )
    content = "label\n" + "".join(f"{item}\n" for item in labels)
    encoded = content.encode("utf-8")
    path = repository.overlay_directory(dataset_id, version_number) / "labels" / "row_labels.csv"
    if path.exists():
        raise CatalogueConflictError("row-level label artifact already exists")
    atomic_write_bytes(path, encoded)
    label_source = {
        "type": source_type,
        "artifact": str(path.relative_to(repository.root)),
        "sha256": _sha256_bytes(encoded),
        "row_count": len(labels),
        "semantics": LABEL_SEMANTICS,
        "derivation": derivation,
        "created_at": utc_now(),
    }
    ground_truth.update(
        {
            "labels_available": True,
            "label_semantics": LABEL_SEMANTICS,
            "active_label_source": label_source,
        }
    )
    repository.update_version(dataset_id, version_number, version)
    _sync_ground_truth_summary(repository, version)
    return label_source


def _feature_order_reference(version: dict[str, Any]) -> dict[str, Any]:
    order = version["technical_schema"].get("feature_order", [])
    encoded = json.dumps(order, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {
        "schema_field": "technical_schema.feature_order",
        "feature_count": len(order),
        "sha256": _sha256_bytes(encoded),
    }


def set_partition(
    repository: FileCatalogueRepository,
    dataset_id: str,
    version_number: int,
    payload: Any,
) -> dict[str, Any]:
    payload = require_object(payload, "request body")
    version = repository.read_version(dataset_id, version_number)
    if version["status"] != "available":
        raise CatalogueValidationError("partitions can be defined only for available versions")
    labeled_evaluation = payload.get("labeled_evaluation", False)
    if not isinstance(labeled_evaluation, bool):
        raise CatalogueValidationError("labeled_evaluation must be boolean")
    partition_input = {key: copy.deepcopy(value) for key, value in payload.items() if key != "labeled_evaluation"}
    partition = validate_partition(partition_input)
    partition.pop("checksum", None)
    observation_count = version["technical_schema"].get("observation_count", 0)
    if partition["mode"] == "time_range":
        timestamps = _timestamps(repository, version)
        parsed_timestamps = [validate_timestamp(item, "timestamp")[1] for item in timestamps]
        source_start = validate_timestamp(version["source"]["start_time"], "source.start_time")[1]
        source_end = validate_timestamp(version["source"]["end_time"], "source.end_time")[1]
        counts = {}
        for name in ("train", "test"):
            start = validate_timestamp(partition[name]["start_time"], f"partition.{name}.start_time")[1]
            end = validate_timestamp(partition[name]["end_time"], f"partition.{name}.end_time")[1]
            if start < source_start or end > source_end:
                raise CatalogueValidationError(
                    f"partition.{name} range must be inside the dataset time window"
                )
            counts[name] = sum(1 for timestamp in parsed_timestamps if start <= timestamp <= end)
            if counts[name] < 2:
                raise CatalogueValidationError(
                    f"partition.{name} must contain at least two observations"
                )
        partition["train"]["observation_count"] = counts["train"]
        partition["test"]["observation_count"] = counts["test"]
    elif partition["mode"] == "predefined":
        if partition["train"]["row_count"] < 2 or partition["test"]["row_count"] < 2:
            raise CatalogueValidationError("predefined train and test partitions need at least two rows")
        if partition["train"]["row_count"] + partition["test"]["row_count"] > observation_count:
            raise CatalogueValidationError("predefined partition row counts exceed the dataset")

    active_labels = version.get("ground_truth", {}).get("active_label_source")
    predefined_labels = partition.get("labels")
    if predefined_labels and predefined_labels.get("semantics") != LABEL_SEMANTICS:
        raise CatalogueValidationError(
            "predefined labels must use 0 = normal and 1 = anomaly semantics"
        )
    labels_available = bool(active_labels or predefined_labels)
    if labeled_evaluation and partition["mode"] == "none":
        raise CatalogueValidationError(
            "labeled evaluation requires an explicit train/test partition"
        )
    if labeled_evaluation and not labels_available:
        raise CatalogueValidationError("labeled evaluation requires row-level labels")
    if labeled_evaluation and active_labels and active_labels["row_count"] != observation_count:
        raise CatalogueValidationError("active labels do not cover all dataset observations")
    if labeled_evaluation and predefined_labels:
        if predefined_labels["row_count"] != partition["test"]["row_count"]:
            raise CatalogueValidationError("predefined labels do not cover the test partition")

    partition["source_dataset_version"] = version_number
    partition["feature_order_reference"] = _feature_order_reference(version)
    partition["labels_source"] = (
        "active_row_labels" if active_labels else ("predefined" if predefined_labels else "none")
    )
    partition["labeled_evaluation"] = labeled_evaluation
    partition["checksum"] = partition_checksum(partition)
    definition = {
        **partition,
        "partition_id": f"partition-{uuid.uuid4().hex}",
        "created_at": utc_now(),
    }
    path = (
        repository.overlay_directory(dataset_id, version_number)
        / "partitions"
        / f"{definition['partition_id']}.json"
    )
    atomic_write_json(path, definition)
    definition["artifact"] = str(path.relative_to(repository.root))
    version["partition"] = definition
    version.setdefault("partition_history", []).append(
        {
            "partition_id": definition["partition_id"],
            "checksum": definition["checksum"],
            "artifact": definition["artifact"],
            "created_at": definition["created_at"],
        }
    )
    repository.update_version(dataset_id, version_number, version)
    _sync_ground_truth_summary(repository, version)
    return definition


def calculate_usability(version: dict[str, Any]) -> dict[str, Any]:
    unsupervised_reasons = []
    if version.get("status") != "available":
        unsupervised_reasons.append("Dataset version is not available")
    schema = version.get("technical_schema", {})
    if schema.get("observation_count", 0) < 2:
        unsupervised_reasons.append("At least two observations are required")
    if schema.get("feature_count", 0) < 1:
        unsupervised_reasons.append("At least one feature is required")

    evaluation_reasons = []
    ground_truth = version.get("ground_truth", {})
    partition = version.get("partition", {"mode": "none"})
    active = ground_truth.get("active_label_source")
    predefined = partition.get("labels") if partition.get("mode") == "predefined" else None
    if not active and not predefined:
        evaluation_reasons.append(
            "No row-level labels or incident-derived labels are available"
        )
    if partition.get("mode") == "none":
        evaluation_reasons.append("No explicit train/test partition is defined")
    if version.get("status") != "available":
        evaluation_reasons.append("Dataset version is not available")
    if active and active.get("row_count") != schema.get("observation_count"):
        evaluation_reasons.append("Row-level labels do not cover all observations")
    if predefined and predefined.get("row_count") != partition.get("test", {}).get("row_count"):
        evaluation_reasons.append("Predefined labels do not cover the test partition")
    return {
        "usages": {
            "unsupervised_training": {
                "supported": not unsupervised_reasons,
                "reasons": unsupervised_reasons,
            },
            "labeled_evaluation": {
                "supported": not evaluation_reasons,
                "reasons": evaluation_reasons,
            },
        }
    }
