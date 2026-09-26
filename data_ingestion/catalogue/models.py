"""Schema builders for dataset records and immutable dataset versions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

from .validation import (
    CatalogueValidationError,
    require_object,
    require_string,
    validate_ground_truth,
    validate_identifier,
    validate_partition,
    validate_purpose,
    validate_queries,
    validate_targets,
    validate_timestamp,
    validate_workload_context,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_dataset_id() -> str:
    return f"ds_{uuid.uuid4().hex}"


def validate_source(value: Any) -> tuple[dict[str, Any], datetime, datetime]:
    source = require_object(value, "source")
    forbidden = {"url", "base_url", "prometheus_url", "credentials", "token"} & set(source)
    if forbidden:
        raise CatalogueValidationError(
            "source must reference a server-configured prometheus_source_id; URLs and credentials are not accepted"
        )
    if source.get("type", "prometheus") != "prometheus":
        raise CatalogueValidationError("source.type must be 'prometheus' for new datasets")
    source_id = validate_identifier(
        source.get("prometheus_source_id"), "source.prometheus_source_id"
    )
    start_value, start = validate_timestamp(source.get("start_time"), "source.start_time")
    end_value, end = validate_timestamp(source.get("end_time"), "source.end_time")
    if start >= end:
        raise CatalogueValidationError("source.start_time must be before source.end_time")
    sampling = source.get("sampling_interval_seconds")
    if isinstance(sampling, bool) or not isinstance(sampling, int) or sampling < 1:
        raise CatalogueValidationError(
            "source.sampling_interval_seconds must be a positive integer"
        )
    return (
        {
            "type": "prometheus",
            "prometheus_source_id": source_id,
            "application": require_string(source.get("application"), "source.application"),
            "namespace": require_string(source.get("namespace"), "source.namespace"),
            "requested_targets": validate_targets(source.get("requested_targets")),
            "resolved_targets": {
                "services": [],
                "pods": [],
                "label_selectors": [],
            },
            "start_time": start_value,
            "end_time": end_value,
            "sampling_interval_seconds": sampling,
            "queries": validate_queries(source.get("queries")),
            "fetch_timestamp": None,
        },
        start,
        end,
    )


def _validate_partition_within_source(
    partition: dict[str, Any], source_start: datetime, source_end: datetime
) -> None:
    if partition["mode"] != "time_range":
        return
    for name in ("train", "test"):
        _start_value, start = validate_timestamp(
            partition[name]["start_time"], f"partition.{name}.start_time"
        )
        _end_value, end = validate_timestamp(
            partition[name]["end_time"], f"partition.{name}.end_time"
        )
        if start < source_start or end > source_end:
            raise CatalogueValidationError(
                f"partition.{name} range must be inside the source time window"
            )


def build_draft_dataset(payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = require_object(payload, "request body")
    dataset_id = validate_identifier(
        payload.get("dataset_id") or new_dataset_id(), "dataset_id"
    )
    display_name = require_string(payload.get("display_name"), "display_name")
    description = require_string(
        payload.get("description", ""), "description", allow_empty=True
    )
    purpose = validate_purpose(payload.get("purpose"))
    workload_context = validate_workload_context(payload.get("workload_context"))
    source, source_start, source_end = validate_source(payload.get("source"))
    partition = validate_partition(payload.get("partition"))
    _validate_partition_within_source(partition, source_start, source_end)
    ground_truth = validate_ground_truth(payload.get("ground_truth"))
    incidents = ground_truth["known_incident_windows"]
    for incident in incidents:
        _value, incident_start = validate_timestamp(
            incident["start_time"], "ground_truth incident start_time"
        )
        _value, incident_end = validate_timestamp(
            incident["end_time"], "ground_truth incident end_time"
        )
        if incident_start < source_start or incident_end > source_end:
            raise CatalogueValidationError(
                "ground_truth incident range must be inside the source time window"
            )
    ordered_incidents = sorted(
        incidents, key=lambda item: (item["start_time"], item["end_time"], item["incident_id"])
    )
    for previous, current in zip(ordered_incidents, ordered_incidents[1:]):
        previous_end = validate_timestamp(previous["end_time"], "incident end_time")[1]
        current_start = validate_timestamp(current["start_time"], "incident start_time")[1]
        if current_start < previous_end:
            raise CatalogueValidationError("ground_truth incident windows must not overlap")
    created_at = utc_now()

    technical_schema = {
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
    version = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "version": 1,
        "status": "draft",
        "created_at": created_at,
        "workload_context": workload_context,
        "source": source,
        "provenance": {
            "configuration_frozen": False,
            "resolved_queries": [],
            "query_results": [],
            "fetch_errors": [],
            "query_warnings": [],
        },
        "technical_schema": technical_schema,
        "artifacts": {},
        "checksums": {},
        "ground_truth": ground_truth,
        "incident_context": incidents,
        "partition": partition,
        "validation": {"status": "not_run", "warnings": [], "errors": []},
    }
    summary = version_summary(version)
    record = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "display_name": display_name,
        "description": description,
        "purpose": purpose,
        "status": "draft",
        "created_at": created_at,
        "created_by": None,
        "metadata_revision": 1,
        "metadata_history": [],
        "tags": [],
        "ground_truth_annotations": [],
        "latest_version": 1,
        "workload_context": workload_context,
        "ground_truth": {
            "labels_available": ground_truth["labels_available"],
            "known_incident_count": len(ground_truth["known_incident_windows"]),
        },
        "technical_schema": technical_schema,
        "versions": [summary],
    }
    return record, version


def version_summary(version: dict[str, Any]) -> dict[str, Any]:
    schema = version["technical_schema"]
    return {
        "version": version["version"],
        "status": version["status"],
        "source_type": version["source"]["type"],
        "created_at": version["created_at"],
        "modality": schema.get("modality"),
        "observation_count": schema.get("observation_count", 0),
        "feature_count": schema.get("feature_count", 0),
        "labels_available": version["ground_truth"].get("labels_available", False),
        "partition_mode": version["partition"]["mode"],
    }
