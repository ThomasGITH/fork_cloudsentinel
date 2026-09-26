"""Validation helpers for catalogue records, versions, and partitions."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Any


SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
VERSION_STATUSES = {"draft", "fetching", "validating", "available", "failed"}
WORKLOAD_INTENSITIES = {"Low", "Normal", "High", "Variable"}
WORKLOAD_CHARACTERISTICS = {
    "CPU-intensive",
    "Memory-intensive",
    "High traffic",
    "Latency-intensive",
    "Mixed",
}
ANOMALY_SCENARIOS = {
    "Normal operation",
    "CPU stress",
    "Memory stress",
    "Network delay",
    "Unknown",
}


class CatalogueValidationError(ValueError):
    """Raised when catalogue input violates its public contract."""


def validate_identifier(value: Any, field: str = "dataset_id") -> str:
    if not isinstance(value, str) or not value:
        raise CatalogueValidationError(f"{field} must be a non-empty string")
    if value in {".", ".."} or not SAFE_IDENTIFIER.fullmatch(value):
        raise CatalogueValidationError(
            f"{field} may contain only letters, digits, '-', '_' and '.' and may not contain paths"
        )
    return value


def validate_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CatalogueValidationError("version must be a positive integer")
    return value


def require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogueValidationError(f"{field} must be an object")
    return value


def require_string(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise CatalogueValidationError(f"{field} must be {qualifier}")
    return value.strip() if not allow_empty else value


def validate_timestamp(value: Any, field: str) -> tuple[str, datetime]:
    value = require_string(value, field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CatalogueValidationError(f"{field} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CatalogueValidationError(f"{field} must include a timezone")
    parsed = parsed.astimezone(timezone.utc)
    normalized = parsed.isoformat().replace("+00:00", "Z")
    return normalized, parsed


def validate_workload_context(value: Any, *, allow_unknown_legacy: bool = False) -> dict[str, Any]:
    context = require_object(value, "workload_context")
    allowed = {
        "workload_intensity": WORKLOAD_INTENSITIES,
        "dominant_workload_characteristic": WORKLOAD_CHARACTERISTICS,
        "anomaly_scenario": ANOMALY_SCENARIOS,
    }
    result = {}
    for field, choices in allowed.items():
        submitted = context.get(field)
        if allow_unknown_legacy and submitted is None:
            result[field] = None
            continue
        if submitted not in choices:
            raise CatalogueValidationError(
                f"workload_context.{field} must be one of: {', '.join(sorted(choices))}"
            )
        result[field] = submitted
    return result


def validate_purpose(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise CatalogueValidationError("purpose must be a non-empty string or list of strings")
    result = []
    for item in value:
        item = item.strip()
        if item not in result:
            result.append(item)
    return result


def validate_artifact_reference(value: Any, field: str) -> str:
    value = require_string(value, field)
    if "\\" in value:
        raise CatalogueValidationError(f"{field} must use a safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CatalogueValidationError(f"{field} must use a safe relative path")
    return str(path)


def _validate_partition_range(value: Any, field: str) -> tuple[dict[str, str], datetime, datetime]:
    value = require_object(value, field)
    start_value, start = validate_timestamp(value.get("start_time"), f"{field}.start_time")
    end_value, end = validate_timestamp(value.get("end_time"), f"{field}.end_time")
    if start >= end:
        raise CatalogueValidationError(f"{field}.start_time must be before {field}.end_time")
    return {"start_time": start_value, "end_time": end_value}, start, end


def _validate_predefined_part(value: Any, field: str) -> dict[str, Any]:
    value = require_object(value, field)
    row_count = value.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        raise CatalogueValidationError(f"{field}.row_count must be a non-negative integer")
    return {
        "artifact": validate_artifact_reference(value.get("artifact"), f"{field}.artifact"),
        "row_count": row_count,
    }


def _partition_checksum(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_partition(value: Any | None) -> dict[str, Any]:
    if value is None:
        value = {"mode": "none"}
    value = require_object(value, "partition")
    mode = value.get("mode")
    if mode == "none":
        result = {"schema_version": 1, "mode": "none"}
    elif mode == "predefined":
        result = {
            "schema_version": 1,
            "mode": "predefined",
            "train": _validate_predefined_part(value.get("train"), "partition.train"),
            "test": _validate_predefined_part(value.get("test"), "partition.test"),
        }
        labels = value.get("labels")
        if labels is not None:
            labels_part = _validate_predefined_part(labels, "partition.labels")
            semantics = require_object(labels.get("semantics"), "partition.labels.semantics")
            if not all(isinstance(key, str) and isinstance(item, str) for key, item in semantics.items()):
                raise CatalogueValidationError("partition.labels.semantics must map strings to strings")
            labels_part["semantics"] = dict(semantics)
            result["labels"] = labels_part
    elif mode == "time_range":
        train_value, _train_start, train_end = _validate_partition_range(
            value.get("train"), "partition.train"
        )
        test_value, test_start, _test_end = _validate_partition_range(
            value.get("test"), "partition.test"
        )
        gap = value.get("gap_seconds", 0)
        if isinstance(gap, bool) or not isinstance(gap, int) or gap < 0:
            raise CatalogueValidationError("partition.gap_seconds must be a non-negative integer")
        if test_start < train_end:
            raise CatalogueValidationError("partition train and test ranges must not overlap")
        if (test_start - train_end).total_seconds() < gap:
            raise CatalogueValidationError(
                "partition test range must start after the configured gap"
            )
        result = {
            "schema_version": 1,
            "mode": "time_range",
            "train": train_value,
            "test": test_value,
            "gap_seconds": gap,
        }
        if value.get("labels_source") is not None:
            result["labels_source"] = require_string(
                value["labels_source"], "partition.labels_source"
            )
    else:
        raise CatalogueValidationError(
            "partition.mode must be one of: none, predefined, time_range"
        )
    result["checksum"] = _partition_checksum(result)
    return result


def validate_targets(value: Any | None) -> dict[str, list[str]]:
    value = {} if value is None else require_object(value, "source.requested_targets")
    result = {}
    for field in ("services", "pods", "label_selectors"):
        items = value.get(field, [])
        if not isinstance(items, list) or any(
            not isinstance(item, str) or not item.strip() for item in items
        ):
            raise CatalogueValidationError(
                f"source.requested_targets.{field} must be a list of non-empty strings"
            )
        result[field] = [item.strip() for item in items]
    return result


def validate_queries(value: Any | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise CatalogueValidationError("source.queries must be a list")
    result = []
    seen = set()
    for index, query in enumerate(value):
        query = require_object(query, f"source.queries[{index}]")
        query_id = validate_identifier(query.get("query_id"), f"source.queries[{index}].query_id")
        if query_id in seen:
            raise CatalogueValidationError(f"duplicate query_id: {query_id!r}")
        seen.add(query_id)
        template = query.get("promql_template", query.get("promql"))
        normalized = {
            "query_id": query_id,
            "display_name": require_string(
                query.get("display_name"), f"source.queries[{index}].display_name"
            ),
            "promql_template": require_string(
                template, f"source.queries[{index}].promql_template"
            ),
            "status": "not_executed",
        }
        # Retain the DC-2 field as an alias for existing catalogue clients.
        normalized["promql"] = normalized["promql_template"]
        for optional in ("feature_name", "expected_modality", "expected_result_type"):
            if query.get(optional) is not None:
                normalized[optional] = require_string(
                    query[optional], f"source.queries[{index}].{optional}"
                )
        normalized["required"] = query.get("required", True)
        if not isinstance(normalized["required"], bool):
            raise CatalogueValidationError(f"source.queries[{index}].required must be boolean")
        parameters = query.get("parameters", {})
        if not isinstance(parameters, dict):
            raise CatalogueValidationError(f"source.queries[{index}].parameters must be an object")
        normalized["parameters"] = parameters
        execution_mode = query.get("execution_mode", "single")
        if execution_mode not in {"single", "per_target"}:
            raise CatalogueValidationError(
                f"source.queries[{index}].execution_mode must be single or per_target"
            )
        normalized["execution_mode"] = execution_mode
        target_type = query.get("target_type")
        if target_type is not None:
            if target_type not in {"pod", "service"}:
                raise CatalogueValidationError(
                    f"source.queries[{index}].target_type must be pod or service"
                )
            normalized["target_type"] = target_type
        if execution_mode == "per_target" and target_type is None:
            raise CatalogueValidationError(
                f"source.queries[{index}].target_type is required for per_target mode"
            )
        identity_labels = query.get("identity_labels", [])
        if not isinstance(identity_labels, list) or any(
            not isinstance(label, str) or not label.strip() for label in identity_labels
        ):
            raise CatalogueValidationError(
                f"source.queries[{index}].identity_labels must be a list of non-empty strings"
            )
        normalized["identity_labels"] = sorted(set(label.strip() for label in identity_labels))
        target_context = query.get("target_context", {})
        if not isinstance(target_context, dict):
            raise CatalogueValidationError(
                f"source.queries[{index}].target_context must be an object"
            )
        unknown_context = set(target_context) - {"pod", "service"}
        if unknown_context or any(
            not isinstance(item, str) or not item.strip() for item in target_context.values()
        ):
            raise CatalogueValidationError(
                f"source.queries[{index}].target_context may contain pod and service strings"
            )
        normalized["target_context"] = {
            key: target_context[key].strip() for key in sorted(target_context)
        }
        result.append(normalized)
    return result


def validate_incidents(value: Any | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise CatalogueValidationError("ground_truth.known_incident_windows must be a list")
    result = []
    for index, incident in enumerate(value):
        incident = require_object(incident, f"ground_truth.known_incident_windows[{index}]")
        start_value, start = validate_timestamp(
            incident.get("start_time"), f"ground_truth.known_incident_windows[{index}].start_time"
        )
        end_value, end = validate_timestamp(
            incident.get("end_time"), f"ground_truth.known_incident_windows[{index}].end_time"
        )
        if start >= end:
            raise CatalogueValidationError(
                f"ground_truth.known_incident_windows[{index}] start_time must be before end_time"
            )
        normalized = {"start_time": start_value, "end_time": end_value}
        for field in ("scenario", "annotation"):
            if incident.get(field) is not None:
                normalized[field] = require_string(
                    incident[field], f"ground_truth.known_incident_windows[{index}].{field}"
                )
        for field in ("affected_services", "affected_metrics"):
            items = incident.get(field, [])
            if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
                raise CatalogueValidationError(
                    f"ground_truth.known_incident_windows[{index}].{field} must be a list of strings"
                )
            normalized[field] = items
        result.append(normalized)
    return result


def validate_ground_truth(value: Any | None) -> dict[str, Any]:
    value = {} if value is None else require_object(value, "ground_truth")
    labels_available = value.get("labels_available", False)
    if not isinstance(labels_available, bool):
        raise CatalogueValidationError("ground_truth.labels_available must be boolean")
    semantics = value.get("label_semantics", {})
    if not isinstance(semantics, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in semantics.items()
    ):
        raise CatalogueValidationError("ground_truth.label_semantics must map strings to strings")
    return {
        "labels_available": labels_available,
        "label_semantics": dict(semantics),
        "known_incident_windows": validate_incidents(value.get("known_incident_windows")),
    }
