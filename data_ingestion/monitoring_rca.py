"""Pure policy and validation helpers for monitoring-triggered CausalRCA."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Any


RCA_TRIGGER_CONTRACT = "cloudsentinel.monitoring-rca/v1"
RCA_INPUT_CONTRACT = "cloudsentinel.crca-input/v1"
RCA_FEATURE_METADATA_CONTRACT = "cloudsentinel.crca-feature-metadata/v1"
RCA_PREPROCESSING_CONTRACT = "cloudsentinel.crca-minmax/v1"
RCA_ALGORITHM_ID = "cloudsentinel.causal-rca/legacy-v1"
DEFAULT_RCA_CONFIGURATION = {
    "enabled": False,
    "threshold_percentage": 50.0,
    "analysis_window_seconds": 600,
    "cooldown_seconds": 3600,
}


class MonitoringRcaError(RuntimeError):
    """Safe monitoring RCA validation error."""

    def __init__(self, message: str, reason_code: str = "invalid_rca_input"):
        super().__init__(message)
        self.reason_code = reason_code


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def parse_utc(value: str) -> datetime:
    if not isinstance(value, str):
        raise MonitoringRcaError("RCA timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MonitoringRcaError("RCA timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise MonitoringRcaError("RCA timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_rca_configuration(value: Any) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    unknown = set(raw) - set(DEFAULT_RCA_CONFIGURATION)
    if unknown:
        raise MonitoringRcaError("automatic RCA configuration contains unknown fields")
    result = {**DEFAULT_RCA_CONFIGURATION, **raw}
    if not isinstance(result["enabled"], bool):
        raise MonitoringRcaError("automatic RCA enabled must be true or false")
    try:
        threshold = float(result["threshold_percentage"])
        window = int(result["analysis_window_seconds"])
        cooldown = int(result["cooldown_seconds"])
    except (TypeError, ValueError) as exc:
        raise MonitoringRcaError("automatic RCA timing configuration is invalid") from exc
    if not math.isfinite(threshold) or threshold < 0 or threshold > 100:
        raise MonitoringRcaError("automatic RCA threshold must be between 0 and 100")
    if window < 60 or window > 86400:
        raise MonitoringRcaError("automatic RCA analysis window must be between 60 and 86400 seconds")
    if cooldown < 0 or cooldown > 604800:
        raise MonitoringRcaError("automatic RCA cooldown must be between 0 and 604800 seconds")
    return {
        "enabled": result["enabled"],
        "threshold_percentage": threshold,
        "analysis_window_seconds": window,
        "cooldown_seconds": cooldown,
    }


def assess_recipe_compatibility(recipe: Any) -> dict[str, Any]:
    """Return a safe readiness result and exact column metadata without imports."""
    if not isinstance(recipe, dict):
        return _incompatible("missing_recipe", "The live input recipe is unavailable.")
    if recipe.get("contract") != "cloudsentinel.live-input/v1":
        return _incompatible(
            "unsupported_recipe_contract", "The live input recipe contract is unsupported."
        )
    sampling = recipe.get("sampling_interval_seconds")
    if not isinstance(sampling, int) or isinstance(sampling, bool) or sampling < 1:
        return _incompatible(
            "invalid_sampling", "The live input sampling interval is invalid."
        )
    if recipe.get("alignment") != {
        "grid": "inclusive_utc_range",
        "timestamp_match": "exact_microsecond",
        "off_grid_samples": "drop_with_warning",
    }:
        return _incompatible(
            "invalid_alignment",
            "The live input alignment policy is unsupported for CausalRCA.",
        )
    if recipe.get("missing_data") != {
        "offline_representation": "empty",
        "live_policy": "reject_window",
        "imputation": "none",
    }:
        return _incompatible(
            "invalid_missing_data_policy",
            "CausalRCA requires a recipe that rejects missing live values without imputation.",
        )
    feature_order = recipe.get("feature_order")
    mappings = recipe.get("series_mapping")
    executions = recipe.get("query_executions")
    if not isinstance(feature_order, list) or len(feature_order) < 2:
        return _incompatible(
            "insufficient_features",
            "CausalRCA requires at least two explicitly identified features.",
        )
    if any(not isinstance(item, str) or not item.strip() for item in feature_order):
        return _incompatible(
            "invalid_feature_id",
            "Every CausalRCA feature must have an explicit feature identity.",
        )
    if len(set(feature_order)) != len(feature_order):
        return _incompatible("duplicate_feature_id", "Feature identities are not unique.")
    if not isinstance(mappings, list) or not isinstance(executions, list):
        return _incompatible("missing_series_mapping", "The recipe lacks explicit series mapping metadata.")
    by_feature: dict[str, list[dict[str, Any]]] = {}
    for mapping in mappings:
        if isinstance(mapping, dict) and isinstance(mapping.get("feature_id"), str):
            by_feature.setdefault(mapping["feature_id"], []).append(mapping)
    by_execution: dict[str, dict[str, Any]] = {}
    for execution in executions:
        if not isinstance(execution, dict):
            return _incompatible(
                "invalid_query_execution",
                "The recipe contains invalid query execution metadata.",
            )
        execution_id = execution.get("execution_id")
        if not isinstance(execution_id, str) or not execution_id.strip() or execution_id in by_execution:
            return _incompatible(
                "invalid_query_execution",
                "Query execution identities must be explicit and unique.",
            )
        metric_name = execution.get("feature_name")
        if not isinstance(metric_name, str) or not metric_name.strip():
            return _incompatible(
                "missing_query_feature_name",
                "Every query execution must declare an explicit feature name.",
            )
        by_execution[execution_id] = execution
    if len(mappings) != len(feature_order) or set(by_feature) != set(feature_order):
        return _incompatible(
            "invalid_series_mapping",
            "Every feature must have exactly one explicit series mapping.",
        )
    metadata = []
    for index, feature_id in enumerate(feature_order):
        candidates = by_feature.get(feature_id, [])
        if len(candidates) != 1:
            return _incompatible(
                "invalid_series_mapping",
                "Every feature must have exactly one explicit series mapping.",
            )
        mapping = candidates[0]
        execution = by_execution.get(mapping.get("execution_id"))
        labels = mapping.get("metric_labels")
        if not isinstance(execution, dict):
            return _incompatible(
                "unknown_execution_id",
                "A feature mapping refers to an unknown query execution.",
            )
        if not isinstance(labels, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in labels.items()
        ):
            return _incompatible(
                "invalid_series_mapping",
                "A feature mapping contains invalid series metadata.",
            )
        metric_name = execution.get("feature_name")
        entity, conflict = _explicit_entity(execution, labels)
        if conflict:
            return _incompatible("conflicting_entity_metadata", conflict)
        metadata.append(
            {
                "column_index": index,
                "feature_id": feature_id,
                "feature_name": feature_id,
                "entity": entity,
                # Flat fields retain the established result/worker boundary.
                "entity_type": entity.get("type") if entity else None,
                "entity_id": entity.get("id") if entity else None,
                "metric_name": metric_name,
            }
        )
    return {
        "status": "ready",
        "reason_code": None,
        "reason": None,
        "input_contract": RCA_INPUT_CONTRACT,
        "feature_metadata_contract": RCA_FEATURE_METADATA_CONTRACT,
        "feature_metadata": metadata,
        "feature_metadata_sha256": canonical_sha256(metadata),
    }


def _explicit_entity(
    execution: dict[str, Any], labels: dict[str, str]
) -> tuple[dict[str, str] | None, str | None]:
    """Resolve only explicit, verifiable pod identity; never parse names or PromQL."""
    series_pod = labels.get("pod")
    if isinstance(series_pod, str):
        series_pod = series_pod.strip() or None
    else:
        series_pod = None

    target_pod = None
    resolved_target = execution.get("resolved_target")
    if (
        execution.get("execution_mode") == "per_target"
        and execution.get("target_type") == "pod"
        and isinstance(resolved_target, dict)
        and isinstance(resolved_target.get("pod"), str)
    ):
        target_pod = resolved_target["pod"].strip() or None
    if series_pod and target_pod and series_pod != target_pod:
        return (
            None,
            "The series pod label conflicts with the explicitly resolved pod target.",
        )
    pod = series_pod or target_pod
    if pod:
        return {
            "type": "pod",
            "id": pod,
            "source": "prometheus_series_label" if series_pod else "resolved_query_target",
        }, None
    return None, None


def _incompatible(reason_code: str, reason: str) -> dict[str, Any]:
    return {"status": "incompatible", "reason_code": reason_code, "reason": reason[:500]}


def initial_rca_state(
    configuration: dict[str, Any], readiness: dict[str, Any], capability: dict[str, Any] | None
) -> dict[str, Any]:
    safe_capability = capability if isinstance(capability, dict) else {}
    return {
        **deepcopy(configuration),
        "contract": RCA_TRIGGER_CONTRACT,
        "readiness": {
            "status": readiness.get("status"),
            "reason_code": readiness.get("reason_code"),
            "reason": readiness.get("reason"),
        },
        "input_contract": readiness.get("input_contract"),
        "feature_metadata_contract": readiness.get("feature_metadata_contract"),
        "feature_metadata_sha256": readiness.get("feature_metadata_sha256"),
        "algorithm_id": safe_capability.get("algorithm_id", RCA_ALGORITHM_ID),
        "algorithm_config_sha256": safe_capability.get("config_sha256"),
        "preprocessing_contract": safe_capability.get(
            "preprocessing_contract", RCA_PREPROCESSING_CONTRACT
        ),
        "minimum_observations": int(safe_capability.get("minimum_observations", 2)),
        "minimum_features": int(safe_capability.get("minimum_features", 2)),
        "episode_state": "armed",
        "episode_id": None,
        "active_run_id": None,
        "cooldown_until": None,
        "recovered_at": None,
    }


def decide_trigger(state: dict[str, Any], cycle: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pure, deterministic episode state transition for one committed cycle."""
    updated = deepcopy(state)
    percentage = cycle.get("anomaly_percentage")
    count = cycle.get("prediction_count")
    at = cycle.get("completed_at") or cycle.get("committed_at")
    if not updated.get("enabled"):
        return updated, _event("disabled", updated)
    if updated.get("readiness", {}).get("status") != "ready":
        return updated, _event("incompatible", updated, updated.get("readiness", {}).get("reason"))
    if not isinstance(count, int) or count <= 0 or percentage is None:
        return updated, _event("no_new_data", updated)
    try:
        percentage = float(percentage)
        moment = parse_utc(at)
    except (TypeError, ValueError, MonitoringRcaError):
        return updated, _event("invalid_cycle", updated, "The committed cycle has invalid trigger metadata.")
    threshold = float(updated["threshold_percentage"])
    state_name = updated.get("episode_state", "armed")
    cooldown_until = updated.get("cooldown_until")
    cooldown_end = parse_utc(cooldown_until) if cooldown_until else None
    if percentage <= threshold:
        if state_name == "armed":
            return updated, _event("below_or_equal", updated)
        updated["active_run_id"] = None
        updated["recovered_at"] = at
        updated["episode_id"] = None
        updated["episode_state"] = "armed" if cooldown_end is None or moment >= cooldown_end else "recovered_cooldown"
        return updated, _event("recovered", updated)
    if state_name == "armed":
        return updated, _event("trigger", updated)
    if state_name == "recovered_cooldown":
        updated["episode_state"] = "cooldown_episode"
        return updated, _event("suppressed_cooldown", updated)
    return updated, _event("suppressed_episode", updated)


def mark_triggered(state: dict[str, Any], run_id: str, cycle_time: str) -> dict[str, Any]:
    updated = deepcopy(state)
    moment = parse_utc(cycle_time)
    updated["episode_state"] = "triggered"
    updated["episode_id"] = run_id
    updated["active_run_id"] = run_id
    updated["last_triggered_at"] = cycle_time
    updated["cooldown_until"] = utc_text(
        moment + timedelta(seconds=int(updated.get("cooldown_seconds", 0)))
    )
    return updated


def _event(status: str, state: dict[str, Any], reason: str | None = None) -> dict[str, Any]:
    value = {
        "status": status,
        "threshold_percentage": state.get("threshold_percentage"),
        "episode_state": state.get("episode_state"),
    }
    if reason:
        value["reason"] = str(reason)[:500]
    return value


def select_analysis_input(
    cycle_input: dict[str, Any],
    *,
    end_timestamp: str,
    window_seconds: int,
    feature_metadata: list[dict[str, Any]],
    sampling_interval_seconds: int,
    minimum_observations: int = 2,
    minimum_features: int = 2,
) -> dict[str, Any]:
    timestamps = cycle_input.get("timestamps")
    matrix = cycle_input.get("matrix")
    feature_order = cycle_input.get("feature_order")
    if not isinstance(timestamps, list) or not isinstance(matrix, list) or len(timestamps) != len(matrix):
        raise MonitoringRcaError(
            "The committed cycle input is invalid.", "invalid_cycle_input"
        )
    if not isinstance(feature_order, list) or len(feature_order) < minimum_features:
        raise MonitoringRcaError(
            "The committed cycle has too few features for CausalRCA.",
            "insufficient_runtime_features",
        )
    if [item.get("feature_id") for item in feature_metadata] != feature_order:
        raise MonitoringRcaError(
            "CausalRCA feature metadata does not match the committed column order.",
            "feature_metadata_mismatch",
        )
    end = parse_utc(end_timestamp)
    start = end - timedelta(seconds=int(window_seconds))
    selected_timestamps = []
    selected_matrix = []
    last = None
    for timestamp, row in zip(timestamps, matrix):
        current = parse_utc(timestamp)
        if last is not None and current <= last:
            raise MonitoringRcaError(
                "Committed input timestamps are not strictly ordered.",
                "invalid_timestamp_order",
            )
        last = current
        if current < start or current > end:
            continue
        if not isinstance(row, list) or len(row) != len(feature_order):
            raise MonitoringRcaError(
                "Committed RCA input dimensions are inconsistent.",
                "invalid_matrix_shape",
            )
        converted = []
        for value in row:
            if isinstance(value, bool):
                raise MonitoringRcaError(
                    "Committed RCA input must be numeric and finite.",
                    "non_finite_runtime_input",
                )
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise MonitoringRcaError(
                    "Committed RCA input must be numeric and finite.",
                    "non_finite_runtime_input",
                ) from exc
            if not math.isfinite(number):
                raise MonitoringRcaError(
                    "Committed RCA input must be numeric and finite.",
                    "non_finite_runtime_input",
                )
            converted.append(number)
        selected_timestamps.append(timestamp)
        selected_matrix.append(converted)
    if len(selected_matrix) < minimum_observations:
        raise MonitoringRcaError(
            "The committed cycle does not contain enough RCA analysis context.",
            "insufficient_runtime_context",
        )
    if parse_utc(selected_timestamps[0]) > start + timedelta(seconds=int(sampling_interval_seconds)):
        raise MonitoringRcaError(
            "The committed cycle does not cover the requested RCA analysis window.",
            "incomplete_analysis_window",
        )
    return {
        "schema_version": 1,
        "contract": RCA_INPUT_CONTRACT,
        "source_cycle_input_reference": cycle_input.get("input_reference"),
        "source_cycle_input_sha256": cycle_input.get("input_sha256"),
        "analysis_window": {
            "start": selected_timestamps[0],
            "end": selected_timestamps[-1],
            "requested_seconds": int(window_seconds),
        },
        "feature_order": deepcopy(feature_order),
        "feature_order_sha256": cycle_input.get("feature_order_sha256"),
        "feature_metadata": deepcopy(feature_metadata),
        "feature_metadata_contract": RCA_FEATURE_METADATA_CONTRACT,
        "feature_metadata_sha256": canonical_sha256(feature_metadata),
        "timestamps": selected_timestamps,
        "matrix": selected_matrix,
    }


def stable_run_id(
    session_id: str,
    cycle_id: str,
    input_sha256: str,
    configuration_identity: dict[str, Any],
) -> str:
    digest = canonical_sha256(
        {
            "session_id": session_id,
            "cycle_id": cycle_id,
            "input_sha256": input_sha256,
            "configuration": configuration_identity,
        }
    )
    return f"rca_{digest[:32]}"
