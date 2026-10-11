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
        return _incompatible("The live input recipe is unavailable.")
    if recipe.get("contract") != "cloudsentinel.live-input/v1":
        return _incompatible("The live input recipe contract is unsupported.")
    sampling = recipe.get("sampling_interval_seconds")
    if not isinstance(sampling, int) or isinstance(sampling, bool) or sampling < 1:
        return _incompatible("The live input sampling interval is invalid.")
    if recipe.get("alignment") != {
        "grid": "inclusive_utc_range",
        "timestamp_match": "exact_microsecond",
        "off_grid_samples": "drop_with_warning",
    }:
        return _incompatible("The live input alignment policy is unsupported for CausalRCA.")
    if recipe.get("missing_data") != {
        "offline_representation": "empty",
        "live_policy": "reject_window",
        "imputation": "none",
    }:
        return _incompatible("CausalRCA requires a recipe that rejects missing live values without imputation.")
    feature_order = recipe.get("feature_order")
    mappings = recipe.get("series_mapping")
    executions = recipe.get("query_executions")
    if not isinstance(feature_order, list) or len(feature_order) < 2:
        return _incompatible("CausalRCA requires at least two explicitly identified features.")
    if len(set(feature_order)) != len(feature_order):
        return _incompatible("Feature identities are not unique.")
    if not isinstance(mappings, list) or not isinstance(executions, list):
        return _incompatible("The recipe lacks explicit series mapping metadata.")
    by_feature: dict[str, list[dict[str, Any]]] = {}
    for mapping in mappings:
        if isinstance(mapping, dict) and isinstance(mapping.get("feature_id"), str):
            by_feature.setdefault(mapping["feature_id"], []).append(mapping)
    by_execution = {
        item.get("execution_id"): item
        for item in executions
        if isinstance(item, dict) and isinstance(item.get("execution_id"), str)
    }
    metadata = []
    seen_pairs = set()
    for index, feature_id in enumerate(feature_order):
        candidates = by_feature.get(feature_id, [])
        if len(candidates) != 1:
            return _incompatible("Every feature must have exactly one explicit series mapping.")
        mapping = candidates[0]
        execution = by_execution.get(mapping.get("execution_id"))
        labels = mapping.get("metric_labels")
        if not isinstance(execution, dict) or not isinstance(labels, dict):
            return _incompatible("A feature mapping refers to incomplete query metadata.")
        identity_labels = execution.get("identity_labels")
        metric_name = execution.get("feature_name")
        entity_id = labels.get("pod")
        if (
            not isinstance(identity_labels, list)
            or "pod" not in identity_labels
            or not isinstance(entity_id, str)
            or not entity_id.strip()
            or not isinstance(metric_name, str)
            or not metric_name.strip()
        ):
            return _incompatible(
                "CausalRCA currently requires an explicit pod identity and metric name for every feature."
            )
        pair = (entity_id, metric_name)
        if pair in seen_pairs:
            return _incompatible("CausalRCA feature entity and metric identities are not unique.")
        seen_pairs.add(pair)
        metadata.append(
            {
                "column_index": index,
                "feature_id": feature_id,
                "feature_name": feature_id,
                "entity_type": "pod",
                "entity_id": entity_id,
                "metric_name": metric_name,
            }
        )
    return {
        "status": "ready",
        "reason": None,
        "input_contract": RCA_INPUT_CONTRACT,
        "feature_metadata": metadata,
        "feature_metadata_sha256": canonical_sha256(metadata),
    }


def _incompatible(reason: str) -> dict[str, Any]:
    return {"status": "incompatible", "reason": reason[:500]}


def initial_rca_state(
    configuration: dict[str, Any], readiness: dict[str, Any], capability: dict[str, Any] | None
) -> dict[str, Any]:
    safe_capability = capability if isinstance(capability, dict) else {}
    return {
        **deepcopy(configuration),
        "contract": RCA_TRIGGER_CONTRACT,
        "readiness": {"status": readiness.get("status"), "reason": readiness.get("reason")},
        "input_contract": readiness.get("input_contract"),
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
        raise MonitoringRcaError("The committed cycle input is invalid.")
    if not isinstance(feature_order, list) or len(feature_order) < minimum_features:
        raise MonitoringRcaError("The committed cycle has too few features for CausalRCA.")
    if [item.get("feature_id") for item in feature_metadata] != feature_order:
        raise MonitoringRcaError("CausalRCA feature metadata does not match the committed column order.")
    end = parse_utc(end_timestamp)
    start = end - timedelta(seconds=int(window_seconds))
    selected_timestamps = []
    selected_matrix = []
    last = None
    for timestamp, row in zip(timestamps, matrix):
        current = parse_utc(timestamp)
        if last is not None and current <= last:
            raise MonitoringRcaError("Committed input timestamps are not strictly ordered.")
        last = current
        if current < start or current > end:
            continue
        if not isinstance(row, list) or len(row) != len(feature_order):
            raise MonitoringRcaError("Committed RCA input dimensions are inconsistent.")
        converted = []
        for value in row:
            if isinstance(value, bool):
                raise MonitoringRcaError("Committed RCA input must be numeric and finite.")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise MonitoringRcaError("Committed RCA input must be numeric and finite.") from exc
            if not math.isfinite(number):
                raise MonitoringRcaError("Committed RCA input must be numeric and finite.")
            converted.append(number)
        selected_timestamps.append(timestamp)
        selected_matrix.append(converted)
    if len(selected_matrix) < minimum_observations:
        raise MonitoringRcaError("The committed cycle does not contain enough RCA analysis context.")
    if parse_utc(selected_timestamps[0]) > start + timedelta(seconds=int(sampling_interval_seconds)):
        raise MonitoringRcaError("The committed cycle does not cover the requested RCA analysis window.")
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
