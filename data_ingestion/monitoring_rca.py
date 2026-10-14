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
RCA_BARO_INPUT_CONTRACT = "cloudsentinel.baro-input/v1"
RCA_FEATURE_METADATA_CONTRACT = "cloudsentinel.crca-feature-metadata/v1"
RCA_PREPROCESSING_CONTRACT = "cloudsentinel.crca-minmax/v1"
RCA_BARO_PREPROCESSING_CONTRACT = "cloudsentinel.baro-robust-scorer-raw/v1"
RCA_ALGORITHM_ID = "cloudsentinel.causal-rca/legacy-v1"
RCA_METHOD_CAUSAL_LEGACY = RCA_ALGORITHM_ID
RCA_METHOD_BARO = "cloudsentinel.baro-robust-scorer/v1"
DEFAULT_RCA_METHOD = RCA_METHOD_CAUSAL_LEGACY
RCA_METHOD_DEFINITIONS = {
    RCA_METHOD_CAUSAL_LEGACY: {
        "id": RCA_METHOD_CAUSAL_LEGACY,
        "display_name": "CausalRCA",
        "input_contract": RCA_INPUT_CONTRACT,
        "preprocessing_contract": RCA_PREPROCESSING_CONTRACT,
        "minimum_requirements": {"observations": 2, "features": 2},
    },
    RCA_METHOD_BARO: {
        "id": RCA_METHOD_BARO,
        "display_name": "BARO robust scorer",
        "input_contract": RCA_BARO_INPUT_CONTRACT,
        "preprocessing_contract": RCA_BARO_PREPROCESSING_CONTRACT,
        "minimum_requirements": {
            "observations": 4,
            "features": 2,
            "baseline_observations": 2,
            "post_change_observations": 2,
        },
    },
}
RCA_REANALYSIS_POLICY = "cloudsentinel.monitoring-rca-reanalysis/v1"
DEFAULT_RCA_CONFIGURATION = {
    "enabled": False,
    "method": DEFAULT_RCA_METHOD,
    "threshold_percentage": 50.0,
    "analysis_window_seconds": 600,
    "cooldown_seconds": 3600,
    "reanalyze_within_episode": False,
    "reanalysis_min_interval_seconds": 900,
    "reanalysis_min_new_observations": 20,
    "max_attempts_per_episode": 3,
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
    if value is not None and not isinstance(value, dict):
        raise MonitoringRcaError("automatic RCA configuration must be an object")
    raw = value if isinstance(value, dict) else {}
    unknown = set(raw) - set(DEFAULT_RCA_CONFIGURATION)
    if unknown:
        raise MonitoringRcaError("automatic RCA configuration contains unknown fields")
    result = {**DEFAULT_RCA_CONFIGURATION, **raw}
    method = result.get("method")
    if not isinstance(method, str) or method not in RCA_METHOD_DEFINITIONS:
        raise MonitoringRcaError(
            "automatic RCA method is unknown", "unknown_rca_method"
        )
    for name in ("enabled", "reanalyze_within_episode"):
        if not isinstance(result[name], bool):
            raise MonitoringRcaError(
                f"automatic RCA {name.replace('_', ' ')} must be true or false"
            )
    if isinstance(result["threshold_percentage"], bool):
        raise MonitoringRcaError("automatic RCA threshold must be numeric")
    integer_fields = (
        "analysis_window_seconds",
        "cooldown_seconds",
        "reanalysis_min_interval_seconds",
        "reanalysis_min_new_observations",
        "max_attempts_per_episode",
    )
    if any(isinstance(result[name], bool) for name in integer_fields):
        raise MonitoringRcaError("automatic RCA timing and budget values must be integers")
    try:
        threshold = float(result["threshold_percentage"])
        window = int(result["analysis_window_seconds"])
        cooldown = int(result["cooldown_seconds"])
        reanalysis_interval = int(result["reanalysis_min_interval_seconds"])
        reanalysis_observations = int(result["reanalysis_min_new_observations"])
        episode_budget = int(result["max_attempts_per_episode"])
    except (TypeError, ValueError) as exc:
        raise MonitoringRcaError("automatic RCA timing configuration is invalid") from exc
    for name in integer_fields:
        raw_value = result[name]
        if not isinstance(raw_value, int) or isinstance(raw_value, bool):
            raise MonitoringRcaError(
                f"automatic RCA {name.replace('_', ' ')} must be an integer"
            )
    if not math.isfinite(threshold) or threshold < 0 or threshold > 100:
        raise MonitoringRcaError("automatic RCA threshold must be between 0 and 100")
    if window < 60 or window > 86400:
        raise MonitoringRcaError("automatic RCA analysis window must be between 60 and 86400 seconds")
    if cooldown < 0 or cooldown > 604800:
        raise MonitoringRcaError("automatic RCA cooldown must be between 0 and 604800 seconds")
    if reanalysis_interval < 60 or reanalysis_interval > 604800:
        raise MonitoringRcaError(
            "automatic RCA reanalysis interval must be between 60 and 604800 seconds"
        )
    if reanalysis_observations < 1 or reanalysis_observations > 100000:
        raise MonitoringRcaError(
            "automatic RCA minimum new observations must be between 1 and 100000"
        )
    if episode_budget < 1 or episode_budget > 20:
        raise MonitoringRcaError(
            "automatic RCA episode budget must be between 1 and 20 attempts"
        )
    return {
        "enabled": result["enabled"],
        "method": method,
        "threshold_percentage": threshold,
        "analysis_window_seconds": window,
        "cooldown_seconds": cooldown,
        "reanalyze_within_episode": result["reanalyze_within_episode"],
        "reanalysis_min_interval_seconds": reanalysis_interval,
        "reanalysis_min_new_observations": reanalysis_observations,
        "max_attempts_per_episode": episode_budget,
    }


def rca_method_id(value: Any) -> str:
    """Resolve legacy records without a method to the established algorithm."""
    if not isinstance(value, dict):
        return DEFAULT_RCA_METHOD
    method = value.get("method")
    if isinstance(method, str) and method:
        return method
    algorithm = value.get("algorithm")
    algorithm_id = algorithm.get("id") if isinstance(algorithm, dict) else None
    legacy_algorithm = value.get("algorithm_id")
    candidate = algorithm_id or legacy_algorithm
    return candidate if isinstance(candidate, str) and candidate else DEFAULT_RCA_METHOD


def safe_method_capabilities(
    recipe_readiness: dict[str, Any] | None,
    runtime_capability: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Project bounded multi-method capability metadata without private config."""
    readiness = recipe_readiness if isinstance(recipe_readiness, dict) else {}
    runtime = runtime_capability if isinstance(runtime_capability, dict) else {}
    runtime_methods = runtime.get("methods")
    if not isinstance(runtime_methods, list):
        runtime_methods = [{
            "id": runtime.get("algorithm_id", DEFAULT_RCA_METHOD),
            "status": "available" if isinstance(runtime.get("config_sha256"), str) else "unavailable",
        }]
    by_id = {
        item.get("id"): item
        for item in runtime_methods
        if isinstance(item, dict) and item.get("id") in RCA_METHOD_DEFINITIONS
    }
    projected = []
    for method_id, definition in RCA_METHOD_DEFINITIONS.items():
        advertised = by_id.get(method_id, {})
        status = advertised.get("status", "unavailable")
        reason_code = advertised.get("reason_code")
        reason = advertised.get("reason")
        if readiness.get("status") != "ready":
            status = "incompatible"
            reason_code = readiness.get("reason_code")
            reason = readiness.get("reason")
        projected.append({
            **deepcopy(definition),
            "status": status,
            "reason_code": reason_code,
            "reason": str(reason)[:500] if reason else None,
        })
    return projected


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
    method = configuration.get("method", DEFAULT_RCA_METHOD)
    capability_method = safe_capability.get("id") or safe_capability.get("algorithm_id")
    if capability_method is not None and capability_method != method:
        raise MonitoringRcaError(
            "RCA runtime capability does not match the selected method",
            "rca_method_capability_mismatch",
        )
    return {
        **deepcopy(configuration),
        "contract": RCA_TRIGGER_CONTRACT,
        "readiness": {
            "status": readiness.get("status"),
            "reason_code": readiness.get("reason_code"),
            "reason": readiness.get("reason"),
        },
        "input_contract": safe_capability.get(
            "input_contract", readiness.get("input_contract")
        ),
        "feature_metadata_contract": readiness.get("feature_metadata_contract"),
        "feature_metadata_sha256": readiness.get("feature_metadata_sha256"),
        "method": method,
        "algorithm_id": method,
        "algorithm_config_sha256": safe_capability.get("config_sha256"),
        "preprocessing_contract": safe_capability.get(
            "preprocessing_contract", RCA_PREPROCESSING_CONTRACT
        ),
        "minimum_observations": int(safe_capability.get("minimum_observations", 2)),
        "minimum_features": int(safe_capability.get("minimum_features", 2)),
        "minimum_baseline_observations": int(
            safe_capability.get("minimum_baseline_observations", 0)
        ),
        "minimum_post_change_observations": int(
            safe_capability.get("minimum_post_change_observations", 0)
        ),
        "policy_version": RCA_REANALYSIS_POLICY,
        "episode_state": "armed",
        "episode_id": None,
        "active_run_id": None,
        "cooldown_until": None,
        "recovered_at": None,
        "attempt_count": 0,
        "last_run_id": None,
        "last_intent_at": None,
        "last_analysis_end": None,
        "last_input_sha256": None,
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
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0 or percentage is None:
        return updated, _event(
            "no_new_data",
            updated,
            "No newly evaluated predictions were available for an RCA decision.",
            "no_new_data",
        )
    try:
        percentage = float(percentage)
        moment = parse_utc(at)
    except (TypeError, ValueError, MonitoringRcaError):
        return updated, _event("invalid_cycle", updated, "The committed cycle has invalid trigger metadata.")
    if not math.isfinite(percentage):
        return updated, _event(
            "invalid_cycle",
            updated,
            "The committed cycle has invalid trigger metadata.",
            "invalid_cycle",
        )
    threshold = float(updated["threshold_percentage"])
    state_name = updated.get("episode_state", "armed")
    cooldown_until = updated.get("cooldown_until")
    cooldown_end = parse_utc(cooldown_until) if cooldown_until else None
    if percentage <= threshold:
        if state_name == "armed":
            return updated, _event(
                "below_or_equal",
                updated,
                "The anomaly percentage did not strictly exceed the RCA threshold.",
                "threshold_not_exceeded",
            )
        updated["active_run_id"] = None
        updated["recovered_at"] = at
        updated["episode_id"] = None
        updated["attempt_count"] = 0
        updated["last_run_id"] = None
        updated["last_intent_at"] = None
        updated["last_analysis_end"] = None
        updated["last_input_sha256"] = None
        updated["episode_state"] = "armed" if cooldown_end is None or moment >= cooldown_end else "recovered_cooldown"
        return updated, _event(
            "recovered",
            updated,
            "The episode recovered at or below the RCA threshold.",
            "episode_recovered",
        )
    if state_name == "armed":
        return updated, _event("trigger", updated)
    if state_name in {"recovered_cooldown", "cooldown_episode"}:
        updated["episode_state"] = "cooldown_episode"
        return updated, _event(
            "suppressed_cooldown",
            updated,
            "The previous episode recovered, but the episode cooldown has not elapsed.",
            "episode_cooldown_active",
        )
    if updated.get("reanalyze_within_episode"):
        return updated, _event("reanalysis_candidate", updated)
    return updated, _event(
        "suppressed_episode",
        updated,
        "RCA was not run again because this episode has already been analysed.",
        "episode_already_analyzed",
        related_run_id=updated.get("last_run_id"),
    )


def mark_triggered(
    state: dict[str, Any],
    run_id: str,
    cycle_time: str,
    *,
    intent_at: str | None = None,
    analysis_end: str | None = None,
    input_sha256: str | None = None,
) -> dict[str, Any]:
    updated = deepcopy(state)
    moment = parse_utc(cycle_time)
    updated["episode_state"] = "triggered"
    updated["episode_id"] = updated.get("episode_id") or run_id
    updated["active_run_id"] = run_id
    updated["last_triggered_at"] = cycle_time
    updated["last_run_id"] = run_id
    updated["last_intent_at"] = intent_at or cycle_time
    updated["last_analysis_end"] = analysis_end
    updated["last_input_sha256"] = input_sha256
    updated["attempt_count"] = int(updated.get("attempt_count", 0)) + 1
    updated["cooldown_until"] = utc_text(
        moment + timedelta(seconds=int(updated.get("cooldown_seconds", 0)))
    )
    return updated


def _event(
    status: str,
    state: dict[str, Any],
    reason: str | None = None,
    reason_code: str | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    value = {
        "status": status,
        "threshold_percentage": state.get("threshold_percentage"),
        "episode_state": state.get("episode_state"),
        "attempt_count": int(state.get("attempt_count", 0)),
        "max_attempts_per_episode": state.get("max_attempts_per_episode"),
    }
    if reason_code:
        value["reason_code"] = reason_code
    if reason:
        value["reason"] = str(reason)[:500]
    for key, item in metadata.items():
        if item is not None:
            value[key] = item
    return value


def evaluate_reanalysis(
    state: dict[str, Any],
    selected_input: dict[str, Any],
    input_sha256: str,
    *,
    decision_time: str,
) -> dict[str, Any]:
    """Evaluate rate, data and budget gates for one same-episode candidate."""
    attempt_count = int(state.get("attempt_count", 0))
    budget = int(state.get("max_attempts_per_episode", 1))
    if attempt_count >= budget:
        return _event(
            "suppressed_budget",
            state,
            "RCA was not run because this episode has reached its execution budget.",
            "episode_budget_reached",
            related_run_id=state.get("last_run_id"),
        )
    last_intent_at = state.get("last_intent_at")
    next_not_before = None
    if last_intent_at:
        next_moment = parse_utc(last_intent_at) + timedelta(
            seconds=int(state.get("reanalysis_min_interval_seconds", 900))
        )
        next_not_before = utc_text(next_moment)
        if parse_utc(decision_time) < next_moment:
            return _event(
                "suppressed_interval",
                state,
                "RCA reanalysis is rate-limited for this episode.",
                "reanalysis_interval_not_elapsed",
                next_reanalysis_not_before=next_not_before,
                related_run_id=state.get("last_run_id"),
            )
    timestamps = selected_input.get("timestamps")
    if not isinstance(timestamps, list):
        raise MonitoringRcaError("The selected RCA input timestamps are invalid.")
    previous_end = state.get("last_analysis_end")
    if previous_end:
        previous_moment = parse_utc(previous_end)
        new_observations = len(
            {item for item in timestamps if parse_utc(item) > previous_moment}
        )
    else:
        new_observations = len(set(timestamps))
    minimum = int(state.get("reanalysis_min_new_observations", 20))
    if new_observations < minimum:
        return _event(
            "suppressed_new_observations",
            state,
            "RCA was not run because too few new observations are available.",
            "insufficient_new_observations",
            new_observation_count=new_observations,
            required_new_observations=minimum,
            next_reanalysis_not_before=next_not_before,
            related_run_id=state.get("last_run_id"),
        )
    if state.get("last_input_sha256") == input_sha256:
        return _event(
            "suppressed_identical_input",
            state,
            "RCA was not run because the selected analysis input is unchanged.",
            "identical_analysis_input",
            new_observation_count=new_observations,
            related_run_id=state.get("last_run_id"),
        )
    return _event(
        "reanalysis",
        state,
        new_observation_count=new_observations,
        required_new_observations=minimum,
        next_reanalysis_not_before=next_not_before,
    )


def analysis_input_identity(selected_input: dict[str, Any]) -> str:
    """Identify analysis content independently of its cycle artifact reference."""
    return canonical_sha256(
        {
            "contract": selected_input.get("contract"),
            "input_contract": selected_input.get("input_contract"),
            "analysis_window": selected_input.get("analysis_window"),
            "feature_order": selected_input.get("feature_order"),
            "feature_order_sha256": selected_input.get("feature_order_sha256"),
            "feature_metadata_sha256": selected_input.get("feature_metadata_sha256"),
            "timestamps": selected_input.get("timestamps"),
            "matrix": selected_input.get("matrix"),
            "change_point": selected_input.get("change_point"),
        }
    )


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


def attach_detected_change_point(
    selected_input: dict[str, Any],
    cycle_result: dict[str, Any],
    *,
    cycle_id: str,
    cycle_result_sha256: str,
    minimum_baseline_observations: int = 2,
    minimum_post_change_observations: int = 2,
) -> dict[str, Any]:
    """Pin the first stored positive prediction inside the selected BARO window."""
    timestamps = selected_input.get("timestamps")
    predictions = cycle_result.get("prediction_timestamps")
    binary = cycle_result.get("binary_predictions")
    if (
        not isinstance(timestamps, list)
        or not timestamps
        or len(set(timestamps)) != len(timestamps)
        or not isinstance(predictions, list)
        or not isinstance(binary, list)
        or len(predictions) != len(binary)
        or any(value not in (0, 1) or isinstance(value, bool) for value in binary)
        or any(not isinstance(value, str) for value in predictions)
    ):
        raise MonitoringRcaError(
            "Stored detector predictions cannot provide a valid BARO change point.",
            "invalid_baro_change_point_source",
        )
    try:
        parsed_predictions = [parse_utc(timestamp) for timestamp in predictions]
    except MonitoringRcaError as exc:
        raise MonitoringRcaError(
            "Stored detector predictions cannot provide a valid BARO change point.",
            "invalid_baro_change_point_source",
        ) from exc
    if len(set(predictions)) != len(predictions) or any(
        current <= previous
        for previous, current in zip(parsed_predictions, parsed_predictions[1:])
    ):
        raise MonitoringRcaError(
            "Stored detector prediction timestamps are not strictly ordered.",
            "invalid_baro_change_point_source",
        )
    selected_indexes = {timestamp: index for index, timestamp in enumerate(timestamps)}
    positive = [
        (parsed, timestamp)
        for parsed, timestamp, prediction in zip(parsed_predictions, predictions, binary)
        if prediction == 1 and timestamp in selected_indexes
    ]
    if not positive:
        raise MonitoringRcaError(
            "No positive detector prediction exists in the selected RCA analysis window.",
            "baro_change_point_unavailable",
        )
    _, detected_timestamp = min(positive, key=lambda item: item[0])
    index = selected_indexes[detected_timestamp]
    baseline_count = index
    anomalous_count = len(timestamps) - index
    if baseline_count < minimum_baseline_observations:
        raise MonitoringRcaError(
            "BARO requires more baseline observations before the detected change point.",
            "insufficient_baro_baseline",
        )
    if anomalous_count < minimum_post_change_observations:
        raise MonitoringRcaError(
            "BARO requires more observations from the detected change point onward.",
            "insufficient_baro_post_change",
        )
    result = deepcopy(selected_input)
    result["contract"] = RCA_BARO_INPUT_CONTRACT
    result["input_contract"] = RCA_BARO_INPUT_CONTRACT
    result["change_point"] = {
        "contract": "cloudsentinel.baro-change-point/v1",
        "source": "stored_monitoring_cycle_result",
        "cycle_id": cycle_id,
        "cycle_result_sha256": cycle_result_sha256,
        "detected_change_point": detected_timestamp,
        "index": index,
        "baseline_observation_count": baseline_count,
        "anomalous_observation_count": anomalous_count,
    }
    return result
