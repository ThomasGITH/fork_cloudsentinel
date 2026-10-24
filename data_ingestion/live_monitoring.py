"""Recipe-driven collection for the existing live-monitoring flow."""

from __future__ import annotations

from datetime import datetime, timezone
from copy import deepcopy
import json
import math
from typing import Any, Callable

import requests

try:
    from data_ingestion.catalogue.fetching import (
        CatalogueFetchError,
        FetchLimits,
        PrometheusRangeClient,
        assemble_time_series,
        resolve_prometheus_source,
    )
    from data_ingestion.catalogue.environment import prometheus_request_options
    from data_ingestion.catalogue.live_input import validate_live_input_recipe
except ModuleNotFoundError:
    from catalogue.fetching import (
        CatalogueFetchError,
        FetchLimits,
        PrometheusRangeClient,
        assemble_time_series,
        resolve_prometheus_source,
    )
    from catalogue.environment import prometheus_request_options
    from catalogue.live_input import validate_live_input_recipe


class LiveMonitoringError(RuntimeError):
    """Safe failure in model resolution, collection, or inference."""


def validate_resolved_live_model(value: Any, requested_model_id: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("model_id") != requested_model_id:
        raise LiveMonitoringError("Learning returned mismatched model provenance")
    artifact_hash = value.get("artifact_manifest_sha256")
    artifact_id = value.get("artifact_id")
    identity = value.get("feature_identity")
    if (
        not isinstance(artifact_id, str)
        or not artifact_id
        or len(artifact_id) > 128
        or value.get("inference_contract") != "cloudsentinel.inference/v1"
        or not isinstance(value.get("artifact_format"), str)
        or not value.get("artifact_format")
        or not isinstance(artifact_hash, str)
        or len(artifact_hash) != 64
        or any(character not in "0123456789abcdef" for character in artifact_hash)
        or not isinstance(identity, dict)
    ):
        raise LiveMonitoringError("model artifact provenance is incomplete")
    try:
        recipe = validate_live_input_recipe(
            value.get("recipe"),
            expected_feature_order=identity.get("feature_order"),
            expected_feature_order_sha256=identity.get("feature_order_sha256"),
        )
    except Exception as exc:
        raise LiveMonitoringError(f"model live input recipe is invalid: {exc}") from exc
    return {**value, "recipe": recipe}


def _utc(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def collect_recipe_window(
    recipe_value: dict[str, Any],
    start_time: float,
    end_time: float,
    config: dict[str, Any],
    *,
    request_get: Callable[..., Any] = requests.get,
) -> dict[str, Any]:
    """Collect a live range through the canonical offline assembler."""

    recipe = validate_live_input_recipe(recipe_value)
    if end_time <= start_time:
        raise LiveMonitoringError("live monitoring time window is invalid")
    version = {
        "source": {
            "start_time": _utc(start_time),
            "end_time": _utc(end_time),
            "sampling_interval_seconds": recipe["sampling_interval_seconds"],
        }
    }
    limits = FetchLimits.from_config(config)
    base_url = resolve_prometheus_source(recipe["prometheus_source_id"], config)
    client = PrometheusRangeClient(
        base_url,
        limits,
        request_get=request_get,
        request_options=prometheus_request_options(config),
    )
    results = []
    try:
        for execution in recipe["query_executions"]:
            response = client.query_range(
                execution["resolved_query"],
                version["source"]["start_time"],
                version["source"]["end_time"],
                recipe["sampling_interval_seconds"],
            )
            results.append({"execution": execution, "response": response})
        assembled = assemble_time_series(version, results, limits)
    except CatalogueFetchError as exc:
        raise LiveMonitoringError(str(exc)) from exc
    expected_mapping = [
        {
            "feature_id": item["feature_id"],
            "execution_id": item["execution_id"],
            "metric_labels": item["metric_labels"],
        }
        for item in recipe["series_mapping"]
    ]
    if assembled["features"] != recipe["feature_order"]:
        raise LiveMonitoringError("live feature order does not match the model recipe")
    if assembled["series_mappings"] != expected_mapping:
        raise LiveMonitoringError("live Prometheus series identity does not match the model recipe")
    missing = sum(assembled["missing_by_feature"].values())
    if missing:
        raise LiveMonitoringError(
            f"live window contains {missing} missing values; the pinned policy rejects it"
        )
    return {
        "timestamps": assembled["timestamps"],
        "feature_order": assembled["features"],
        "feature_order_sha256": recipe["feature_order_sha256"],
        "rows": assembled["rows"],
        "warnings": assembled["warnings"],
        "missing_values": missing,
    }


def safe_session_projection(session: dict[str, Any]) -> dict[str, Any]:
    """Remove recipes, matrices, URLs and task internals from API responses."""

    allowed = (
        "session_id",
        "model_id",
        "status",
        "created_at",
        "started_at",
        "updated_at",
        "stopped_at",
        "artifact_manifest_sha256",
        "recipe_sha256",
        "feature_order_sha256",
        "sampling_interval_seconds",
        "window_seconds",
        "poll_interval_seconds",
        "iteration",
        "buffered_observations",
        "missing_values",
        "warnings",
        "warmup_observations",
        "latest_result",
        "latest_cycle",
        "cycle_count",
        "last_successful_update",
        "display_name",
        "detector_id",
        "detector_version",
        "artifact_id",
        "inference_contract",
        "artifact_format",
        "error",
    )
    return {key: session.get(key) for key in allowed if key in session}


def merge_execution_projection(
    stored: dict[str, Any], live: dict[str, Any] | None
) -> dict[str, Any]:
    """Overlay temporary state while preserving durable terminal decisions."""

    merged = dict(stored)
    if stored.get("status") in {"stopped", "failed", "dispatch_failed"}:
        merged["execution_state"] = "unavailable"
        return merged
    if live:
        merged.update(safe_session_projection(live))
        merged["execution_state"] = "available"
        return merged
    if stored.get("status") in {"queued", "collecting", "warmup", "active"}:
        merged["status"] = "interrupted"
        merged["execution_state"] = "unavailable"
    return merged


def merge_bounded_buffer(
    existing_items: list[dict[str, Any]],
    timestamps: list[str],
    rows: list[list[float]],
    maximum: int,
) -> list[dict[str, Any]]:
    if maximum < 1:
        raise LiveMonitoringError("live monitoring buffer limit is invalid")
    values = {
        item["timestamp"]: item["values"]
        for item in existing_items
        if isinstance(item, dict)
        and isinstance(item.get("timestamp"), str)
        and isinstance(item.get("values"), list)
    }
    values.update(dict(zip(timestamps, rows)))
    return [
        {"timestamp": timestamp, "values": row}
        for timestamp, row in sorted(values.items())[-maximum:]
    ]


def merge_buffer_with_watermark(
    existing_items: list[dict[str, Any]],
    timestamps: list[str],
    rows: list[list[float]],
    maximum: int,
    prediction_watermark: str | None,
) -> tuple[list[dict[str, Any]], int]:
    """Merge new samples without mutating input at/before a committed watermark.

    Prometheus may revise or return late samples. Once a prediction timestamp is
    committed, its historical input remains immutable. New samples at or before
    the watermark are ignored; overlapping uncommitted timestamps may still be
    refreshed.
    """

    if len(timestamps) != len(rows):
        raise LiveMonitoringError("live timestamps and rows are misaligned")
    current = {
        item["timestamp"]: item["values"]
        for item in existing_items
        if isinstance(item, dict)
        and isinstance(item.get("timestamp"), str)
        and isinstance(item.get("values"), list)
    }
    late = 0
    for timestamp, row in zip(timestamps, rows):
        if prediction_watermark is not None and timestamp <= prediction_watermark:
            if timestamp not in current or current[timestamp] != row:
                late += 1
            continue
        current[timestamp] = row
    merged = [
        {"timestamp": timestamp, "values": row}
        for timestamp, row in sorted(current.items())[-maximum:]
    ]
    return merged, late


def align_new_predictions(
    input_timestamps: list[str],
    runtime_payload: dict[str, Any],
    prediction_watermark: str | None,
) -> dict[str, Any]:
    """Validate generic output and select timestamps not previously committed."""

    if not isinstance(input_timestamps, list) or any(
        not isinstance(value, str) or not value for value in input_timestamps
    ):
        raise LiveMonitoringError("inference timestamps are invalid")
    if input_timestamps != sorted(input_timestamps) or len(input_timestamps) != len(
        set(input_timestamps)
    ):
        raise LiveMonitoringError("inference timestamps must be unique and ordered")
    predictions = runtime_payload.get("binary_predictions")
    scores = runtime_payload.get("anomaly_scores")
    warmup = runtime_payload.get("warmup_observations")
    input_count = runtime_payload.get("input_observation_count")
    if (
        not isinstance(predictions, list)
        or not isinstance(scores, list)
        or len(predictions) != len(scores)
        or not isinstance(warmup, int)
        or isinstance(warmup, bool)
        or warmup < 0
        or input_count != len(input_timestamps)
        or warmup + len(predictions) != len(input_timestamps)
    ):
        raise LiveMonitoringError("generic inference output cannot be aligned")
    if any(type(value) is not int or value not in {0, 1} for value in predictions):
        raise LiveMonitoringError("generic inference predictions are invalid")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in scores
    ):
        raise LiveMonitoringError("generic inference scores are invalid")
    prediction_timestamps = input_timestamps[warmup:]
    eligible_indexes = [
        index
        for index, timestamp in enumerate(prediction_timestamps)
        if prediction_watermark is None or timestamp > prediction_watermark
    ]
    eligible_timestamps = [prediction_timestamps[index] for index in eligible_indexes]
    eligible_predictions = [predictions[index] for index in eligible_indexes]
    eligible_scores = [float(scores[index]) for index in eligible_indexes]
    anomaly_count = sum(eligible_predictions)
    count = len(eligible_predictions)
    return {
        "warmup_observations": warmup,
        "all_prediction_timestamps": prediction_timestamps,
        "all_binary_predictions": predictions,
        "all_anomaly_scores": [float(value) for value in scores],
        "prediction_timestamps": eligible_timestamps,
        "binary_predictions": eligible_predictions,
        "anomaly_scores": eligible_scores,
        "prediction_count": count,
        "anomaly_count": anomaly_count,
        "anomaly_percentage": (anomaly_count / count * 100.0) if count else None,
        "next_watermark": eligible_timestamps[-1] if eligible_timestamps else prediction_watermark,
    }


def stopped_session(
    value: dict[str, Any],
    revoke: Callable[[str], None],
    *,
    stopped_at: str,
) -> dict[str, Any]:
    session = deepcopy(value)
    for field in ("current_task_id", "next_task_id"):
        task_id = session.get(field)
        if isinstance(task_id, str) and task_id:
            revoke(task_id)
    session["status"] = "stopped"
    session["stopped_at"] = stopped_at
    session.pop("buffer", None)
    return session


def encode_session(session: dict[str, Any]) -> str:
    return json.dumps(session, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def decode_session(value: bytes | str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        decoded = value.decode("utf-8") if isinstance(value, bytes) else value
        result = json.loads(decoded)
    except (UnicodeError, json.JSONDecodeError):
        return None
    return result if isinstance(result, dict) else None
