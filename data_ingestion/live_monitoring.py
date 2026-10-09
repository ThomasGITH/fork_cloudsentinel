"""Recipe-driven collection for the existing live-monitoring flow."""

from __future__ import annotations

from datetime import datetime, timezone
from copy import deepcopy
import json
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
    from data_ingestion.catalogue.live_input import validate_live_input_recipe
except ModuleNotFoundError:
    from catalogue.fetching import (
        CatalogueFetchError,
        FetchLimits,
        PrometheusRangeClient,
        assemble_time_series,
        resolve_prometheus_source,
    )
    from catalogue.live_input import validate_live_input_recipe


class LiveMonitoringError(RuntimeError):
    """Safe failure in model resolution, collection, or inference."""


def validate_resolved_live_model(value: Any, requested_model_id: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("model_id") != requested_model_id:
        raise LiveMonitoringError("Learning returned mismatched model provenance")
    artifact_hash = value.get("artifact_manifest_sha256")
    identity = value.get("feature_identity")
    if (
        not isinstance(artifact_hash, str)
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
    client = PrometheusRangeClient(base_url, limits, request_get=request_get)
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
        "error",
    )
    return {key: session.get(key) for key in allowed if key in session}


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
