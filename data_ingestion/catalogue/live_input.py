"""Immutable Prometheus input recipes shared by offline and live collection."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any


LIVE_INPUT_CONTRACT = "cloudsentinel.live-input/v1"


class LiveInputRecipeError(ValueError):
    """Raised when a live-input recipe is incomplete or has changed."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_order_sha256(feature_order: list[str]) -> str:
    encoded = json.dumps(
        feature_order, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_live_input_recipe(
    version: dict[str, Any],
    executions: list[dict[str, Any]],
    assembled: dict[str, Any],
) -> dict[str, Any]:
    """Freeze the exact successful Prometheus-to-feature construction."""

    features = list(assembled["features"])
    mappings = deepcopy(assembled.get("series_mappings", []))
    recipe = {
        "schema_version": 1,
        "contract": LIVE_INPUT_CONTRACT,
        "prometheus_source_id": version["source"]["prometheus_source_id"],
        "sampling_interval_seconds": version["source"]["sampling_interval_seconds"],
        "feature_order": features,
        "feature_order_sha256": feature_order_sha256(features),
        "query_executions": [
            {
                key: deepcopy(execution.get(key))
                for key in (
                    "execution_id",
                    "query_id",
                    "display_name",
                    "feature_name",
                    "required",
                    "execution_mode",
                    "target_type",
                    "identity_labels",
                    "resolved_target",
                    "query_template",
                    "resolved_query",
                )
            }
            for execution in executions
        ],
        "series_mapping": mappings,
        "alignment": {
            "grid": "inclusive_utc_range",
            "timestamp_match": "exact_microsecond",
            "off_grid_samples": "drop_with_warning",
        },
        "missing_data": {
            "offline_representation": "empty",
            "live_policy": "reject_window",
            "imputation": "none",
        },
    }
    recipe["recipe_sha256"] = canonical_sha256(recipe)
    return validate_live_input_recipe(recipe)


def validate_live_input_recipe(
    value: Any,
    *,
    expected_feature_order: list[str] | None = None,
    expected_feature_order_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LiveInputRecipeError("live input recipe must be an object")
    recipe = deepcopy(value)
    digest = recipe.pop("recipe_sha256", None)
    if recipe.get("schema_version") != 1 or recipe.get("contract") != LIVE_INPUT_CONTRACT:
        raise LiveInputRecipeError("live input recipe contract is unsupported")
    if not isinstance(digest, str) or digest != canonical_sha256(recipe):
        raise LiveInputRecipeError("live input recipe checksum mismatch")
    source_id = recipe.get("prometheus_source_id")
    step = recipe.get("sampling_interval_seconds")
    if not isinstance(source_id, str) or not source_id or not isinstance(step, int) or step < 1:
        raise LiveInputRecipeError("live input recipe source or sampling interval is invalid")
    order = recipe.get("feature_order")
    if (
        not isinstance(order, list)
        or not order
        or len(order) != len(set(order))
        or any(not isinstance(item, str) or not item for item in order)
        or recipe.get("feature_order_sha256") != feature_order_sha256(order)
    ):
        raise LiveInputRecipeError("live input recipe feature identity is invalid")
    if expected_feature_order is not None and order != expected_feature_order:
        raise LiveInputRecipeError("live input recipe feature order mismatch")
    if expected_feature_order_sha256 is not None and recipe.get(
        "feature_order_sha256"
    ) != expected_feature_order_sha256:
        raise LiveInputRecipeError("live input recipe feature-order checksum mismatch")
    executions = recipe.get("query_executions")
    mappings = recipe.get("series_mapping")
    if not isinstance(executions, list) or not executions or len(executions) > 100:
        raise LiveInputRecipeError("live input recipe query executions are invalid")
    execution_ids = set()
    for execution in executions:
        if not isinstance(execution, dict):
            raise LiveInputRecipeError("live input recipe query execution is invalid")
        execution_id = execution.get("execution_id")
        query = execution.get("resolved_query")
        if (
            not isinstance(execution_id, str)
            or not execution_id
            or execution_id in execution_ids
            or not isinstance(query, str)
            or not query.strip()
            or len(query) > 20_000
        ):
            raise LiveInputRecipeError("live input recipe query execution is invalid")
        execution_ids.add(execution_id)
    if not isinstance(mappings, list) or len(mappings) != len(order):
        raise LiveInputRecipeError("live input recipe series mapping is incomplete")
    mapped_features = []
    for mapping in mappings:
        if not isinstance(mapping, dict) or mapping.get("execution_id") not in execution_ids:
            raise LiveInputRecipeError("live input recipe series mapping is invalid")
        feature_id = mapping.get("feature_id")
        labels = mapping.get("metric_labels")
        if not isinstance(feature_id, str) or not isinstance(labels, dict):
            raise LiveInputRecipeError("live input recipe series mapping is invalid")
        if any(not isinstance(key, str) or not isinstance(item, str) for key, item in labels.items()):
            raise LiveInputRecipeError("live input recipe series labels are invalid")
        mapped_features.append(feature_id)
    if mapped_features != order:
        raise LiveInputRecipeError("live input recipe series order mismatch")
    if recipe.get("alignment") != {
        "grid": "inclusive_utc_range",
        "timestamp_match": "exact_microsecond",
        "off_grid_samples": "drop_with_warning",
    }:
        raise LiveInputRecipeError("live input recipe alignment policy is unsupported")
    if recipe.get("missing_data") != {
        "offline_representation": "empty",
        "live_policy": "reject_window",
        "imputation": "none",
    }:
        raise LiveInputRecipeError("live input recipe missing-data policy is unsupported")
    recipe["recipe_sha256"] = digest
    return recipe


def public_live_monitoring_projection(value: Any) -> dict[str, Any]:
    try:
        recipe = validate_live_input_recipe(value)
    except LiveInputRecipeError:
        return {
            "status": "not_ready",
            "reason": "No complete validated live input recipe is available",
        }
    return {
        "status": "ready",
        "contract": recipe["contract"],
        "recipe_sha256": recipe["recipe_sha256"],
        "feature_order_sha256": recipe["feature_order_sha256"],
        "sampling_interval_seconds": recipe["sampling_interval_seconds"],
    }
