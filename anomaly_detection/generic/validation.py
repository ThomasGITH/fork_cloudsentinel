"""Request and adapter-output validation for the generic runtime."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from detectors.contracts import PredictionContractError, PredictionResult


class DetectionRequestError(ValueError):
    pass


ALLOWED_CONTEXT_FIELDS = {
    "task_id",
    "iteration",
    "start_time",
    "end_time",
    "containers",
    "metrics",
    "data_interval",
    "crca_threshold",
    "crca_pods",
    "live_monitoring_session_id",
    "recipe_sha256",
}
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def feature_order_sha256(order: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(list(order), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_matrix(file_object: Any, *, max_observations: int, max_features: int) -> np.ndarray:
    try:
        matrix = pd.read_csv(file_object, header=None, skip_blank_lines=False).to_numpy()
    except Exception as exc:
        raise DetectionRequestError("matrix must be a readable headerless CSV") from exc
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise DetectionRequestError("matrix must be a non-empty two-dimensional matrix")
    if matrix.shape[0] > max_observations or matrix.shape[1] > max_features:
        raise DetectionRequestError("matrix exceeds configured shape limits")
    if matrix.dtype.kind not in "iuf" or not np.isfinite(matrix).all():
        raise DetectionRequestError("matrix must contain only finite numeric values")
    return matrix


def parse_metadata(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "")
    except (TypeError, json.JSONDecodeError) as exc:
        raise DetectionRequestError("metadata must be valid JSON") from exc
    if not isinstance(value, dict):
        raise DetectionRequestError("metadata must be a JSON object")
    allowed = {"model_id", "feature_order", "feature_order_sha256", "timestamps", "context"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise DetectionRequestError("unsupported metadata field(s): " + ", ".join(unknown))
    if not isinstance(value.get("model_id"), str) or not SAFE_IDENTIFIER.fullmatch(
        value["model_id"]
    ):
        raise DetectionRequestError("metadata.model_id must be a safe identifier")
    order = value.get("feature_order")
    if order is not None and (
        not isinstance(order, list)
        or any(not isinstance(item, str) or not item for item in order)
        or len(set(order)) != len(order)
    ):
        raise DetectionRequestError("metadata.feature_order is invalid")
    digest = value.get("feature_order_sha256")
    if digest is not None and (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise DetectionRequestError("metadata.feature_order_sha256 is invalid")
    if order is None and digest is None:
        raise DetectionRequestError("feature order or feature-order hash is required")
    timestamps = value.get("timestamps", [])
    if not isinstance(timestamps, list) or any(not isinstance(item, str) for item in timestamps):
        raise DetectionRequestError("metadata.timestamps is invalid")
    context = value.get("context", {})
    if not isinstance(context, dict) or len(context) > 20:
        raise DetectionRequestError("metadata.context is invalid")
    unknown_context = sorted(set(context) - ALLOWED_CONTEXT_FIELDS)
    if unknown_context:
        raise DetectionRequestError(
            "unsupported metadata.context field(s): " + ", ".join(unknown_context)
        )
    for field, item in context.items():
        if isinstance(item, str):
            if len(item) > 500 or "://" in item:
                raise DetectionRequestError(f"metadata.context.{field} is invalid")
        elif isinstance(item, list):
            if len(item) > 100 or any(
                not isinstance(entry, str) or not entry or len(entry) > 200 or "://" in entry
                for entry in item
            ):
                raise DetectionRequestError(f"metadata.context.{field} is invalid")
        elif item is not None and (
            not isinstance(item, (int, float)) or isinstance(item, bool)
        ):
            raise DetectionRequestError(f"metadata.context.{field} is invalid")
        elif isinstance(item, (int, float)) and not math.isfinite(item):
            raise DetectionRequestError(f"metadata.context.{field} is invalid")
    session_id = context.get("live_monitoring_session_id")
    if session_id is not None and (
        not isinstance(session_id, str) or not SAFE_IDENTIFIER.fullmatch(session_id)
    ):
        raise DetectionRequestError(
            "metadata.context.live_monitoring_session_id is invalid"
        )
    recipe_digest = context.get("recipe_sha256")
    if recipe_digest is not None and (
        not isinstance(recipe_digest, str)
        or len(recipe_digest) != 64
        or any(character not in "0123456789abcdef" for character in recipe_digest)
    ):
        raise DetectionRequestError("metadata.context.recipe_sha256 is invalid")
    return value


def validate_feature_identity(metadata: Mapping[str, Any], record: Mapping[str, Any], width: int) -> None:
    stored = record.get("feature_identity", {})
    stored_order = stored.get("feature_order") or []
    stored_hash = stored.get("feature_order_sha256")
    supplied_order = metadata.get("feature_order")
    supplied_hash = metadata.get("feature_order_sha256")
    if supplied_order is not None:
        if len(supplied_order) != width or list(supplied_order) != list(stored_order):
            raise DetectionRequestError("feature order does not match the trained model")
        supplied_hash = feature_order_sha256(supplied_order)
    elif stored_order and len(stored_order) != width:
        raise DetectionRequestError("feature count does not match the trained model")
    if stored_hash and supplied_hash != stored_hash:
        raise DetectionRequestError("feature-order hash does not match the trained model")


def validate_prediction(result: Any, input_count: int) -> dict[str, Any]:
    if not isinstance(result, PredictionResult):
        raise PredictionContractError("adapter returned an invalid PredictionResult")
    predictions = np.asarray(result.binary_predictions)
    scores = np.asarray(result.anomaly_scores)
    if predictions.ndim != 1 or scores.ndim != 1 or predictions.shape != scores.shape:
        raise PredictionContractError("prediction and score outputs must be equal one-dimensional arrays")
    if not isinstance(result.warmup_observations, int) or result.warmup_observations < 0:
        raise PredictionContractError("warmup_observations is invalid")
    if predictions.size != input_count - result.warmup_observations:
        raise PredictionContractError("prediction count does not match input and warmup")
    if not np.isin(predictions, (0, 1)).all() or not np.isfinite(scores).all():
        raise PredictionContractError("predictions or anomaly scores are invalid")
    diagnostics = {}
    for key, value in list(dict(result.diagnostics).items())[:20]:
        if value is None or isinstance(value, (str, int, float, bool)):
            diagnostics[str(key)[:100]] = value[:500] if isinstance(value, str) else value
    json.dumps(diagnostics)
    anomaly_count = int(predictions.sum())
    count = int(predictions.size)
    return {
        "warmup_observations": result.warmup_observations,
        "prediction_count": count,
        "anomaly_count": anomaly_count,
        "anomaly_percentage": float(anomaly_count / count * 100.0) if count else 0.0,
        "diagnostics": diagnostics,
    }
