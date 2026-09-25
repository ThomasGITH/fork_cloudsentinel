"""Lightweight request validation for Isolation Forest training."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from detectors.registry import load_manifest


MANIFEST_PATH = Path(__file__).with_name("manifest.yaml")
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class IsolationForestRequestError(ValueError):
    """Raised when a training request cannot be safely dispatched."""


def _matrix(value: Any, name: str, *, require_multiple_rows: bool = False) -> np.ndarray:
    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise IsolationForestRequestError(
            f"{name} must be a rectangular two-dimensional numeric matrix"
        ) from exc
    if matrix.ndim != 2:
        raise IsolationForestRequestError(
            f"{name} must be a two-dimensional numeric matrix"
        )
    if matrix.dtype.kind not in "iuf":
        raise IsolationForestRequestError(f"{name} must contain only numeric values")
    if matrix.shape[1] < 1:
        raise IsolationForestRequestError(f"{name} must contain at least one feature")
    if require_multiple_rows and matrix.shape[0] <= 1:
        raise IsolationForestRequestError(
            f"{name} must contain more than one observation"
        )
    if not np.isfinite(matrix).all():
        raise IsolationForestRequestError(f"{name} must contain only finite values")
    return matrix


def _labels(value: Any, expected_length: int) -> np.ndarray:
    try:
        labels = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise IsolationForestRequestError(
            "anomaly_label_array must be a rectangular one- or two-dimensional array"
        ) from exc
    if labels.ndim == 2:
        if 1 not in labels.shape:
            raise IsolationForestRequestError(
                "anomaly_label_array must be one-dimensional or a single-row/single-column matrix"
            )
        labels = labels.reshape(-1)
    elif labels.ndim != 1:
        raise IsolationForestRequestError(
            "anomaly_label_array must be one-dimensional or two-dimensional"
        )
    if labels.dtype.kind not in "iuf" or not np.isfinite(labels).all():
        raise IsolationForestRequestError(
            "anomaly_label_array must contain only finite numeric values"
        )
    if labels.shape[0] != expected_length:
        raise IsolationForestRequestError(
            "anomaly_label_array must have the same number of observations as test_array"
        )
    if not np.isin(labels, (0, 1)).all():
        raise IsolationForestRequestError(
            "anomaly_label_array may contain only 0 and 1"
        )
    return labels.astype(np.int8, copy=False)


def validate_training_parameters(value: Any) -> dict[str, Any]:
    """Apply manifest defaults and validate the supported initial parameter set."""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise IsolationForestRequestError("training_parameters must be a JSON object")

    manifest_parameters = load_manifest(MANIFEST_PATH)["training_parameters"]
    unknown = sorted(set(value).difference(manifest_parameters))
    if unknown:
        raise IsolationForestRequestError(
            f"unknown training parameter(s): {', '.join(unknown)}"
        )
    parameters = {
        name: value.get(name, metadata["default"])
        for name, metadata in manifest_parameters.items()
    }

    for name in ("n_estimators", "random_state", "n_jobs"):
        if not isinstance(parameters[name], int) or isinstance(parameters[name], bool):
            raise IsolationForestRequestError(
                f"training parameter '{name}' must be an integer"
            )
    if parameters["n_estimators"] < 1:
        raise IsolationForestRequestError(
            "training parameter 'n_estimators' must be positive"
        )
    if parameters["n_jobs"] == 0:
        raise IsolationForestRequestError(
            "training parameter 'n_jobs' must not be zero"
        )
    if parameters["max_samples"] != "auto":
        raise IsolationForestRequestError(
            "training parameter 'max_samples' must be 'auto'"
        )
    if parameters["contamination"] != "auto":
        raise IsolationForestRequestError(
            "training parameter 'contamination' must be 'auto'"
        )
    if (
        not isinstance(parameters["max_features"], (int, float))
        or isinstance(parameters["max_features"], bool)
        or not 0 < parameters["max_features"] <= 1
    ):
        raise IsolationForestRequestError(
            "training parameter 'max_features' must be greater than 0 and at most 1"
        )
    if not isinstance(parameters["bootstrap"], bool):
        raise IsolationForestRequestError(
            "training parameter 'bootstrap' must be boolean"
        )
    return parameters


def validate_model_id(value: Any) -> str:
    if not isinstance(value, str) or not MODEL_ID_PATTERN.fullmatch(value):
        raise IsolationForestRequestError(
            "model_id must contain 1-128 letters, digits, dots, underscores, or hyphens"
        )
    if value in {".", ".."}:
        raise IsolationForestRequestError("model_id must identify a model directory")
    return value


def validate_training_data(
    train_array: Any,
    test_array: Any,
    anomaly_label_array: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = _matrix(train_array, "train_array", require_multiple_rows=True)
    test = _matrix(test_array, "test_array")
    if train.shape[1] != test.shape[1]:
        raise IsolationForestRequestError(
            "test_array must have the same number of features as train_array"
        )
    labels = _labels(anomaly_label_array, test.shape[0])
    return train, test, labels

