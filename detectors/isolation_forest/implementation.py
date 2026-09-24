"""Runtime implementation for the Isolation Forest detector plugin."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DETECTOR_ID = "isolation-forest"
DETECTOR_VERSION = "1.0.0"
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "model_metadata.json"
EVALUATION_FILENAME = "model_evaluation.json"

DEFAULT_TRAINING_PARAMETERS: dict[str, Any] = {
    "n_estimators": 100,
    "max_samples": "auto",
    "contamination": "auto",
    "max_features": 1.0,
    "bootstrap": False,
    "random_state": 42,
    "n_jobs": 1,
}


class IsolationForestPluginError(Exception):
    """Base error for Isolation Forest plugin operations."""


class IsolationForestInputError(IsolationForestPluginError, ValueError):
    """Raised when detector input violates the plugin contract."""


class IsolationForestArtifactError(IsolationForestPluginError):
    """Raised when a model artifact is absent or inconsistent."""


def _numeric_matrix(value: Any, name: str, *, more_than_one_row: bool = False) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise IsolationForestInputError(
            f"{name} must be a rectangular two-dimensional numeric matrix"
        ) from exc
    if array.ndim != 2:
        raise IsolationForestInputError(f"{name} must be a two-dimensional numeric matrix")
    if array.dtype.kind not in "iuf":
        raise IsolationForestInputError(f"{name} must contain numeric values without coercion")
    if array.shape[1] < 1:
        raise IsolationForestInputError(f"{name} must contain at least one feature")
    if more_than_one_row and array.shape[0] <= 1:
        raise IsolationForestInputError(f"{name} must contain more than one observation")
    if not np.isfinite(array).all():
        raise IsolationForestInputError(f"{name} must contain only finite values")
    return array


def _binary_labels(value: Any, expected_length: int) -> np.ndarray:
    try:
        labels = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise IsolationForestInputError(
            "anomaly_label_array must be a rectangular one- or two-dimensional array"
        ) from exc
    if labels.ndim == 2:
        if 1 not in labels.shape:
            raise IsolationForestInputError(
                "anomaly_label_array must be one-dimensional or a single-row/single-column matrix"
            )
        labels = labels.reshape(-1)
    elif labels.ndim != 1:
        raise IsolationForestInputError(
            "anomaly_label_array must be one-dimensional or two-dimensional"
        )
    if labels.dtype.kind not in "iuf":
        raise IsolationForestInputError(
            "anomaly_label_array must contain numeric binary values without coercion"
        )
    if not np.isfinite(labels).all():
        raise IsolationForestInputError("anomaly_label_array must contain only finite values")
    if labels.shape[0] != expected_length:
        raise IsolationForestInputError(
            "anomaly_label_array must have the same number of observations as test_array"
        )
    if not np.isin(labels, (0, 1)).all():
        raise IsolationForestInputError("anomaly_label_array may contain only 0 and 1")
    return labels.astype(np.int8, copy=False)


def _validated_parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    if overrides is None:
        return DEFAULT_TRAINING_PARAMETERS.copy()
    if not isinstance(overrides, dict):
        raise IsolationForestInputError("training_parameters must be a mapping")
    unknown = sorted(set(overrides).difference(DEFAULT_TRAINING_PARAMETERS))
    if unknown:
        raise IsolationForestInputError(
            f"unknown training parameter(s): {', '.join(unknown)}"
        )
    parameters = {**DEFAULT_TRAINING_PARAMETERS, **overrides}

    for name in ("n_estimators", "random_state", "n_jobs"):
        if not isinstance(parameters[name], int) or isinstance(parameters[name], bool):
            raise IsolationForestInputError(f"training parameter '{name}' must be an integer")
    if parameters["n_estimators"] < 1:
        raise IsolationForestInputError("training parameter 'n_estimators' must be positive")
    if parameters["n_jobs"] == 0:
        raise IsolationForestInputError("training parameter 'n_jobs' must not be zero")
    if parameters["max_samples"] != "auto":
        raise IsolationForestInputError("training parameter 'max_samples' must be 'auto'")
    if parameters["contamination"] != "auto":
        raise IsolationForestInputError("training parameter 'contamination' must be 'auto'")
    if (
        not isinstance(parameters["max_features"], (int, float))
        or isinstance(parameters["max_features"], bool)
        or not 0 < parameters["max_features"] <= 1
    ):
        raise IsolationForestInputError(
            "training parameter 'max_features' must be a number greater than 0 and at most 1"
        )
    if not isinstance(parameters["bootstrap"], bool):
        raise IsolationForestInputError("training parameter 'bootstrap' must be boolean")
    return parameters


def _artifact_paths(artifact_dir: str | PathLike[str]) -> tuple[Path, Path, Path]:
    directory = Path(artifact_dir)
    return (
        directory / MODEL_FILENAME,
        directory / METADATA_FILENAME,
        directory / EVALUATION_FILENAME,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_artifacts(artifact_dir: str | PathLike[str]) -> tuple[Pipeline, dict[str, Any]]:
    model_path, metadata_path, _ = _artifact_paths(artifact_dir)
    if not model_path.is_file():
        raise IsolationForestArtifactError(f"model artifact does not exist: {model_path}")
    if not metadata_path.is_file():
        raise IsolationForestArtifactError(f"model metadata does not exist: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = joblib.load(model_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise IsolationForestArtifactError(
            f"could not load Isolation Forest artifacts from {Path(artifact_dir)}: {exc}"
        ) from exc
    if metadata.get("detector_id") != DETECTOR_ID:
        raise IsolationForestArtifactError("model metadata belongs to a different detector")
    return model, metadata


def _validate_feature_count(matrix: np.ndarray, metadata: dict[str, Any], name: str) -> None:
    expected = metadata.get("n_features")
    if matrix.shape[1] != expected:
        raise IsolationForestInputError(
            f"{name} has {matrix.shape[1]} features; the stored model expects {expected}"
        )


def _predictions(model: Pipeline, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sklearn_predictions = model.predict(matrix)
    binary_predictions = (sklearn_predictions == -1).astype(np.int8)
    anomaly_scores = -model.decision_function(matrix)
    return binary_predictions, anomaly_scores


def train_model(
    train_array: Any,
    artifact_dir: str | PathLike[str],
    *,
    training_parameters: dict[str, Any] | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Fit and persist a StandardScaler/IsolationForest pipeline."""
    matrix = _numeric_matrix(train_array, "train_array", more_than_one_row=True)
    parameters = _validated_parameters(training_parameters)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("detector", IsolationForest(**parameters)),
        ]
    )
    pipeline.fit(matrix)

    model_path, metadata_path, _ = _artifact_paths(artifact_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "detector_id": DETECTOR_ID,
        "detector_version": DETECTOR_VERSION,
        "model_id": model_id or "unassigned",
        "artifact_format": "joblib",
        "sklearn_version": sklearn.__version__,
        "n_features": int(matrix.shape[1]),
        "feature_identity": "positional",
        "preprocessing": "StandardScaler persisted in pipeline",
        "training_parameters": parameters,
        "score_semantics": {
            "higher_is_more_anomalous": True,
            "binary_threshold": 0.0,
        },
    }
    joblib.dump(pipeline, model_path)
    _write_json(metadata_path, metadata)
    return metadata


def predict_with_model(
    test_array: Any,
    artifact_dir: str | PathLike[str],
) -> dict[str, Any]:
    """Load a persisted pipeline and return scores and binary predictions."""
    matrix = _numeric_matrix(test_array, "test_array")
    model, metadata = _load_artifacts(artifact_dir)
    _validate_feature_count(matrix, metadata, "test_array")
    binary_predictions, anomaly_scores = _predictions(model, matrix)
    percentage = (
        float(binary_predictions.mean() * 100.0) if binary_predictions.size else 0.0
    )
    return {
        "binary_predictions": binary_predictions.tolist(),
        "anomaly_scores": anomaly_scores.tolist(),
        "anomaly_percentage": percentage,
    }


def evaluate_model(
    test_array: Any,
    anomaly_label_array: Any,
    artifact_dir: str | PathLike[str],
) -> dict[str, Any]:
    """Evaluate a persisted pipeline and write its binary classification metrics."""
    matrix = _numeric_matrix(test_array, "test_array")
    labels = _binary_labels(anomaly_label_array, matrix.shape[0])
    model, metadata = _load_artifacts(artifact_dir)
    _validate_feature_count(matrix, metadata, "test_array")
    predictions, scores = _predictions(model, matrix)

    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    evaluation = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "test_observations": int(labels.size),
        "actual_anomalies": int(labels.sum()),
        "predicted_anomalies": int(predictions.sum()),
        "binary_predictions": predictions.tolist(),
        "anomaly_scores": scores.tolist(),
    }
    _, _, evaluation_path = _artifact_paths(artifact_dir)
    _write_json(evaluation_path, evaluation)
    return evaluation
