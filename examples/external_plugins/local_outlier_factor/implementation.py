"""Runtime implementation for the external Local Outlier Factor plugin."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DETECTOR_ID = "local-outlier-factor"
DETECTOR_VERSION = "1.0.0"
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "model_metadata.json"
EVALUATION_FILENAME = "model_evaluation.json"

DEFAULT_PARAMETERS: dict[str, Any] = {
    "n_neighbors": 20,
    "metric": "minkowski",
    "contamination": "auto",
    "algorithm": "auto",
    "leaf_size": 30,
    "p": 2,
}
ALLOWED_METRICS = {"minkowski", "euclidean", "manhattan", "chebyshev"}
ALLOWED_ALGORITHMS = {"auto", "ball_tree", "kd_tree", "brute"}


class LocalOutlierFactorInputError(ValueError):
    """Raised when snapshot data or detector parameters are incompatible."""


class LocalOutlierFactorArtifactError(RuntimeError):
    """Raised when persisted model artifacts are absent or inconsistent."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(payload, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_numeric_matrix(path: Path, name: str) -> np.ndarray:
    try:
        frame = pd.read_csv(path, header=None, skip_blank_lines=False)
    except Exception as exc:
        raise LocalOutlierFactorInputError(
            f"{name} must be a readable headerless CSV matrix"
        ) from exc
    matrix = frame.to_numpy(copy=True)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise LocalOutlierFactorInputError(
            f"{name} must be a non-empty two-dimensional matrix"
        )
    if matrix.dtype.kind not in "iuf":
        raise LocalOutlierFactorInputError(
            f"{name} must contain numeric values without coercion"
        )
    if not np.isfinite(matrix).all():
        raise LocalOutlierFactorInputError(
            f"{name} must contain only finite values"
        )
    return matrix


def numeric_matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise LocalOutlierFactorInputError(
            f"{name} must be a non-empty two-dimensional matrix"
        )
    if matrix.dtype.kind not in "iuf" or not np.isfinite(matrix).all():
        raise LocalOutlierFactorInputError(
            f"{name} must contain only finite numeric values without coercion"
        )
    return matrix


def _read_binary_labels(path: Path | None, expected_length: int) -> np.ndarray:
    if path is None:
        raise LocalOutlierFactorInputError(
            "metrics-partition-v1 requires labels for training-time evaluation"
        )
    labels_matrix = _read_numeric_matrix(path, "labels.csv")
    if labels_matrix.shape[1] != 1:
        raise LocalOutlierFactorInputError(
            "labels.csv must contain exactly one label column"
        )
    labels = labels_matrix[:, 0]
    if labels.shape[0] != expected_length:
        raise LocalOutlierFactorInputError(
            "labels.csv must have the same number of observations as test.csv"
        )
    if not np.isin(labels, (0, 1)).all():
        raise LocalOutlierFactorInputError(
            "labels.csv may contain only 0 for normal and 1 for anomaly"
        )
    return labels.astype(np.int8, copy=False)


def load_training_data(
    train_path: Path, test_path: Path, labels_path: Path | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = _read_numeric_matrix(train_path, "train.csv")
    test = _read_numeric_matrix(test_path, "test.csv")
    labels = _read_binary_labels(labels_path, test.shape[0])
    return train, test, labels


def validate_parameters(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LocalOutlierFactorInputError("training parameters must be an object")
    unknown = sorted(set(value).difference(DEFAULT_PARAMETERS))
    if unknown:
        raise LocalOutlierFactorInputError(
            "unknown training parameter(s): " + ", ".join(unknown)
        )
    parameters = {**DEFAULT_PARAMETERS, **dict(value)}
    for name in ("n_neighbors", "leaf_size", "p"):
        if not isinstance(parameters[name], int) or isinstance(parameters[name], bool):
            raise LocalOutlierFactorInputError(
                f"training parameter '{name}' must be an integer"
            )
    if parameters["n_neighbors"] < 1:
        raise LocalOutlierFactorInputError(
            "training parameter 'n_neighbors' must be at least 1"
        )
    if not 1 <= parameters["leaf_size"] <= 10000:
        raise LocalOutlierFactorInputError(
            "training parameter 'leaf_size' must be between 1 and 10000"
        )
    if not 1 <= parameters["p"] <= 20:
        raise LocalOutlierFactorInputError(
            "training parameter 'p' must be between 1 and 20"
        )
    if parameters["metric"] not in ALLOWED_METRICS:
        raise LocalOutlierFactorInputError(
            "training parameter 'metric' is not supported"
        )
    if parameters["algorithm"] not in ALLOWED_ALGORITHMS:
        raise LocalOutlierFactorInputError(
            "training parameter 'algorithm' is not supported"
        )
    if parameters["contamination"] != "auto":
        raise LocalOutlierFactorInputError(
            "training parameter 'contamination' must be 'auto'"
        )
    return parameters


def validate_training_compatibility(
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    parameters: Mapping[str, Any],
    *,
    expected_features: int,
    feature_order: Any,
) -> None:
    if train.shape[1] != test.shape[1]:
        raise LocalOutlierFactorInputError(
            "test.csv must have the same feature count as train.csv"
        )
    if train.shape[1] != expected_features:
        raise LocalOutlierFactorInputError(
            "snapshot feature count does not match the training context"
        )
    if labels.shape[0] != test.shape[0]:
        raise LocalOutlierFactorInputError(
            "labels.csv must have the same number of observations as test.csv"
        )
    if parameters["n_neighbors"] >= train.shape[0]:
        raise LocalOutlierFactorInputError(
            "training parameter 'n_neighbors' must be smaller than the number of training observations"
        )
    if feature_order:
        if not isinstance(feature_order, (list, tuple)) or len(feature_order) != train.shape[1]:
            raise LocalOutlierFactorInputError(
                "feature order must contain exactly one identity per feature column"
            )


def create_pipeline(parameters: Mapping[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "detector",
                LocalOutlierFactor(novelty=True, **dict(parameters)),
            ),
        ]
    )


def _artifact_paths(directory: Path) -> tuple[Path, Path, Path]:
    return (
        directory / MODEL_FILENAME,
        directory / METADATA_FILENAME,
        directory / EVALUATION_FILENAME,
    )


def label_source(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    provenance = snapshot.get("catalogue_provenance")
    if isinstance(provenance, Mapping) and isinstance(
        provenance.get("label_source"), Mapping
    ):
        source = provenance["label_source"]
        return {
            "type": str(source.get("type", "catalogue"))[:100],
            **(
                {"sha256": source["sha256"]}
                if isinstance(source.get("sha256"), str)
                else {}
            ),
        }
    return {"type": "snapshot_labels"}


def train_and_persist(
    train: np.ndarray,
    artifact_dir: Path,
    *,
    parameters: Mapping[str, Any],
    model_id: str,
    feature_identity: Mapping[str, Any],
    train_observations: int,
    test_observations: int,
    label_source: Mapping[str, Any],
) -> dict[str, Any]:
    pipeline = create_pipeline(parameters)
    # Labels are deliberately absent from this fitting boundary.
    pipeline.fit(train)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path, metadata_path, _evaluation_path = _artifact_paths(artifact_dir)
    temporary_model = artifact_dir / f".{MODEL_FILENAME}.tmp"
    try:
        joblib.dump(pipeline, temporary_model)
        os.replace(temporary_model, model_path)
    except Exception:
        temporary_model.unlink(missing_ok=True)
        raise

    feature_order = feature_identity.get("feature_order", [])
    feature_hash = feature_identity.get("feature_order_sha256")
    metadata = {
        "schema_version": 1,
        "detector_id": DETECTOR_ID,
        "detector_version": DETECTOR_VERSION,
        "plugin_version": DETECTOR_VERSION,
        "model_id": model_id,
        "artifact_format": "joblib",
        "sklearn_version": sklearn.__version__,
        "n_features": int(train.shape[1]),
        "feature_identity": {
            "type": "positional",
            "feature_order": list(feature_order),
            "feature_order_sha256": feature_hash,
        },
        "preprocessing": {
            "type": "StandardScaler",
            "persisted_in_pipeline": True,
            "fit_source": "train.csv",
        },
        "training_parameters": dict(parameters),
        "train_observations": int(train_observations),
        "test_observations": int(test_observations),
        "label_source": dict(label_source),
        "label_semantics": {"0": "normal", "1": "anomaly"},
        "score_semantics": {
            "higher_is_more_anomalous": True,
            "binary_threshold": 0.0,
            "sklearn_anomaly_label": -1,
        },
    }
    _atomic_json(metadata_path, metadata)
    return metadata


def _load_pipeline(artifact_dir: Path) -> tuple[Pipeline, dict[str, Any]]:
    model_path, metadata_path, _evaluation_path = _artifact_paths(artifact_dir)
    if not model_path.is_file() or not metadata_path.is_file():
        raise LocalOutlierFactorArtifactError("LOF model artifacts are incomplete")
    try:
        pipeline = joblib.load(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LocalOutlierFactorArtifactError("LOF model artifacts cannot be loaded") from exc
    if metadata.get("detector_id") != DETECTOR_ID:
        raise LocalOutlierFactorArtifactError(
            "LOF model metadata belongs to another detector"
        )
    return pipeline, metadata


def predict(pipeline: Pipeline, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_prediction = pipeline.predict(test)
    binary_prediction = (raw_prediction == -1).astype(np.int8)
    anomaly_scores = -pipeline.decision_function(test)
    return binary_prediction, anomaly_scores


def evaluate_and_persist(
    test: np.ndarray, labels: np.ndarray, artifact_dir: Path
) -> dict[str, Any]:
    pipeline, metadata = _load_pipeline(artifact_dir)
    if test.shape[1] != metadata.get("n_features"):
        raise LocalOutlierFactorInputError(
            "test.csv feature count does not match the stored LOF model"
        )
    predictions, scores = predict(pipeline, test)
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
        "score_semantics": {
            "higher_is_more_anomalous": True,
            "binary_threshold": 0.0,
        },
    }
    _atomic_json(artifact_dir / EVALUATION_FILENAME, evaluation)
    return evaluation
