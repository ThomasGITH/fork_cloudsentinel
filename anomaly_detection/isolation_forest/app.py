"""Local Isolation Forest model upload and prediction service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from werkzeug.exceptions import RequestEntityTooLarge

from detectors import registry

from .storage import (
    DETECTOR_ID,
    EVALUATION_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    ArtifactConflictError,
    ArtifactValidationError,
    install_artifacts,
    safe_model_id,
    validate_active_artifacts,
    validate_checksum,
)


DEFAULT_STORAGE_ROOT = Path(__file__).resolve().parent / "storage"
DEFAULT_MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def _error(message: str, status: int):
    return jsonify({"error": message}), status


def _required_form_value(name: str) -> str:
    value = request.form.get(name)
    if value is None or not value.strip():
        raise ArtifactValidationError(f"missing required form field: {name}")
    return value


def _numeric_csv(file_storage: Any) -> np.ndarray:
    try:
        matrix = pd.read_csv(
            file_storage, header=None, skip_blank_lines=False
        ).to_numpy()
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise ArtifactValidationError(f"test_array must be a valid headerless CSV: {exc}") from exc
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ArtifactValidationError(
            "test_array must be a non-empty two-dimensional numeric matrix"
        )
    if matrix.dtype.kind not in "iuf":
        raise ArtifactValidationError("test_array must contain only numeric values")
    if not np.isfinite(matrix).all():
        raise ArtifactValidationError("test_array must contain only finite values")
    return matrix


def _parse_test_info(raw_value: str | None) -> tuple[str, str]:
    if raw_value is None:
        raise ArtifactValidationError("missing required form field: test_info")
    try:
        test_info = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError("test_info must contain valid JSON") from exc
    if not isinstance(test_info, dict) or not isinstance(test_info.get("data"), dict):
        raise ArtifactValidationError("test_info.data must be a JSON object")
    data = test_info["data"]
    if data.get("detector_id") != DETECTOR_ID:
        raise ArtifactValidationError(
            "test_info.data.detector_id must be 'isolation-forest'"
        )
    model_id = safe_model_id(data.get("model"))
    iteration = data.get("iteration")
    if not isinstance(iteration, (str, int)) or isinstance(iteration, bool):
        raise ArtifactValidationError(
            "test_info.data.iteration must be a string or integer"
        )
    return model_id, str(iteration)


def create_app(config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    app.config.from_mapping(
        IF_MODEL_STORAGE_ROOT=os.getenv(
            "IF_MODEL_STORAGE_ROOT", str(DEFAULT_STORAGE_ROOT / "models")
        ),
        IF_RESULTS_STORAGE_ROOT=os.getenv(
            "IF_RESULTS_STORAGE_ROOT", str(DEFAULT_STORAGE_ROOT / "results")
        ),
        MAX_CONTENT_LENGTH=int(
            os.getenv("IF_MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES))
        ),
    )
    if config:
        app.config.update(config)

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_error):
        return _error_response("upload exceeds the configured maximum size", 413)

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "healthy", "detector_id": DETECTOR_ID}), 200

    @app.post("/save_model")
    def save_model():
        try:
            detector_id = _required_form_value("detector_id")
            if detector_id != DETECTOR_ID:
                raise ArtifactValidationError(
                    "detector_id must be 'isolation-forest'"
                )
            model_id = safe_model_id(_required_form_value("model_id"))

            required_files = {
                "model": MODEL_FILENAME,
                "metadata": METADATA_FILENAME,
                "evaluation": EVALUATION_FILENAME,
            }
            missing_files = [field for field in required_files if field not in request.files]
            if missing_files:
                raise ArtifactValidationError(
                    f"missing required file(s): {', '.join(missing_files)}"
                )

            expected_hashes = {
                MODEL_FILENAME: validate_checksum(
                    _required_form_value("model_sha256"), "model_sha256"
                ),
                METADATA_FILENAME: validate_checksum(
                    _required_form_value("metadata_sha256"), "metadata_sha256"
                ),
                EVALUATION_FILENAME: validate_checksum(
                    _required_form_value("evaluation_sha256"), "evaluation_sha256"
                ),
            }
            uploads = {
                filename: request.files[field].stream
                for field, filename in required_files.items()
            }
            idempotent = install_artifacts(
                app.config["IF_MODEL_STORAGE_ROOT"],
                model_id,
                uploads,
                expected_hashes,
            )
            return jsonify(
                {
                    "status": "available",
                    "detector_id": DETECTOR_ID,
                    "model_id": model_id,
                    "idempotent": idempotent,
                }
            ), 200
        except ArtifactConflictError as exc:
            return _error(str(exc), 409)
        except ArtifactValidationError as exc:
            return _error(str(exc), 400)

    @app.post("/detect_anomalies")
    def detect_anomalies():
        if "test_array" not in request.files:
            return _error("missing required file: test_array", 400)
        try:
            matrix = _numeric_csv(request.files["test_array"])
            model_id, iteration = _parse_test_info(request.form.get("test_info"))
            model_directory, metadata = validate_active_artifacts(
                app.config["IF_MODEL_STORAGE_ROOT"], model_id
            )
            if matrix.shape[1] != metadata["n_features"]:
                raise ArtifactValidationError(
                    f"test_array has {matrix.shape[1]} features; "
                    f"the stored model expects {metadata['n_features']}"
                )

            adapter = registry.get_adapter(DETECTOR_ID)
            prediction = adapter.predict(matrix, model_directory)
            percentage = prediction.get("anomaly_percentage")
            binary_predictions = prediction.get("binary_predictions")
            anomaly_scores = prediction.get("anomaly_scores")
            if (
                not isinstance(percentage, (int, float))
                or isinstance(percentage, bool)
                or not 0.0 <= float(percentage) <= 100.0
                or not isinstance(binary_predictions, list)
                or not isinstance(anomaly_scores, list)
                or len(binary_predictions) != matrix.shape[0]
                or len(anomaly_scores) != matrix.shape[0]
            ):
                raise RuntimeError("detector returned an invalid prediction result")

            return jsonify(
                {
                    "status": "success",
                    "detector_id": DETECTOR_ID,
                    "model_id": model_id,
                    "iteration": iteration,
                    "percentage": float(percentage),
                }
            ), 200
        except FileNotFoundError as exc:
            return _error(str(exc), 404)
        except ArtifactValidationError as exc:
            return _error(str(exc), 400)
        except Exception as exc:
            return _error(f"model prediction failed: {exc}", 422)

    return app


def _error_response(message: str, status: int):
    return jsonify({"error": message}), status


app = create_app()


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5014)
