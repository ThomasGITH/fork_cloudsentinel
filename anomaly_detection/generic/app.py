"""Single detector-agnostic HTTP runtime for batch anomaly detection."""

from __future__ import annotations

import os
from pathlib import Path
import re
import time

from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from detectors.contracts import (
    DetectorInferenceError,
    InferenceContext,
    PredictionContractError,
)
from detectors.model_artifacts import ModelArtifactError
from learning_adaptation.model_catalogue import ModelNotFoundError, ModelRecordError

from .model_cache import ModelCache
from .model_repository import GenericModelRepository, ModelNotReadyError
from .results import DetectionResultStore
from .validation import (
    DetectionRequestError,
    parse_metadata,
    read_matrix,
    validate_feature_identity,
    validate_prediction,
)


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        MODEL_CATALOGUE_STORAGE_ROOT=os.getenv("MODEL_CATALOGUE_STORAGE_ROOT", "model_catalogue"),
        MODEL_ARTIFACT_STORAGE_ROOT=os.getenv("MODEL_ARTIFACT_STORAGE_ROOT", "model_artifacts"),
        GENERIC_DETECTION_RESULTS_ROOT=os.getenv("GENERIC_DETECTION_RESULTS_ROOT", "generic_detection_results"),
        MAX_CONTENT_LENGTH=int(os.getenv("GENERIC_DETECTION_MAX_REQUEST_BYTES", str(10 * 1024 * 1024))),
        MAX_OBSERVATIONS=int(os.getenv("GENERIC_DETECTION_MAX_OBSERVATIONS", "100000")),
        MAX_FEATURES=int(os.getenv("GENERIC_DETECTION_MAX_FEATURES", "1000")),
        MODEL_CACHE_SIZE=int(os.getenv("GENERIC_DETECTION_MODEL_CACHE_SIZE", "4")),
        MAX_EVALUATION_OUTPUTS=int(
            os.getenv("GENERIC_DETECTION_MAX_EVALUATION_OUTPUTS", "100000")
        ),
    )
    if config:
        app.config.update(config)
    repository = GenericModelRepository(
        app.config["MODEL_CATALOGUE_STORAGE_ROOT"],
        app.config["MODEL_ARTIFACT_STORAGE_ROOT"],
    )
    cache = ModelCache(int(app.config["MODEL_CACHE_SIZE"]))
    results = DetectionResultStore(app.config["GENERIC_DETECTION_RESULTS_ROOT"])
    app.extensions["generic_model_repository"] = repository
    app.extensions["generic_model_cache"] = cache

    def error(message: str, status: int):
        safe = re.sub(r"\b(?:https?|redis)://\S+", "[service URL redacted]", str(message))
        safe = re.sub(
            r"(?<![A-Za-z0-9_.-])/(?:[^\s/:]+/)+[^\s:]+",
            "[path redacted]",
            safe,
        )
        safe = re.sub(r"\b(?:task|celery)[-_][A-Za-z0-9._-]+", "[task redacted]", safe, flags=re.I)
        return jsonify({"status": "error", "error": safe.replace("\n", " ")[:500]}), status

    def activated(model_id: str):
        record = repository.record(model_id)
        inference = record.get("inference") or {"status": "legacy_external"}
        key = (model_id, str(inference.get("artifact_manifest_sha256", "")))
        value = cache.get(key)
        return record, inference, key, value

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        return error("request exceeds configured size limit", 413)

    @app.get("/healthz")
    def health():
        return jsonify({"status": "ok", "service": "generic-anomaly-detection"})

    @app.post("/models/<model_id>/activate")
    def activate(model_id: str):
        try:
            record, verified, adapter, context = repository.prepare(model_id)
            key = (model_id, verified["manifest_sha256"])
            cache.load(key, lambda: (adapter, adapter.load_model(context)))
            return jsonify(
                {
                    "status": "ready",
                    "model_id": model_id,
                    "detector_id": record["detector_id"],
                    "detector_version": record["detector_version"],
                    "artifact_manifest_sha256": verified["manifest_sha256"],
                }
            )
        except ModelNotFoundError:
            return error("unknown model", 404)
        except ModelNotReadyError as exc:
            return error(str(exc), 409)
        except (ModelArtifactError, ModelRecordError, DetectorInferenceError) as exc:
            return error(str(exc), 422)
        except Exception:
            return error("model activation failed", 500)

    @app.get("/models/<model_id>/runtime-status")
    def runtime_status(model_id: str):
        try:
            record, inference, key, value = activated(model_id)
            return jsonify(
                {
                    "model_id": model_id,
                    "detector_id": record["detector_id"],
                    "detector_version": record["detector_version"],
                    "artifact_manifest_sha256": inference["artifact_manifest_sha256"],
                    "status": "ready" if value is not None else inference.get("status", "not_ready"),
                    "activated": value is not None,
                }
            )
        except ModelNotFoundError:
            return error("unknown model", 404)
        except Exception:
            return error("runtime status unavailable", 500)

    @app.post("/detect")
    def detect():
        try:
            metadata = parse_metadata(request.form.get("metadata"))
            matrix_file = request.files.get("matrix")
            if matrix_file is None:
                raise DetectionRequestError("matrix file is required")
            matrix = read_matrix(
                matrix_file,
                max_observations=int(app.config["MAX_OBSERVATIONS"]),
                max_features=int(app.config["MAX_FEATURES"]),
            )
            record, inference, key, loaded = activated(metadata["model_id"])
            if inference.get("status") != "ready":
                raise ModelNotReadyError(
                    f"model is not ready for generic inference ({inference.get('status')})"
                )
            if loaded is None:
                raise ModelNotReadyError("model must be activated before detection")
            validate_feature_identity(metadata, record, matrix.shape[1])
            timestamps = tuple(metadata.get("timestamps") or ())
            if timestamps and len(timestamps) != matrix.shape[0]:
                raise DetectionRequestError("timestamps must match matrix observations")
            adapter, opaque_model = loaded
            prediction = adapter.predict_inference(
                opaque_model,
                InferenceContext(
                    matrix=matrix,
                    feature_order=tuple(metadata.get("feature_order") or ()),
                    feature_order_sha256=metadata.get("feature_order_sha256"),
                    timestamps=timestamps,
                    metadata=metadata.get("context") or {},
                ),
            )
            summary = validate_prediction(prediction, matrix.shape[0])
            result_reference = results.write(
                {
                    "model_id": record["model_id"],
                    "detector_id": record["detector_id"],
                    "detector_version": record["detector_version"],
                    "artifact_manifest_sha256": inference["artifact_manifest_sha256"],
                    "input_observation_count": int(matrix.shape[0]),
                    **summary,
                }
            )
            return jsonify(
                {
                    "status": "success",
                    "model_id": record["model_id"],
                    "detector_id": record["detector_id"],
                    "detector_version": record["detector_version"],
                    "input_observation_count": int(matrix.shape[0]),
                    **{key: value for key, value in summary.items() if key != "diagnostics"},
                    "result_reference": result_reference,
                }
            )
        except DetectionRequestError as exc:
            return error(str(exc), 400)
        except ModelNotFoundError:
            return error("unknown model", 404)
        except ModelNotReadyError as exc:
            return error(str(exc), 409)
        except PredictionContractError as exc:
            return error(str(exc), 500)
        except DetectorInferenceError as exc:
            return error(str(exc), 422)
        except (ModelArtifactError, ModelRecordError) as exc:
            return error(str(exc), 503)
        except RequestEntityTooLarge:
            return error("request exceeds configured size limit", 413)
        except Exception:
            return error("detection failed", 500)

    @app.post("/internal/evaluate")
    def internal_evaluate():
        """Return bounded per-observation output to trusted comparison workers.

        This route uses the exact same activated model cache and adapter contract
        as `/detect`. It is intentionally exposed only by the internal ClusterIP
        service; browsers receive comparison summaries from learning adaptation.
        """
        try:
            metadata = parse_metadata(request.form.get("metadata"))
            matrix_file = request.files.get("matrix")
            if matrix_file is None:
                raise DetectionRequestError("matrix file is required")
            matrix = read_matrix(
                matrix_file,
                max_observations=min(
                    int(app.config["MAX_OBSERVATIONS"]),
                    int(app.config["MAX_EVALUATION_OUTPUTS"]),
                ),
                max_features=int(app.config["MAX_FEATURES"]),
            )
            record, inference, key, loaded = activated(metadata["model_id"])
            if inference.get("status") != "ready":
                raise ModelNotReadyError(
                    f"model is not ready for generic inference ({inference.get('status')})"
                )
            if loaded is None:
                raise ModelNotReadyError("model must be activated before evaluation")
            validate_feature_identity(metadata, record, matrix.shape[1])
            timestamps = tuple(metadata.get("timestamps") or ())
            if timestamps and len(timestamps) != matrix.shape[0]:
                raise DetectionRequestError("timestamps must match matrix observations")
            adapter, opaque_model = loaded
            started = time.perf_counter()
            prediction = adapter.predict_inference(
                opaque_model,
                InferenceContext(
                    matrix=matrix,
                    feature_order=tuple(metadata.get("feature_order") or ()),
                    feature_order_sha256=metadata.get("feature_order_sha256"),
                    timestamps=timestamps,
                    metadata=metadata.get("context") or {},
                ),
            )
            runtime_ms = (time.perf_counter() - started) * 1000.0
            summary = validate_prediction(prediction, matrix.shape[0])
            predictions = [int(value) for value in prediction.binary_predictions]
            scores = [float(value) for value in prediction.anomaly_scores]
            if len(predictions) > int(app.config["MAX_EVALUATION_OUTPUTS"]):
                raise DetectionRequestError("evaluation output exceeds configured limit")
            return jsonify(
                {
                    "status": "success",
                    "model_id": record["model_id"],
                    "detector_id": record["detector_id"],
                    "detector_version": record["detector_version"],
                    "artifact_manifest_sha256": inference["artifact_manifest_sha256"],
                    "input_observation_count": int(matrix.shape[0]),
                    "runtime_ms": runtime_ms,
                    "warmup_observations": summary["warmup_observations"],
                    "prediction_count": summary["prediction_count"],
                    "anomaly_count": summary["anomaly_count"],
                    "binary_predictions": predictions,
                    "anomaly_scores": scores,
                }
            )
        except DetectionRequestError as exc:
            return error(str(exc), 400)
        except ModelNotFoundError:
            return error("unknown model", 404)
        except ModelNotReadyError as exc:
            return error(str(exc), 409)
        except DetectorInferenceError as exc:
            return error(str(exc), 422)
        except PredictionContractError as exc:
            return error(str(exc), 500)
        except (ModelArtifactError, ModelRecordError) as exc:
            return error(str(exc), 503)
        except RequestEntityTooLarge:
            return error("request exceeds configured size limit", 413)
        except Exception:
            return error("model evaluation failed", 500)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5015, debug=False)
