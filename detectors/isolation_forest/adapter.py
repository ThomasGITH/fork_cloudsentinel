"""Thin lazy adapter for the Isolation Forest plugin implementation."""

from __future__ import annotations

from importlib import import_module
from os import PathLike
from typing import Any

from detectors.contracts import (
    ArtifactResult,
    DetectorCompatibilityError,
    DetectorExecutionError,
    DetectorPromotionError,
    InferenceCompatibilityError,
    InferenceContext,
    ModelLoadContext,
    ModelLoadError,
    PredictionResult,
    ProgressReporter,
    PromotionResult,
    TrainingContext,
    TrainingResult,
)


def _promotion_runtime() -> Any:
    """Load promotion code from source checkouts and flat service images."""
    try:
        return import_module("learning_adaptation.isolation_forest_promotion")
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "learning_adaptation",
            "learning_adaptation.isolation_forest_promotion",
        }:
            raise
        # The learning-adaptation image copies the service modules directly
        # into /app, while repository execution exposes them as a package.
        return import_module("isolation_forest_promotion")


class IsolationForestAdapter:
    """Load the Isolation Forest runtime only when an operation is invoked."""

    @staticmethod
    def train(
        train_array: Any,
        artifact_dir: str | PathLike[str],
        *,
        training_parameters: dict[str, Any] | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        implementation = import_module("detectors.isolation_forest.implementation")
        return implementation.train_model(
            train_array,
            artifact_dir,
            training_parameters=training_parameters,
            model_id=model_id,
        )

    @staticmethod
    def evaluate(
        test_array: Any,
        anomaly_label_array: Any,
        artifact_dir: str | PathLike[str],
    ) -> dict[str, Any]:
        implementation = import_module("detectors.isolation_forest.implementation")
        return implementation.evaluate_model(test_array, anomaly_label_array, artifact_dir)

    @staticmethod
    def predict(
        test_array: Any,
        artifact_dir: str | PathLike[str],
    ) -> dict[str, Any]:
        implementation = import_module("detectors.isolation_forest.implementation")
        return implementation.predict_with_model(test_array, artifact_dir)

    @staticmethod
    def _load_training_data(context: TrainingContext) -> tuple[Any, Any, Any]:
        try:
            pd = import_module("pandas")
            train = pd.read_csv(
                context.data.train_path, header=None, skip_blank_lines=False
            ).to_numpy()
            test = pd.read_csv(
                context.data.test_path, header=None, skip_blank_lines=False
            ).to_numpy()
            if context.data.labels_path is None:
                raise DetectorCompatibilityError(
                    "Isolation Forest requires labels for its current training-time evaluation"
                )
            labels = pd.read_csv(
                context.data.labels_path, header=None, skip_blank_lines=False
            ).to_numpy()
            training_request = import_module(
                "detectors.isolation_forest.training_request"
            )
            return training_request.validate_training_data(train, test, labels)
        except DetectorCompatibilityError:
            raise
        except Exception as exc:
            raise DetectorCompatibilityError(
                f"Isolation Forest input is incompatible: {exc}"
            ) from exc

    def validate_training(self, context: TrainingContext) -> None:
        self._load_training_data(context)
        try:
            validator = import_module(
                "detectors.isolation_forest.training_request"
            ).validate_training_parameters
            validator(dict(context.parameters))
        except Exception as exc:
            raise DetectorCompatibilityError(str(exc)) from exc

    def run_training(
        self, context: TrainingContext, progress: ProgressReporter
    ) -> TrainingResult:
        artifact_dir = (
            context.artifact_workspace / "isolation-forest" / context.model_id
        )
        try:
            train, test, labels = self._load_training_data(context)
            progress.report("TRAINING", "Training the Isolation Forest model")
            metadata = self.train(
                train,
                artifact_dir,
                training_parameters=dict(context.parameters),
                model_id=context.model_id,
            )
            progress.report("EVALUATING", "Evaluating the Isolation Forest model")
            evaluation = self.evaluate(test, labels, artifact_dir)
            progress.report("PROMOTING", "Promoting the Isolation Forest model")
            try:
                promotion_payload = _promotion_runtime().promote_isolation_forest_model(
                    artifact_dir, context.model_id
                )
            except Exception as exc:
                raise DetectorPromotionError(
                    f"Isolation Forest promotion failed: {exc}"
                ) from exc
            return TrainingResult(
                status="completed",
                artifact=ArtifactResult(
                    directory=artifact_dir,
                    required_files=(
                        "model.joblib",
                        "model_metadata.json",
                        "model_evaluation.json",
                    ),
                    format="joblib",
                    safe_reference=context.model_id,
                ),
                evaluation=evaluation,
                promotion=PromotionResult(
                    status="promoted",
                    safe_reference=promotion_payload.get("model_id"),
                ),
                model_metadata={
                    "n_features": metadata["n_features"],
                    "training_parameters": metadata["training_parameters"],
                },
                cleanup="remove_after_catalogue",
            )
        except (DetectorCompatibilityError, DetectorPromotionError):
            raise
        except Exception as exc:
            raise DetectorExecutionError(
                f"Isolation Forest training failed: {exc}"
            ) from exc

    def load_model(self, context: ModelLoadContext) -> Any:
        try:
            implementation = import_module("detectors.isolation_forest.implementation")
            model, metadata = implementation._load_artifacts(
                context.artifact_directory
            )
            if metadata.get("n_features") != context.model_record.get(
                "model_metadata", {}
            ).get("n_features"):
                raise ModelLoadError("Isolation Forest feature metadata mismatch")
            return {"pipeline": model, "metadata": metadata}
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError("Isolation Forest model could not be loaded") from exc

    def predict_inference(
        self, model: Any, context: InferenceContext
    ) -> PredictionResult:
        try:
            implementation = import_module("detectors.isolation_forest.implementation")
            matrix = implementation._numeric_matrix(context.matrix, "matrix")
            implementation._validate_feature_count(
                matrix, model["metadata"], "matrix"
            )
            predictions, scores = implementation._predictions(
                model["pipeline"], matrix
            )
            return PredictionResult(
                binary_predictions=predictions,
                anomaly_scores=scores,
                warmup_observations=0,
            )
        except Exception as exc:
            raise InferenceCompatibilityError(
                f"Isolation Forest inference input is incompatible: {exc}"
            ) from exc
