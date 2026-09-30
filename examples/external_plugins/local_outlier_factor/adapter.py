"""CloudSentinel training adapter for the external Local Outlier Factor plugin."""

from __future__ import annotations

from detectors.contracts import (
    ArtifactResult,
    DetectorCompatibilityError,
    DetectorExecutionError,
    ProgressReporter,
    PromotionResult,
    TrainingContext,
    TrainingResult,
)

from . import implementation


class LocalOutlierFactorAdapter:
    """Train and evaluate LOF through the generic detector protocol."""

    def _validated_inputs(self, context: TrainingContext):
        try:
            parameters = implementation.validate_parameters(dict(context.parameters))
            train, test, labels = implementation.load_training_data(
                context.data.train_path,
                context.data.test_path,
                context.data.labels_path,
            )
            implementation.validate_training_compatibility(
                train,
                test,
                labels,
                parameters,
                expected_features=context.data.feature_count,
                feature_order=context.feature_identity.get("feature_order", []),
            )
            return train, test, labels, parameters
        except implementation.LocalOutlierFactorInputError as exc:
            raise DetectorCompatibilityError(str(exc)) from exc

    def validate_training(self, context: TrainingContext) -> None:
        self._validated_inputs(context)

    def run_training(
        self, context: TrainingContext, progress: ProgressReporter
    ) -> TrainingResult:
        try:
            train, test, labels, parameters = self._validated_inputs(context)
            artifact_dir = context.suggested_artifact_directory
            progress.report("TRAINING", "Training Local Outlier Factor model")
            metadata = implementation.train_and_persist(
                train,
                artifact_dir,
                parameters=parameters,
                model_id=context.model_id,
                feature_identity=dict(context.feature_identity),
                train_observations=context.data.train_observations,
                test_observations=context.data.test_observations,
                label_source=implementation.label_source(context.snapshot),
            )
            progress.report("EVALUATING", "Evaluating Local Outlier Factor model")
            evaluation = implementation.evaluate_and_persist(
                test, labels, artifact_dir
            )
            return TrainingResult(
                status="completed",
                artifact=ArtifactResult(
                    directory=artifact_dir,
                    required_files=(
                        implementation.MODEL_FILENAME,
                        implementation.METADATA_FILENAME,
                        implementation.EVALUATION_FILENAME,
                    ),
                    format="joblib",
                    safe_reference=context.model_id,
                ),
                evaluation=evaluation,
                promotion=PromotionResult(status="not_applicable"),
                model_metadata={
                    "artifact_format": metadata["artifact_format"],
                    "sklearn_version": metadata["sklearn_version"],
                    "n_features": metadata["n_features"],
                    "train_observations": metadata["train_observations"],
                    "test_observations": metadata["test_observations"],
                    "label_source": metadata["label_source"],
                    "score_semantics": metadata["score_semantics"],
                },
                cleanup="keep",
            )
        except DetectorCompatibilityError:
            raise
        except implementation.LocalOutlierFactorInputError as exc:
            raise DetectorCompatibilityError(str(exc)) from exc
        except Exception as exc:
            raise DetectorExecutionError(
                f"Local Outlier Factor training failed: {exc}"
            ) from exc
