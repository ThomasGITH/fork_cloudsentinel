"""Starting point for an external CloudSentinel detector adapter.

Replace the two method bodies with detector-specific validation, training,
evaluation and artifact creation. Do not train or perform filesystem work when
this module is imported.
"""

from __future__ import annotations

from detectors.contracts import (
    DetectorCompatibilityError,
    DetectorExecutionError,
    ProgressReporter,
    TrainingContext,
    TrainingResult,
)


class TemplateDetectorAdapter:
    """Minimal cloudsentinel.training/v1 adapter skeleton."""

    def validate_training(self, context: TrainingContext) -> None:
        """Validate parameters and dataset-dependent detector requirements.

        Generic manifest validation has already checked parameter types and
        simple bounds. Check relationships such as lookback versus row count,
        finite-value requirements and detector-specific feature constraints
        here. Raise DetectorCompatibilityError with a safe user-facing reason.
        """

        raise DetectorCompatibilityError(
            "Template detector is not implemented; complete validate_training first"
        )

    def run_training(
        self, context: TrainingContext, progress: ProgressReporter
    ) -> TrainingResult:
        """Train, evaluate and return a validated TrainingResult.

        Write artifacts only below context.suggested_artifact_directory. Report
        bounded progress through progress.report(). Return ArtifactResult paths
        privately through TrainingResult; never expose filesystem paths as
        public model metadata.
        """

        raise DetectorExecutionError(
            "Template detector is not implemented; complete run_training first"
        )


# To enable capabilities.inference, also implement:
#
#   def load_model(self, context: ModelLoadContext) -> Any: ...
#   def predict_inference(
#       self, model: Any, context: InferenceContext
#   ) -> PredictionResult: ...
#
# Use CloudSentinel semantics: binary 0 = normal, 1 = anomaly, and larger
# anomaly_scores mean more anomalous. Keep the loaded model opaque to the host.
