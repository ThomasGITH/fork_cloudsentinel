"""Shared interface for anomaly detector adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


@runtime_checkable
class DetectorAdapter(Protocol):
    """Minimal operations required by the current detector integration."""

    def train(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        ...


class DetectorTrainingError(RuntimeError):
    """Base class for failures produced by a training adapter."""


class DetectorCompatibilityError(DetectorTrainingError, ValueError):
    """The selected data or parameters are incompatible with the detector."""


class DetectorExecutionError(DetectorTrainingError):
    """Detector training, evaluation, or artifact creation failed."""


class DetectorPromotionError(DetectorTrainingError):
    """A trained artifact could not be promoted as required by the adapter."""


@dataclass(frozen=True)
class TrainingData:
    """Private references to one immutable metrics-partition-v1 snapshot."""

    train_path: Path
    test_path: Path
    labels_path: Path | None
    train_observations: int
    test_observations: int
    feature_count: int


@dataclass(frozen=True)
class ArtifactResult:
    """Private artifact evidence plus its safe logical description."""

    directory: Path
    required_files: tuple[str, ...]
    format: str
    safe_reference: str | None = None


@dataclass(frozen=True)
class PromotionResult:
    status: str
    promoted_at: str | None = None
    safe_reference: str | None = None


@dataclass(frozen=True)
class TrainingResult:
    status: str
    artifact: ArtifactResult
    evaluation: Mapping[str, Any] = field(default_factory=dict)
    promotion: PromotionResult = field(
        default_factory=lambda: PromotionResult(status="not_applicable")
    )
    model_metadata: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    cleanup: str = "keep"


@dataclass(frozen=True)
class TrainingContext:
    run_id: str
    child_id: str
    model_id: str
    detector_id: str
    detector_version: str
    parameters: Mapping[str, Any]
    dataset: Mapping[str, Any]
    snapshot: Mapping[str, Any]
    feature_identity: Mapping[str, Any]
    data: TrainingData
    artifact_workspace: Path
    suggested_artifact_directory: Path
    model_record_context: Mapping[str, Any]


class ProgressReporter:
    """Bounded adapter-facing progress interface."""

    ALLOWED_STATES = {
        "INITIATING",
        "VALIDATING",
        "TRAINING",
        "EVALUATING",
        "PROMOTING",
        "COMPLETED",
    }

    def __init__(self, callback: Callable[[str, Any], None]):
        self._callback = callback

    def report(self, state: str, detail: Any) -> None:
        if state not in self.ALLOWED_STATES:
            raise ValueError(f"unsupported training progress state: {state!r}")
        if isinstance(detail, str):
            safe_detail: Any = detail[:1000]
        elif detail is None or isinstance(detail, (int, float, bool)):
            safe_detail = detail
        elif isinstance(detail, dict):
            safe_detail = {
                str(key)[:100]: value[:1000] if isinstance(value, str) else value
                for key, value in list(detail.items())[:20]
                if value is None or isinstance(value, (str, int, float, bool))
            }
        else:
            safe_detail = str(detail)[:1000]
        self._callback(state, safe_detail)


@runtime_checkable
class TrainableDetectorAdapter(Protocol):
    """Training protocol used by the manifest-driven TrainingRun executor."""

    def validate_training(self, context: TrainingContext) -> None:
        ...

    def run_training(
        self, context: TrainingContext, progress: ProgressReporter
    ) -> TrainingResult:
        ...
