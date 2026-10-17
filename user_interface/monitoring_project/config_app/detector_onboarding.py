"""Stable presentation content for the external detector onboarding page.

The examples mirror schema version 1 and the public adapter protocols. Tests
validate the manifest snippet with the real detector registry so drift is
detected during development rather than by an operator during publication.
"""

from __future__ import annotations


TEMPLATE_PLUGIN_PATH = "examples/external_plugins/template_detector"
LOF_EXAMPLE_PATH = "examples/external_plugins/local_outlier_factor"

PACKAGE_LAYOUT = """my_detector/
├── manifest.yaml
├── adapter.py
└── requirements.lock"""

COPY_TEMPLATE_COMMAND = (
    "cp -R examples/external_plugins/template_detector "
    "/absolute/path/to/my_detector"
)

PUBLISH_COMMAND = """./scripts/external_plugin.sh publish \\
  /absolute/path/to/my_detector \\
  --id my-detector \\
  --version 1.0.0 \\
  --activate"""

MANIFEST_EXAMPLE = """schema_version: 1
id: my-detector
name: My Detector
version: 1.0.0
description: Detects anomalous metric observations.

supported_modalities:
  - metrics

runtime:
  profile: python-ml-cpu/v1

capabilities:
  training:
    enabled: true
    protocol: cloudsentinel.training/v1
    input_profile: metrics-partition-v1
  inference:
    enabled: false

input_requirements:
  training:
    format: Numeric two-dimensional matrix
    feature_identity: Stable positional feature order
    finite_values_required: true
  evaluation:
    labels_required: true

training_parameters:
  sensitivity:
    type: number
    default: 0.5
    minimum: 0
    maximum: 1
    advanced: false
    description: Detector sensitivity used during training.

entry_point: adapter:MyDetectorAdapter"""

TRAINING_ADAPTER_EXAMPLE = """from detectors.contracts import (
    ProgressReporter,
    TrainingContext,
    TrainingResult,
)

class MyDetectorAdapter:
    def validate_training(self, context: TrainingContext) -> None:
        ...

    def run_training(
        self,
        context: TrainingContext,
        progress: ProgressReporter,
    ) -> TrainingResult:
        ..."""

INFERENCE_ADAPTER_EXAMPLE = """from typing import Any
from detectors.contracts import (
    InferenceContext,
    ModelLoadContext,
    PredictionResult,
)

def load_model(self, context: ModelLoadContext) -> Any:
    ...

def predict_inference(
    self,
    model: Any,
    context: InferenceContext,
) -> PredictionResult:
    ..."""

CHECKLIST = (
    "Manifest has a unique detector ID.",
    "Version identifies a new immutable release.",
    "Supported modality and input requirements are correct.",
    "Training capability uses supported protocols.",
    "Adapter implements the training contract.",
    "Inference contract is implemented if live monitoring is required.",
    "Dependencies are exactly pinned and already available in the runtime.",
    "Plugin has been validated before activation.",
    "Detector appears in the Detector Library after activation.",
)

DISCOVERY_ACTIONS = {
    "reserved_detector_id": "Choose a unique detector ID that does not match a built-in detector, then publish a new version.",
    "duplicate_detector_id": "Resolve the duplicate active detector ID and validate the intended release again.",
    "invalid_manifest": "Correct the manifest fields, then validate and publish the package again.",
    "invalid_plugin_root": "Ask the operator to correct the configured external plugin root.",
    "duplicate_plugin_root": "Ask the operator to remove the duplicate or nested plugin-root configuration.",
    "directory_unavailable": "Ask the operator to verify that the external plugin repository is mounted and readable.",
}

DISCOVERY_LABELS = {
    "reserved_detector_id": "Reserved detector ID",
    "duplicate_detector_id": "Duplicate detector ID",
    "invalid_manifest": "Invalid manifest",
    "invalid_plugin_root": "Invalid plugin repository configuration",
    "duplicate_plugin_root": "Duplicate plugin repository configuration",
    "directory_unavailable": "Plugin repository unavailable",
}


def discovery_action(code: object) -> str:
    """Return a fixed action without reflecting untrusted discovery content."""

    if isinstance(code, str) and code in DISCOVERY_ACTIONS:
        return DISCOVERY_ACTIONS[code]
    return "Review the package with the operator, validate it again, and refresh the Detector Library."


def discovery_label(code: object) -> str:
    if isinstance(code, str) and code in DISCOVERY_LABELS:
        return DISCOVERY_LABELS[code]
    return "Plugin validation issue"


def onboarding_context() -> dict[str, object]:
    return {
        "template_plugin_path": TEMPLATE_PLUGIN_PATH,
        "lof_example_path": LOF_EXAMPLE_PATH,
        "package_layout": PACKAGE_LAYOUT,
        "copy_template_command": COPY_TEMPLATE_COMMAND,
        "publish_command": PUBLISH_COMMAND,
        "manifest_example": MANIFEST_EXAMPLE,
        "training_adapter_example": TRAINING_ADAPTER_EXAMPLE,
        "inference_adapter_example": INFERENCE_ADAPTER_EXAMPLE,
        "checklist": CHECKLIST,
        "required_manifest_fields": (
            "schema_version",
            "id",
            "name",
            "version",
            "description",
            "supported_modalities",
            "capabilities",
            "input_requirements",
            "training_parameters",
            "entry_point",
        ),
        "parameter_types": ("boolean", "integer", "number", "string"),
    }
