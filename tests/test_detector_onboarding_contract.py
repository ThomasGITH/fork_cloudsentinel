from pathlib import Path
import sys
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = ROOT / "user_interface" / "monitoring_project"
if str(UI_ROOT) not in sys.path:
    sys.path.insert(0, str(UI_ROOT))

from config_app.detector_onboarding import (  # noqa: E402
    MANIFEST_EXAMPLE,
    PUBLISH_COMMAND,
    TRAINING_ADAPTER_EXAMPLE,
)
from detectors import registry  # noqa: E402


class DetectorOnboardingContractTests(unittest.TestCase):
    def test_manifest_example_is_accepted_by_current_registry(self):
        manifest = yaml.safe_load(MANIFEST_EXAMPLE)
        validated = registry.validate_manifest(
            manifest, Path("onboarding-manifest.yaml")
        )
        self.assertEqual(validated["schema_version"], registry.SUPPORTED_SCHEMA_VERSION)
        self.assertTrue(registry.REQUIRED_FIELDS.issubset(validated))
        self.assertEqual(
            validated["capabilities"]["training"]["protocol"],
            next(iter(registry.SUPPORTED_TRAINING_PROTOCOLS)),
        )
        self.assertEqual(
            validated["capabilities"]["training"]["input_profile"],
            next(iter(registry.SUPPORTED_INPUT_PROFILES)),
        )
        self.assertTrue(
            set(item["type"] for item in validated["training_parameters"].values())
            .issubset(registry.PARAMETER_TYPES)
        )

    def test_adapter_and_publish_examples_match_public_contract(self):
        self.assertIn("validate_training(self, context: TrainingContext)", TRAINING_ADAPTER_EXAMPLE)
        self.assertIn("context: TrainingContext,", TRAINING_ADAPTER_EXAMPLE)
        self.assertIn("progress: ProgressReporter,", TRAINING_ADAPTER_EXAMPLE)
        self.assertIn(") -> TrainingResult:", TRAINING_ADAPTER_EXAMPLE)
        self.assertIn("./scripts/external_plugin.sh publish", PUBLISH_COMMAND)
        self.assertIn("/absolute/path/to/my_detector", PUBLISH_COMMAND)
        self.assertIn("--activate", PUBLISH_COMMAND)


if __name__ == "__main__":
    unittest.main()
