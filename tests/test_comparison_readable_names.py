from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "user_interface/monitoring_project/config_app/comparison_labels.py"
)
SPEC = spec_from_file_location("comparison_labels", MODULE_PATH)
labels = module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(labels)


class ComparisonReadableModelNameTests(unittest.TestCase):
    def test_technical_model_id_gets_a_readable_generic_name(self):
        value = labels.readable_model_name(
            {
                "model_id": "run_9db880e4243e49d8acdb405bdc60ff9e-cgnn-ac2846b2",
                "display_name": "run_9db880e4243e49d8acdb405bdc60ff9e-cgnn-ac2846b2",
                "detector_id": "cgnn",
                "detector_version": "1.0.0",
                "model_created_at": "2026-10-10T20:24:00Z",
            }
        )
        self.assertEqual(value, "CGNN v1.0.0 · trained 10 Oct 2026, 20:24 UTC")

    def test_explicit_user_facing_name_is_preserved(self):
        value = labels.readable_model_name(
            {
                "model_id": "model-one",
                "display_name": "Checkout baseline",
                "detector_id": "isolation-forest",
            }
        )
        self.assertEqual(value, "Checkout baseline")

    def test_legacy_comparison_projection_is_humanized_without_changing_identity(self):
        model_id = "run_9db880e4243e49d8acdb405bdc60ff9e-cgnn-ac2846b2"
        payload = {
            "selected_models": [
                {
                    "model_id": model_id,
                    "display_name": model_id,
                    "detector_id": "cgnn",
                    "detector_version": "1.0.0",
                    "model_created_at": "2026-10-10T20:24:00Z",
                }
            ],
            "results": [{"model_id": model_id, "display_name": model_id}],
            "f1_winner": model_id,
            "deterministic_summary": f"{model_id} has the highest F1-score.",
        }

        labels.humanize_comparison(payload)

        readable = "CGNN v1.0.0 · trained 10 Oct 2026, 20:24 UTC"
        self.assertEqual(payload["selected_models"][0]["model_id"], model_id)
        self.assertEqual(payload["selected_models"][0]["display_name"], readable)
        self.assertEqual(payload["results"][0]["display_name"], readable)
        self.assertEqual(payload["f1_winner"], readable)
        self.assertEqual(
            payload["deterministic_summary"],
            f"{readable} has the highest F1-score.",
        )

    def test_generic_detector_formatting_has_no_detector_id_mapping(self):
        self.assertEqual(
            labels.readable_model_name(
                {
                    "model_id": "model-one",
                    "detector_id": "local-outlier-factor",
                    "detector_version": "1.0.1",
                }
            ),
            "Local Outlier Factor v1.0.1 Saved Model",
        )


if __name__ == "__main__":
    unittest.main()
