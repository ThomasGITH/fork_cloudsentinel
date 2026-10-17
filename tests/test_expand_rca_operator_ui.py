from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "user_interface/monitoring_project/config_app"
PRESENTATION = UI / "static/js/monitoring_rca_presentation.js"
HISTORY = UI / "static/js/monitoring_history.js"
SETUP = UI / "templates/config_app/monitoring/monitoring_setup.html"


class ExpandRcaOperatorUiTests(unittest.TestCase):
    def setUp(self):
        self.presentation = PRESENTATION.read_text()
        self.history = HISTORY.read_text()
        self.setup = SETUP.read_text()

    def test_setup_describes_both_methods_and_separates_rca_from_detection(self):
        self.assertIn("About the RCA methods", self.setup)
        self.assertIn("independent of the selected anomaly-detection model", self.setup)
        self.assertIn("method.input_type", self.setup)
        self.assertIn("method.output_type", self.setup)
        self.assertIn("method.reason", self.setup)

    def test_presentation_supports_causal_baro_and_historical_causal_default(self):
        self.assertIn('const CAUSAL = "cloudsentinel.causal-rca/legacy-v1"', self.presentation)
        self.assertIn('const BARO = "cloudsentinel.baro-robust-scorer/v1"', self.presentation)
        self.assertIn("return CAUSAL;", self.presentation)
        self.assertIn("result.change_point_evidence || run.change_point_evidence", self.presentation)

    def test_operator_summary_and_baro_evidence_are_grounded(self):
        for text in (
            "Operator summary",
            "Analysis window",
            "Most strongly represented entity",
            "Strongest metric",
            "Analysed observations",
            "Analysed features",
            "Top suspects",
            "BARO change evidence",
            "Detector change point",
            "Baseline period",
            "Anomalous period",
        ):
            self.assertIn(text, self.history)

    def test_missing_entity_and_empty_failed_skipped_rankings_have_honest_states(self):
        self.assertIn("Unavailable — no entity metadata", self.presentation)
        self.assertIn("This completed run contains no ranking rows.", self.history)
        self.assertIn("No ranking is available for this run.", self.history)
        self.assertIn('["failed","skipped"]', self.history)

    def test_scores_are_relative_and_not_probabilities(self):
        self.assertIn("relativeScore", self.presentation)
        self.assertIn("within this run only", self.history)
        self.assertIn("Scores are not probabilities", self.history)
        self.assertIn("must not be compared between RCA methods", self.history)

    def test_result_data_uses_text_content_and_never_inner_html(self):
        self.assertIn("element.textContent = String(text)", self.history)
        self.assertNotIn("innerHTML", self.history)
        self.assertNotIn("innerHTML", self.presentation)
        # Strange names remain plain strings until the renderer assigns textContent.
        self.assertIn("String(value)", self.presentation)
        self.assertIn("item.feature_id||item.feature_name", self.history)

    def test_causal_graph_full_ranking_provenance_and_warning_remain(self):
        self.assertIn("Stored CausalRCA dependency graph", self.history)
        self.assertIn("Full technical ranking", self.history)
        self.assertIn("RCA provenance", self.history)
        self.assertIn(
            "This result is diagnostic evidence, not proof of the actual root cause.",
            self.history,
        )


if __name__ == "__main__":
    unittest.main()
