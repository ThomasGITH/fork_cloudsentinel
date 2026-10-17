import math

from django.test import SimpleTestCase

from config_app.comparison_decision import build_decision_support


def _result(model_id, *, precision, recall, f1, runtime, lead, status="completed"):
    return {
        "model_id": model_id,
        "display_name": model_id.upper(),
        "detector_id": f"detector-{model_id}",
        "detector_version": "1.0.0",
        "status": status,
        "ground_truth_available": True,
        "metrics": {"precision": precision, "recall": recall, "f1_score": f1},
        "runtime_ms": runtime,
        "lead_time_status": "before_or_at_incident_start",
        "lead_time_seconds": lead,
    }


class ComparisonDecisionSupportTests(SimpleTestCase):
    def _comparison(self):
        return {
            "selected_models": [
                {"model_id": "a", "live_monitoring": {"status": "ready"}},
                {"model_id": "b", "live_monitoring": {"status": "not_ready", "reason": "Recipe unavailable"}},
                {"model_id": "c", "live_monitoring": {"status": "ready"}},
                {"model_id": "failed", "live_monitoring": {"status": "ready"}},
            ],
            "results": [
                _result("a", precision=.92, recall=.72, f1=.81, runtime=20.0, lead=60.0),
                _result("b", precision=.70, recall=.96, f1=.80, runtime=30.0, lead=180.0),
                _result("c", precision=.95, recall=.60, f1=.74, runtime=8.0, lead=120.0),
                _result("failed", precision=1.0, recall=1.0, f1=1.0, runtime=1.0, lead=999.0, status="failed"),
            ],
        }

    def test_each_goal_uses_its_documented_metric_and_direction(self):
        support = build_decision_support(self._comparison())
        goals = {goal["id"]: goal for goal in support["goals"]}
        self.assertEqual(goals["balanced_detection"]["best_models"][0]["model_id"], "a")
        self.assertEqual(goals["catch_incidents"]["best_models"][0]["model_id"], "b")
        self.assertEqual(goals["minimize_false_alarms"]["best_models"][0]["model_id"], "c")
        self.assertEqual(goals["earliest_detection"]["best_models"][0]["model_id"], "b")
        self.assertEqual(goals["minimize_compute_cost"]["best_models"][0]["model_id"], "c")
        self.assertEqual(goals["minimize_compute_cost"]["difference_to_second"], 12.0)
        self.assertEqual(goals["balanced_detection"]["best_models"][0]["live_monitoring"]["status"], "ready")

    def test_ties_show_every_best_model_without_arbitrary_choice(self):
        comparison = self._comparison()
        comparison["results"][1]["metrics"]["f1_score"] = .81
        goal = build_decision_support(comparison)["selected_goal"]
        self.assertTrue(goal["tie"])
        self.assertEqual([item["model_id"] for item in goal["best_models"]], ["a", "b"])
        self.assertAlmostEqual(goal["difference_to_second"], .07)

    def test_missing_invalid_nonfinite_and_failed_metrics_are_excluded(self):
        comparison = self._comparison()
        comparison["results"] = [
            _result("a", precision=None, recall=None, f1=math.nan, runtime=-1, lead=-1),
            _result("failed", precision=1, recall=1, f1=1, runtime=1, lead=1, status="failed"),
        ]
        support = build_decision_support(comparison, "unknown-goal")
        self.assertEqual(support["selected_goal_id"], "balanced_detection")
        self.assertTrue(all(goal["status"] == "not_enough_data" for goal in support["goals"]))

    def test_unlabelled_and_after_incident_values_are_not_ranked(self):
        comparison = self._comparison()
        comparison["results"][0]["ground_truth_available"] = False
        comparison["results"][0]["lead_time_status"] = "after_incident_start"
        goals = {goal["id"]: goal for goal in build_decision_support(comparison)["goals"]}
        self.assertNotEqual(goals["balanced_detection"]["best_models"][0]["model_id"], "a")
        self.assertNotEqual(goals["earliest_detection"]["best_models"][0]["model_id"], "a")
