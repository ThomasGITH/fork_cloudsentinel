from unittest.mock import Mock, patch

from django.test import Client, SimpleTestCase
from django.urls import resolve, reverse

from config_app.learning_client import LearningAdaptationClient, LearningClientError


def evaluation_dataset(labelled=True):
    return {
        "dataset_id": "ds-eval", "display_name": "CPU incident", "version": 2,
        "partition_id": "part-eval", "partition_checksum": "p" * 64,
        "modality": "metrics", "feature_order": ["cpu", "memory"],
        "feature_order_sha256": "f" * 64, "observation_count": 30,
        "workload_context": {"workload_intensity": "High", "dominant_workload_characteristic": "CPU-intensive", "anomaly_scenario": "CPU stress"},
        "ground_truth_available": labelled,
        "known_incident_windows": ([{"start_time": "2026-10-01T10:10:00Z", "end_time": "2026-10-01T10:20:00Z"}] if labelled else []),
    }


def compatible_models():
    return {
        "items": [
            {"model_id": "model-one", "display_name": "IF baseline", "detector_id": "isolation-forest", "detector_version": "1.0.0", "model_created_at": "2026-10-01", "training_dataset": {"dataset_id": "ds-train", "version": 1, "partition_id": "part-train"}, "stored_evaluation": {"available": True, "context": {"dataset_id": "ds-train", "version": 1, "partition_id": "part-train"}, "metric_sets": [{"method": "stored evaluation", "metrics": {"precision": .8, "recall": .7, "f1_score": .746}}]}, "compatible": True, "compatibility_reasons": []},
            {"model_id": "model-two", "display_name": "LOF baseline", "detector_id": "local-outlier-factor", "detector_version": "1.0.1", "model_created_at": "2026-10-01", "training_dataset": {"dataset_id": "ds-train", "version": 1}, "compatible": True, "compatibility_reasons": []},
            {"model_id": "model-bad", "display_name": "Wrong features", "detector_id": "future", "detector_version": "1", "model_created_at": "2026-10-01", "training_dataset": {}, "compatible": False, "compatibility_reasons": ["Feature identity does not match this evaluation dataset"]},
        ]
    }


def comparison(status="completed"):
    return {
        "comparison_id": "comparison-one", "name": "CPU comparison", "analysis_type": "anomaly_detection",
        "status": status, "created_at": "2026-10-01", "updated_at": "2026-10-01", "completed_at": "2026-10-01" if status == "completed" else None,
        "evaluation_dataset": evaluation_dataset(), "modality": "metrics", "ground_truth_available": True,
        "known_incident_window": evaluation_dataset()["known_incident_windows"][0],
        "deterministic_summary": "IF baseline has the highest F1-score.",
        "selected_models": [
            {"model_id": "model-one", "display_name": "IF baseline", "detector_id": "isolation-forest", "detector_version": "1.0.0", "model_created_at": "2026-10-01", "training_dataset": {"dataset_id": "ds-train", "version": 1}, "artifact_identity": {"artifact_id": "artifact-one", "artifact_manifest_sha256": "a" * 64}, "live_monitoring": {"status": "ready"}},
            {"model_id": "model-two", "display_name": "LOF baseline", "detector_id": "local-outlier-factor", "detector_version": "1.0.1", "model_created_at": "2026-10-01", "training_dataset": {"dataset_id": "ds-train", "version": 1}, "artifact_identity": {"artifact_id": "artifact-two", "artifact_manifest_sha256": "b" * 64}},
        ],
        "results": [
            {"model_id": "model-one", "display_name": "IF baseline", "detector_id": "isolation-forest", "detector_version": "1.0.0", "status": "completed", "runtime_ms": 12.5, "ground_truth_available": True, "metrics": {"precision": .8, "recall": .7, "f1_score": .746}, "lead_time_status": "before_or_at_incident_start", "lead_time_minutes": 2.0, "timeline": [{"timestamp": "2026-10-01T10:00:00Z", "prediction": 0, "score": .1}]},
            {"model_id": "model-two", "display_name": "LOF baseline", "detector_id": "local-outlier-factor", "detector_version": "1.0.1", "status": "failed", "safe_error_summary": "Evaluation failed safely", "ground_truth_available": True, "metrics": {}, "timeline": []},
        ],
    }


def robustness_payload():
    context = {
        "dataset_id": "ds-eval", "display_name": "CPU incident", "version": 2,
        "partition_id": "part-eval", "partition_checksum": "p" * 64,
        "workload_context": {"workload_intensity": "High"},
        "scenario": "CPU stress", "dominant_characteristic": "CPU-intensive",
        "ground_truth_available": True, "known_incident_windows": [],
    }
    artifact = {"artifact_id": "artifact-one", "artifact_manifest_sha256": "a" * 64}
    row = {
        "comparison_id": "comparison-one", "comparison_name": "CPU comparison",
        "comparison_date": "2026-10-01", "model_id": "model-one",
        "display_name": "IF baseline", "detector_id": "isolation-forest",
        "detector_version": "1.0.0", "artifact_identity": artifact,
        "context": context, "ground_truth_available": True,
        "metrics": {"precision": .8, "recall": .7, "f1_score": .746},
        "runtime_ms": 12.5, "lead_time_seconds": 120.0,
        "lead_time_status": "before_or_at_incident_start",
    }
    statistics = {
        "f1_score": {"count": 1, "median": .746, "minimum": .746, "maximum": .746, "range": 0.0, "standard_deviation": None},
        "precision": {"count": 1, "median": .8, "minimum": .8, "maximum": .8},
        "recall": {"count": 1, "median": .7, "minimum": .7, "maximum": .7},
        "runtime_ms": {"count": 1, "median": 12.5, "minimum": 12.5, "maximum": 12.5},
        "lead_time_seconds": {"count": 1, "median": 120.0, "minimum": 120.0, "maximum": 120.0},
    }
    model = {
        "model_id": "model-one", "display_name": "IF baseline",
        "detector_id": "isolation-forest", "detector_version": "1.0.0",
        "artifact_identity": artifact, "context_count": 1,
        "labelled_context_count": 1, "sufficient_contexts": False,
        "statistics": statistics, "contexts": [row],
        "related_comparison_ids": ["comparison-one"],
    }
    return {
        "comparison_id": "comparison-one",
        "aggregation_version": "cloudsentinel.comparison-robustness/v1",
        "models": [model], "shared_context_matrix": [],
        "coverage_warnings": ["Showing the latest completed result per evaluation context."],
        "deduplicated_repeat_count": 1, "truncated": False,
    }


class ComparisonUiTests(SimpleTestCase):
    def setUp(self):
        self.learning = Mock()
        self.learning.list_comparisons.return_value = {"items": [{**comparison(), "model_count": 2, "f1_winner": "IF baseline"}], "page": 1, "page_size": 20, "total": 1, "pages": 1}
        self.learning.list_evaluation_datasets.return_value = {"items": [evaluation_dataset(), evaluation_dataset(False)]}
        self.learning.list_compatible_models.return_value = compatible_models()
        self.learning.create_comparison.return_value = {"comparison_id": "comparison-created", "status": "queued"}
        self.learning.get_comparison.return_value = comparison()
        self.learning.get_comparison_status.return_value = {"comparison_id": "comparison-one", "status": "running", "results": []}
        self.learning.get_comparison_robustness.return_value = robustness_payload()
        patcher = patch("config_app.views.comparison.get_learning_client", return_value=self.learning)
        self.addCleanup(patcher.stop); patcher.start()

    def test_navigation_routes_overview_filters_and_no_rca(self):
        response = self.client.get(reverse("comparison_overview"), {"status": "completed", "workload": "High", "page": 1})
        self.assertContains(response, "Past comparisons")
        self.assertContains(response, "+ New comparison")
        self.assertContains(response, "F1 winner")
        self.assertContains(response, 'aria-current="page"', html=False)
        self.assertNotContains(response, "Root Cause Analysis")
        self.assertNotContains(response, "View RCA")
        self.learning.list_comparisons.assert_called_once_with({"status": "completed", "workload": "High", "page": "1"})
        self.assertEqual(resolve(reverse("comparison_new")).url_name, "comparison_new")
        self.assertContains(self.client.get(reverse("home")), reverse("comparison_overview"))

    def test_overview_empty_and_safe_error(self):
        self.learning.list_comparisons.return_value = {"items": [], "page": 1, "pages": 0, "total": 0}
        self.assertContains(self.client.get(reverse("comparison_overview")), "No comparisons found")
        self.learning.list_comparisons.side_effect = LearningClientError("failure at http://secret and /app/private", retryable=True)
        response = self.client.get(reverse("comparison_overview"))
        self.assertContains(response, "Retry")
        self.assertNotContains(response, "http://secret")
        self.assertNotContains(response, "/app/private")

    def test_related_comparison_filter_is_preserved_and_explained(self):
        query = {
            "model_id": "model-one",
            "artifact_id": "artifact-one",
            "artifact_manifest_sha256": "a" * 64,
        }
        response = self.client.get(reverse("comparison_overview"), query)
        self.assertContains(response, "Showing comparisons for the exact Saved Model artefact")
        self.learning.list_comparisons.assert_called_once_with(query)

    def _step(self, target):
        response = self.client.get(reverse("comparison_new"))
        if target == 1: return response
        response = self.client.post(reverse("comparison_new"), {"step": 1, "wizard_state": response.context["state_token"], "dataset_selection": "ds-eval|2"})
        if target == 2: return response
        response = self.client.post(reverse("comparison_new"), {"step": 2, "wizard_state": response.context["state_token"], "model_ids": ["model-one", "model-two"]})
        return response

    def test_wizard_dataset_warning_and_incompatible_model(self):
        first = self._step(1)
        self.assertContains(first, "Ground truth unavailable")
        second = self._step(2)
        self.assertContains(second, "Feature identity does not match")
        self.assertContains(second, 'value="model-bad" disabled', html=False)
        self.assertContains(second, "Stored post-training evaluation")
        self.assertContains(second, "F1 0.746")
        self.assertContains(second, "partition part-train")
        self.assertContains(second, "models are not ranked")
        self.assertContains(second, "Unavailable — no evaluation metrics were stored")
        rejected = self.client.post(reverse("comparison_new"), {"step": 2, "wizard_state": second.context["state_token"], "model_ids": ["model-one", "model-bad"]})
        self.assertEqual(rejected.status_code, 400)
        self.learning.create_comparison.assert_not_called()

    def test_wizard_requires_two_and_creates_payload(self):
        second = self._step(2)
        too_few = self.client.post(reverse("comparison_new"), {"step": 2, "wizard_state": second.context["state_token"], "model_ids": ["model-one"]})
        self.assertEqual(too_few.status_code, 400)
        review = self._step(3)
        response = self.client.post(reverse("comparison_new"), {"step": 3, "wizard_state": review.context["state_token"], "name": "CPU comparison"})
        self.assertRedirects(response, "/comparison/comparison-created/", fetch_redirect_response=False)
        payload = self.learning.create_comparison.call_args.args[0]
        self.assertEqual(payload["model_ids"], ["model-one", "model-two"])
        self.assertEqual(payload["evaluation_dataset"], {"dataset_id": "ds-eval", "version": 2, "partition_id": "part-eval"})
        self.assertTrue(payload["client_request_id"])

    def test_detail_tabs_grounded_results_and_polling(self):
        overview = self.client.get(reverse("comparison_detail", args=["comparison-one"]))
        self.assertContains(overview, "0.746")
        self.assertContains(overview, "Detection lead time")
        self.assertNotContains(overview, "Suggested investigation")
        timeline = self.client.get(reverse("comparison_detail", args=["comparison-one"]), {"tab": "timeline"})
        self.assertContains(timeline, "comparison-timeline")
        self.assertContains(timeline, "data-timeline-error")
        self.assertContains(timeline, "comparison-incident-windows")
        self.assertContains(timeline, "Shaded areas show known ground-truth incident windows")
        self.assertContains(timeline, '"timestamp": "2026-10-01T10:00:00Z"')
        models = self.client.get(reverse("comparison_detail", args=["comparison-one"]), {"tab": "models"})
        self.assertContains(models, reverse("models_saved_detail", args=["model-one"]))
        self.assertContains(models, "Use for live monitoring")
        self.assertContains(models, "/monitoring/?model_id=model-one")
        robustness = self.client.get(reverse("comparison_detail", args=["comparison-one"]), {"tab": "robustness"})
        self.assertContains(robustness, "Robustness across workload contexts")
        self.assertContains(robustness, "same saved model artefact")
        self.assertContains(robustness, "One evaluation context found")
        self.assertContains(robustness, "View related comparisons")
        self.assertNotContains(robustness, "robustness score")
        filtered = self.client.get(
            reverse("comparison_detail", args=["comparison-one"]),
            {"tab": "robustness", "workload": "High", "labelled_only": "true"},
        )
        self.assertContains(filtered, 'value="High" selected', html=False)
        self.learning.get_comparison_robustness.assert_called_with(
            "comparison-one", {"workload": "High", "labelled_only": "true"}
        )
        status = self.client.get(reverse("comparison_status", args=["comparison-one"]))
        self.assertEqual(status.json()["status"], "running")
        self.assertNotContains(status, "task_id")

    def test_mutating_wizard_requires_csrf(self):
        csrf = Client(enforce_csrf_checks=True)
        self.assertEqual(csrf.post(reverse("comparison_new"), {}).status_code, 403)


class ComparisonLearningClientTests(SimpleTestCase):
    def test_comparison_contract_paths(self):
        response = Mock(status_code=200, headers={}); response.json.return_value = {"items": []}
        transport = Mock(); transport.request.return_value = response
        client = LearningAdaptationClient("http://learning.internal", transport=transport)
        client.list_comparisons({"page": 2}); client.get_comparison("comparison-one")
        client.create_comparison({"name": "test"}); client.get_comparison_status("comparison-one")
        client.get_comparison_robustness("comparison-one", {"labelled_only": "true"})
        client.list_evaluation_datasets(); client.list_compatible_models("ds-one", 2, "part-one")
        paths = [call.args[1] for call in transport.request.call_args_list]
        self.assertEqual(paths, [
            "http://learning.internal/api/comparisons",
            "http://learning.internal/api/comparisons/comparison-one",
            "http://learning.internal/api/comparisons",
            "http://learning.internal/api/comparisons/comparison-one/status",
            "http://learning.internal/api/comparisons/comparison-one/robustness",
            "http://learning.internal/api/comparisons/evaluation-datasets",
            "http://learning.internal/api/comparisons/compatible-models",
        ])
