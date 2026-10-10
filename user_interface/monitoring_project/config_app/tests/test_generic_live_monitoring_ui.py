from unittest.mock import Mock, patch

from django.test import SimpleTestCase
from django.urls import reverse


def model(model_id="model-ready", ready=True):
    return {
        "model_id": model_id,
        "detector_id": "isolation-forest",
        "detector_version": "1.0.0",
        "status": "available",
        "inference": {
            "status": "ready",
            "contract": "cloudsentinel.inference/v1",
        },
        "live_monitoring": (
            {"status": "ready", "sampling_interval_seconds": 60}
            if ready
            else {"status": "not_ready", "reason": "No pinned recipe"}
        ),
    }


class GenericLiveMonitoringUiTests(SimpleTestCase):
    def setUp(self):
        self.learning = Mock()
        self.learning.list_models.return_value = {
            "items": [model(), model("model-old", False)],
            "page": 1,
            "pages": 1,
        }
        self.patch = patch(
            "config_app.views.monitoring.get_learning_client",
            return_value=self.learning,
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_existing_monitoring_page_uses_saved_models_and_prefill(self):
        response = self.client.get(reverse("monitoring"), {"model_id": "model-ready"})
        self.assertContains(response, "Saved model artefact")
        self.assertContains(response, "model-ready")
        self.assertContains(response, "Live-ready")
        self.assertContains(response, "Offline only")
        self.assertNotContains(response, "Containers")

    @patch("config_app.views.monitoring.requests.post")
    def test_start_sends_only_explicit_model_and_safe_timing(self, post):
        post.return_value.status_code = 202
        post.return_value.json.return_value = {
            "status": "monitoring_started",
            "session": {"session_id": "monitor-one"},
        }
        response = self.client.post(
            reverse("monitoring"),
            {
                "model_id": "model-ready",
                "window_minutes": 10,
                "poll_interval_seconds": 30,
            },
        )
        self.assertEqual(response.status_code, 202)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model_id"], "model-ready")
        self.assertEqual(payload["window_seconds"], 600)
        self.assertNotIn("detector_id", payload)
        self.assertNotIn("prometheus_url", payload)

    @patch("config_app.views.monitoring.requests.post")
    def test_offline_only_model_cannot_start(self, post):
        response = self.client.post(
            reverse("monitoring"),
            {
                "model_id": "model-old",
                "window_minutes": 10,
                "poll_interval_seconds": 30,
            },
        )
        self.assertEqual(response.status_code, 200)
        post.assert_not_called()

    def test_dashboard_exposes_warmup_and_safe_status_language(self):
        response = self.client.get(reverse("monitoring_overview"))
        self.assertContains(response, "Warm-up observations")
        self.assertContains(response, "Buffered observations")
        self.assertContains(response, "Monitoring sessions could not be loaded")
        self.assertContains(response, "response.text()")
        self.assertNotContains(response, "API_CGNN_ANOMALY_DETECTION_URL")

    @patch("config_app.views.task_results.requests.get")
    def test_active_session_proxy_accepts_json_array(self, get):
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.return_value = [
            {"session_id": "monitor-one", "model_id": "model-ready", "status": "active"}
        ]

        response = self.client.get(reverse("fetch_active_tasks"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["session_id"], "monitor-one")
        self.assertEqual(get.call_args.kwargs["timeout"], (5, 15))

    @patch("config_app.views.task_results.requests.get")
    def test_active_session_proxy_translates_invalid_json_safely(self, get):
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.side_effect = ValueError("private upstream response")

        response = self.client.get(reverse("fetch_active_tasks"))

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["message"],
            "Monitoring status is temporarily unavailable.",
        )
        self.assertNotContains(response, "private upstream response", status_code=502)
