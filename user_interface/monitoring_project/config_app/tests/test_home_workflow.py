from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase
from django.urls import reverse


class GuidedWorkflowHomeTests(SimpleTestCase):
    def test_home_explains_one_ordered_lifecycle_without_contacting_services(self):
        response = self.client.get(reverse("home"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "From monitoring data to a traceable diagnosis")
        self.assertContains(response, "Start a connected experiment")
        for heading in (
            "Connect and verify the monitoring source",
            "Capture and describe a dataset",
            "Train one or more detectors",
            "Compare models in a shared context",
            "Run a selected model against live metrics",
            "Review detections and RCA evidence",
        ):
            self.assertContains(response, heading)
        self.assertContains(response, reverse("workflow_status"))
        self.assertContains(response, reverse("models_add_detector"))
        self.assertContains(response, "home_workflow.js")

    def test_status_selects_first_incomplete_step(self):
        source = {
            "state": "ready", "label": "Connected", "metric": "Source responded",
            "detail": "Prometheus accepted a live query from CloudSentinel.",
        }
        catalogue = {
            "state": "ready", "label": "Ready", "metric": "2 datasets",
            "detail": "Datasets are available.", "count": 2,
        }
        learning = {
            "models": {
                "state": "ready", "label": "Ready", "metric": "3 saved models",
                "detail": "Models are available.", "count": 3, "live_ready_count": 1,
            },
            "comparisons": {
                "state": "action", "label": "Next action", "metric": "0 completed comparisons",
                "detail": "Compare models.", "count": 0,
            },
        }
        monitoring = {
            "state": "action", "label": "Not started", "metric": "0 monitoring sessions",
            "detail": "Start monitoring.", "count": 0,
        }
        with (
            patch("config_app.views.home._prometheus_status", return_value=source),
            patch("config_app.views.home._catalogue_status", return_value=catalogue),
            patch("config_app.views.home._learning_status", return_value=learning),
            patch("config_app.views.home._monitoring_status", return_value=monitoring),
        ):
            response = self.client.get(reverse("workflow_status"))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["next_step"]["label"], "Compare saved models")
        self.assertEqual(payload["next_step"]["url"], reverse("comparison_new"))
        self.assertNotIn("PROMETHEUS_URL", str(payload))

    def test_system_connections_is_truthful_read_only_deployment_view(self):
        response = self.client.get(reverse("config"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "System connections")
        self.assertContains(response, "Why these fields are read-only")
        self.assertContains(response, "CATALOGUE_CLUSTER_DEFAULT_PROMETHEUS_URL")
        self.assertNotContains(response, "Update Config")
        self.assertEqual(self.client.post(reverse("config"), {}).status_code, 405)

    def test_status_script_uses_text_content_for_service_data(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "static"
            / "js"
            / "home_workflow.js"
        ).read_text(encoding="utf-8")

        self.assertIn("textContent", script)
        self.assertNotIn("innerHTML", script)
