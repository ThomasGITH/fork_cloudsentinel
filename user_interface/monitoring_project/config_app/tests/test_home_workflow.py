from datetime import datetime, timedelta, timezone
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

    def test_completed_history_suggests_a_new_experiment(self):
        status = {
            "source": {"state": "ready"},
            "catalogue": {"state": "ready", "count": 3},
            "models": {"state": "ready", "count": 4, "live_ready_count": 2},
            "comparisons": {"state": "ready", "count": 2},
            "monitoring": {"state": "ready", "count": 5, "active_count": 0},
        }

        from config_app.views.home import _next_step

        result = _next_step(status)
        self.assertEqual(result["label"], "Start a new experiment")
        self.assertEqual(result["url"], reverse("telemetry_explorer"))

    def test_active_monitoring_remains_the_current_action(self):
        status = {
            "source": {"state": "ready"},
            "catalogue": {"state": "ready", "count": 3},
            "models": {"state": "ready", "count": 4, "live_ready_count": 2},
            "comparisons": {"state": "ready", "count": 2},
            "monitoring": {"state": "active", "count": 5, "active_count": 1},
        }

        from config_app.views.home import _next_step

        result = _next_step(status)
        self.assertEqual(result["label"], "View active monitoring")
        self.assertEqual(result["url"], reverse("monitoring_overview"))

    def test_stale_active_monitoring_is_not_treated_as_current_work(self):
        from config_app.views.home import _is_recent_active_session

        now = datetime.now(timezone.utc)
        stale = {
            "status": "active",
            "updated_at": (now - timedelta(hours=2)).isoformat(),
            "poll_interval_seconds": 300,
        }
        recent = {
            "status": "active",
            "updated_at": (now - timedelta(minutes=2)).isoformat(),
            "poll_interval_seconds": 300,
        }
        self.assertFalse(_is_recent_active_session(stale, now))
        self.assertTrue(_is_recent_active_session(recent, now))

    @patch("config_app.views.update_config.get_catalogue_client")
    def test_monitored_environment_is_editable_and_separates_diagnostics(self, get_client):
        get_client.return_value.get_monitored_environment.return_value = {
            "source_id": "cluster-default",
            "prometheus_url": "http://prometheus.example:9090",
            "default_namespace": "online-boutique",
        }
        response = self.client.get(reverse("config"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Monitored environment")
        self.assertContains(response, "Test connection")
        self.assertContains(response, "Deployment diagnostics")
        self.assertContains(response, "online-boutique")
        self.assertNotContains(response, "Why these fields are read-only")

        saved = self.client.post(reverse("config"), {
            "prometheus_url": "http://prometheus.example:9090",
            "default_namespace": "production",
            "action": "save",
        })
        self.assertEqual(saved.status_code, 302)
        get_client.return_value.update_monitored_environment.assert_called_once_with({
            "prometheus_url": "http://prometheus.example:9090",
            "default_namespace": "production",
            "source_id": "cluster-default",
        })

    def test_status_script_uses_text_content_for_service_data(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "static"
            / "js"
            / "home_workflow.js"
        ).read_text(encoding="utf-8")

        self.assertIn("textContent", script)
        self.assertNotIn("innerHTML", script)
