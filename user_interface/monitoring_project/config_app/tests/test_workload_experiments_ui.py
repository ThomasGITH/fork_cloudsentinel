from unittest.mock import Mock, patch

from django.test import SimpleTestCase
from django.urls import reverse


class WorkloadExperimentUiTests(SimpleTestCase):
    def setUp(self):
        self.catalogue = Mock()
        self.catalogue.get_workload_experiments.return_value = {
            "profiles": [
                {
                    "id": "burst",
                    "label": "Traffic burst",
                    "description": "Bounded high load",
                    "users": "80",
                    "rate": "10",
                }
            ],
            "target": {"namespace": "default", "deployment": "loadgenerator"},
            "active": None,
            "history": [
                {
                    "experiment_id": "workload-demo",
                    "profile_label": "Traffic burst",
                    "started_at": "2026-10-24T10:00:00Z",
                    "completed_at": "2026-10-24T10:05:00Z",
                    "baseline": {"users": "10", "rate": "1"},
                    "applied": {"users": "80", "rate": "10"},
                    "completion_reason": "automatic timeout",
                    "workload_context": {
                        "workload_intensity": "High",
                        "dominant_workload_characteristic": "High traffic",
                        "anomaly_scenario": "Other",
                        "anomaly_scenario_detail": "Traffic burst",
                    },
                }
            ],
        }
        patcher = patch(
            "config_app.views.workload_experiments.get_catalogue_client",
            return_value=self.catalogue,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_completed_experiment_links_exact_window_to_dataset_form(self):
        response = self.client.get(reverse("workload_experiments"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Controlled workload")
        self.assertContains(response, "Create dataset from window")
        self.assertContains(response, "incident_start_time=2026-10-24T10%3A00%3A00Z", html=False)
        self.assertContains(response, "anomaly_scenario_detail=Traffic+burst", html=False)

    def test_start_and_stop_are_proxied_server_side(self):
        start = self.client.post(
            reverse("workload_experiments"),
            {"action": "start", "profile": "burst", "duration_minutes": "5"},
        )
        self.assertEqual(start.status_code, 302)
        self.catalogue.update_workload_experiment.assert_called_with(
            {"action": "start", "profile": "burst", "duration_minutes": "5"}
        )
        stop = self.client.post(reverse("workload_experiments"), {"action": "stop"})
        self.assertEqual(stop.status_code, 302)
        self.catalogue.update_workload_experiment.assert_called_with({"action": "stop"})
