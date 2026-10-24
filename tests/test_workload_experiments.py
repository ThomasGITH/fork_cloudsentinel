import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flask import Flask

from data_ingestion.workload_experiments import (
    configure_workload_experiment_defaults,
    create_workload_experiments_blueprint,
)


class WorkloadExperimentApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.app = Flask(__name__)
        configure_workload_experiment_defaults(self.app)
        self.app.config.update(
            EXPERIMENT_OPERATOR_KEY="experiment-secret",
            WORKLOAD_EXPERIMENT_STORAGE_PATH=str(Path(self.temporary.name) / "experiments.json"),
        )
        self.app.register_blueprint(create_workload_experiments_blueprint())
        self.client = self.app.test_client()
        self.headers = {"X-Experiment-Operator-Key": "experiment-secret"}

    def test_access_is_operator_gated(self):
        response = self.client.get("/experiments/workload")
        self.assertEqual(response.status_code, 403)

    @patch("data_ingestion.workload_experiments._schedule_restore")
    @patch("data_ingestion.workload_experiments._patch_load")
    @patch("data_ingestion.workload_experiments._deployment_environment")
    def test_start_is_bounded_and_records_reversible_context(self, environment, patch_load, schedule):
        environment.return_value = {"USERS": "10", "RATE": "1"}
        response = self.client.post(
            "/experiments/workload",
            headers=self.headers,
            json={"action": "start", "profile": "burst", "duration_minutes": 5},
        )
        self.assertEqual(response.status_code, 202)
        active = response.get_json()["active"]
        self.assertEqual(active["baseline"], {"users": "10", "rate": "1"})
        self.assertEqual(active["applied"], {"users": "80", "rate": "10"})
        self.assertEqual(active["workload_context"]["dominant_workload_characteristic"], "High traffic")
        patch_load.assert_called_once()
        schedule.assert_called_once()

        duplicate = self.client.post(
            "/experiments/workload",
            headers=self.headers,
            json={"action": "start", "profile": "elevated", "duration_minutes": 5},
        )
        self.assertEqual(duplicate.status_code, 409)

    @patch("data_ingestion.workload_experiments._schedule_restore")
    @patch("data_ingestion.workload_experiments._patch_load")
    @patch("data_ingestion.workload_experiments._deployment_environment")
    def test_stop_restores_baseline_and_preserves_completed_window(self, environment, patch_load, _schedule):
        environment.return_value = {"USERS": "10", "RATE": "1"}
        self.client.post(
            "/experiments/workload",
            headers=self.headers,
            json={"action": "start", "profile": "elevated", "duration_minutes": 2},
        )
        response = self.client.post(
            "/experiments/workload", headers=self.headers, json={"action": "stop"}
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsNone(payload["active"])
        self.assertEqual(payload["history"][0]["status"], "completed")
        self.assertTrue(payload["history"][0]["completed_at"])
        self.assertEqual(patch_load.call_args_list[-1].args[1], {"users": "10", "rate": "1"})

    def test_duration_outside_safety_bounds_is_rejected(self):
        response = self.client.post(
            "/experiments/workload",
            headers=self.headers,
            json={"action": "start", "profile": "burst", "duration_minutes": 30},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
