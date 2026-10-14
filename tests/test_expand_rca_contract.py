import unittest
import importlib
import sys
import types
from unittest.mock import patch
from unittest.mock import Mock

from data_ingestion.monitoring_history import MonitoringHistoryStore
from data_ingestion.monitoring_rca import (
    DEFAULT_RCA_METHOD,
    RCA_METHOD_BARO,
    RCA_METHOD_CAUSAL_LEGACY,
    MonitoringRcaError,
    initial_rca_state,
    rca_method_id,
    safe_method_capabilities,
    stable_run_id,
    validate_rca_configuration,
)


class ExpandRcaContractTests(unittest.TestCase):
    def test_legacy_configuration_defaults_to_causal_rca(self):
        configuration = validate_rca_configuration({"enabled": True})
        self.assertEqual(configuration["method"], RCA_METHOD_CAUSAL_LEGACY)
        self.assertEqual(rca_method_id({}), DEFAULT_RCA_METHOD)
        self.assertEqual(
            rca_method_id({"algorithm": {"id": RCA_METHOD_CAUSAL_LEGACY}}),
            RCA_METHOD_CAUSAL_LEGACY,
        )

    def test_explicit_causal_and_unknown_method_validation(self):
        configuration = validate_rca_configuration(
            {"enabled": True, "method": RCA_METHOD_CAUSAL_LEGACY}
        )
        self.assertEqual(configuration["method"], RCA_METHOD_CAUSAL_LEGACY)
        with self.assertRaises(MonitoringRcaError) as raised:
            validate_rca_configuration({"enabled": True, "method": "unknown/method"})
        self.assertEqual(raised.exception.reason_code, "unknown_rca_method")

    def test_baro_is_advertised_but_unavailable(self):
        methods = safe_method_capabilities(
            {"status": "ready"},
            {
                "config_sha256": "a" * 64,
                "methods": [
                    {"id": RCA_METHOD_CAUSAL_LEGACY, "status": "available"},
                    {
                        "id": RCA_METHOD_BARO,
                        "status": "unavailable",
                        "reason_code": "dependency_unavailable",
                        "reason": "BARO RobustScorer is unavailable in this runtime.",
                    },
                ],
            },
        )
        by_id = {item["id"]: item for item in methods}
        self.assertEqual(by_id[RCA_METHOD_CAUSAL_LEGACY]["status"], "available")
        self.assertEqual(by_id[RCA_METHOD_BARO]["status"], "unavailable")
        self.assertEqual(by_id[RCA_METHOD_BARO]["reason_code"], "dependency_unavailable")
        self.assertNotIn("config_sha256", by_id[RCA_METHOD_CAUSAL_LEGACY])

    def test_selected_method_is_persisted_in_state_and_provenance(self):
        configuration = validate_rca_configuration(
            {"enabled": True, "method": RCA_METHOD_CAUSAL_LEGACY}
        )
        state = initial_rca_state(
            configuration,
            {"status": "ready", "input_contract": "cloudsentinel.crca-input/v1"},
            {
                "id": RCA_METHOD_CAUSAL_LEGACY,
                "config_sha256": "b" * 64,
                "minimum_observations": 2,
                "minimum_features": 2,
            },
        )
        self.assertEqual(state["method"], RCA_METHOD_CAUSAL_LEGACY)
        self.assertEqual(state["algorithm_id"], RCA_METHOD_CAUSAL_LEGACY)
        public_run = MonitoringHistoryStore.public_rca_run({
            "run_id": "rca_one",
            "method": RCA_METHOD_CAUSAL_LEGACY,
            "algorithm": {"id": RCA_METHOD_CAUSAL_LEGACY},
            "input_reference": "private/path.json",
        })
        self.assertEqual(public_run["method"], RCA_METHOD_CAUSAL_LEGACY)
        self.assertNotIn("input_reference", public_run)

        baro_configuration = validate_rca_configuration(
            {"enabled": True, "method": RCA_METHOD_BARO}
        )
        baro_state = initial_rca_state(
            baro_configuration,
            {"status": "ready", "input_contract": "cloudsentinel.crca-input/v1"},
            {
                "id": RCA_METHOD_BARO,
                "input_contract": "cloudsentinel.baro-input/v1",
                "preprocessing_contract": "cloudsentinel.baro-robust-scorer-raw/v1",
                "config_sha256": "c" * 64,
                "minimum_observations": 4,
                "minimum_features": 2,
                "minimum_baseline_observations": 2,
                "minimum_post_change_observations": 2,
            },
        )
        self.assertEqual(baro_state["input_contract"], "cloudsentinel.baro-input/v1")
        self.assertEqual(
            baro_state["preprocessing_contract"],
            "cloudsentinel.baro-robust-scorer-raw/v1",
        )
        self.assertNotEqual(
            baro_state["algorithm_config_sha256"], state["algorithm_config_sha256"]
        )

    def test_stable_run_identity_depends_on_method(self):
        common = {"threshold_percentage": 50.0, "configuration_version": 1}
        causal = stable_run_id(
            "monitor_one", "cycle_one", "a" * 64,
            {**common, "method": RCA_METHOD_CAUSAL_LEGACY},
        )
        baro = stable_run_id(
            "monitor_one", "cycle_one", "a" * 64,
            {**common, "method": RCA_METHOD_BARO},
        )
        self.assertNotEqual(causal, baro)

    def test_explicit_unknown_record_is_not_silently_treated_as_legacy(self):
        self.assertEqual(rca_method_id({"method": "unknown/method"}), "unknown/method")

    def test_runtime_capability_and_submission_are_method_aware(self):
        tasks_module = types.ModuleType("anomaly_detection.crca.tasks")
        tasks_module.run_crca_task = Mock()
        tasks_module.run_monitoring_rca_task = Mock()
        config_module = types.ModuleType("anomaly_detection.crca.config")
        config_module.get_config = lambda: {"alpha": 0.1}
        config_module.set_config = Mock()
        config_module.set_initial_config = Mock()
        sys.modules.pop("anomaly_detection.crca.app", None)
        with patch.dict(sys.modules, {
            "anomaly_detection.crca.tasks": tasks_module,
            "anomaly_detection.crca.config": config_module,
        }):
            crca_app = importlib.import_module("anomaly_detection.crca.app")

        client = crca_app.app.test_client()
        available_baro = {
            "id": RCA_METHOD_BARO,
            "display_name": "BARO RobustScorer",
            "status": "available",
            "input_contract": "cloudsentinel.baro-input/v1",
            "preprocessing_contract": "cloudsentinel.baro-robust-scorer-raw/v1",
            "minimum_requirements": {
                "observations": 4,
                "features": 2,
                "baseline_observations": 2,
                "post_change_observations": 2,
            },
            "config_sha256": "b" * 64,
        }
        with (
            patch.object(crca_app, "get_config", return_value={"alpha": 0.1}),
            patch.object(crca_app, "baro_capability", return_value=available_baro),
        ):
            response = client.get("/internal/monitoring-capability")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["algorithm_id"], RCA_METHOD_CAUSAL_LEGACY)
        methods = {item["id"]: item for item in payload["methods"]}
        self.assertEqual(methods[RCA_METHOD_CAUSAL_LEGACY]["status"], "available")
        self.assertEqual(methods[RCA_METHOD_BARO]["status"], "available")

        with patch.object(crca_app, "baro_capability", return_value={"status": "unavailable"}):
            baro = client.post(
                "/internal/monitoring-runs",
                json={"session_id": "monitor_one", "run_id": "rca_one", "method": RCA_METHOD_BARO},
            )
        unknown = client.post(
            "/internal/monitoring-runs",
            json={"session_id": "monitor_one", "run_id": "rca_one", "method": "unknown/method"},
        )
        self.assertEqual(baro.status_code, 409)
        self.assertEqual(unknown.status_code, 400)

        with patch.object(crca_app.run_monitoring_rca_task, "apply_async") as submit:
            legacy = client.post(
                "/internal/monitoring-runs",
                json={"session_id": "monitor_one", "run_id": "rca_one"},
            )
        self.assertEqual(legacy.status_code, 202)
        self.assertEqual(
            submit.call_args.kwargs["args"],
            ["monitor_one", "rca_one", RCA_METHOD_CAUSAL_LEGACY],
        )
        with (
            patch.object(crca_app, "baro_capability", return_value=available_baro),
            patch.object(crca_app.run_monitoring_rca_task, "apply_async") as submit,
        ):
            available = client.post(
                "/internal/monitoring-runs",
                json={"session_id": "monitor_one", "run_id": "rca_baro", "method": RCA_METHOD_BARO},
            )
        self.assertEqual(available.status_code, 202)
        self.assertEqual(
            submit.call_args.kwargs["args"],
            ["monitor_one", "rca_baro", RCA_METHOD_BARO],
        )


if __name__ == "__main__":
    unittest.main()
