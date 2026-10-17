import tempfile
import unittest
from pathlib import Path

from anomaly_detection.crca.monitoring_result import (
    safe_monitoring_result,
    validate_pinned_method_contract,
)
from data_ingestion.monitoring_history import MonitoringHistoryStore, canonical_sha256
from data_ingestion.monitoring_rca import (
    RCA_METHOD_BARO,
    RCA_METHOD_CAUSAL_LEGACY,
    MonitoringRcaError,
    assess_recipe_compatibility,
    decide_trigger,
    initial_rca_state,
    method_preprocessing_configuration,
    validate_method_capability,
    validate_rca_configuration,
)


def _capability(method):
    if method == RCA_METHOD_BARO:
        return {
            "id": method,
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
    return {
        "id": method,
        "status": "available",
        "input_contract": "cloudsentinel.crca-input/v1",
        "preprocessing_contract": "cloudsentinel.crca-minmax/v1",
        "minimum_requirements": {"observations": 2, "features": 2},
        "config_sha256": "a" * 64,
    }


class SelectableRcaContractHardeningTests(unittest.TestCase):
    def test_runtime_capability_must_match_selected_contract_and_minimums(self):
        for method in (RCA_METHOD_CAUSAL_LEGACY, RCA_METHOD_BARO):
            self.assertEqual(validate_method_capability(method, _capability(method))["id"], method)

        crossed = _capability(RCA_METHOD_BARO)
        crossed["preprocessing_contract"] = "cloudsentinel.crca-minmax/v1"
        with self.assertRaises(MonitoringRcaError) as raised:
            validate_method_capability(RCA_METHOD_BARO, crossed)
        self.assertEqual(raised.exception.reason_code, "rca_method_contract_mismatch")

        invalid = _capability(RCA_METHOD_BARO)
        invalid["minimum_requirements"]["baseline_observations"] = True
        with self.assertRaises(MonitoringRcaError) as raised:
            validate_method_capability(RCA_METHOD_BARO, invalid)
        self.assertEqual(raised.exception.reason_code, "invalid_rca_method_metadata")

        invalid = _capability(RCA_METHOD_CAUSAL_LEGACY)
        invalid["config_sha256"] = "not-a-digest"
        with self.assertRaises(MonitoringRcaError):
            validate_method_capability(RCA_METHOD_CAUSAL_LEGACY, invalid)

    def test_data_ingestion_revalidates_runtime_capability(self):
        source = (
            Path(__file__).resolve().parents[1] / "data_ingestion" / "app.py"
        ).read_text()
        start = source.index("def _rca_method_capability")
        section = source[start : source.index("def _dispatch_rca_method", start)]
        self.assertIn("validate_method_capability(method_id, value)", section)

    def test_baro_and_causal_preprocessing_are_explicit_and_distinct(self):
        causal = method_preprocessing_configuration(RCA_METHOD_CAUSAL_LEGACY)
        baro = method_preprocessing_configuration(RCA_METHOD_BARO)
        self.assertEqual(causal, {"feature_range": [0.0, 1.0]})
        self.assertEqual(baro["input"], "raw_finite_numeric")
        self.assertNotIn("feature_range", baro)

    def test_worker_rejects_cross_wired_method_contracts_but_keeps_legacy_causal(self):
        validate_pinned_method_contract({}, {}, RCA_METHOD_CAUSAL_LEGACY)
        validate_pinned_method_contract(
            {"preprocessing": {"contract": "cloudsentinel.baro-robust-scorer-raw/v1"}},
            {"input_contract": "cloudsentinel.baro-input/v1"},
            RCA_METHOD_BARO,
        )
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            validate_pinned_method_contract(
                {"preprocessing": {"contract": "cloudsentinel.crca-minmax/v1"}},
                {"input_contract": "cloudsentinel.baro-input/v1"},
                RCA_METHOD_BARO,
            )
        with self.assertRaisesRegex(ValueError, "input contract"):
            validate_pinned_method_contract(
                {"preprocessing": {"contract": "cloudsentinel.baro-robust-scorer-raw/v1"}},
                {"input_contract": "cloudsentinel.crca-input/v1"},
                RCA_METHOD_BARO,
            )

    def test_legacy_worker_record_cannot_be_redefined_by_dispatch_argument(self):
        import importlib
        import sys
        import types
        from unittest.mock import patch

        original_modules = set(sys.modules)
        crca = types.ModuleType("anomaly_detection.crca.crca")
        crca.run_crca = lambda *_args, **_kwargs: None
        config = types.ModuleType("anomaly_detection.crca.config")
        config.get_config = lambda: {}
        sys.modules.pop("anomaly_detection.crca.tasks", None)
        with patch.dict(
            sys.modules,
            {
                "anomaly_detection.crca.crca": crca,
                "anomaly_detection.crca.config": config,
            },
        ):
            tasks = importlib.import_module("anomaly_detection.crca.tasks")
        try:
            self.assertEqual(tasks._resolve_method({}, None), RCA_METHOD_CAUSAL_LEGACY)
            self.assertEqual(
                tasks._resolve_method({}, RCA_METHOD_CAUSAL_LEGACY),
                RCA_METHOD_CAUSAL_LEGACY,
            )
            with self.assertRaisesRegex(ValueError, "pinned run"):
                tasks._resolve_method({}, RCA_METHOD_BARO)
        finally:
            for module_name in set(sys.modules) - original_modules:
                sys.modules.pop(module_name, None)

    def test_trigger_policy_is_method_independent(self):
        readiness = {"status": "ready"}
        cycle = {
            "prediction_count": 20,
            "anomaly_percentage": 75.0,
            "completed_at": "2026-10-17T10:00:00Z",
        }
        events = []
        for method in (RCA_METHOD_CAUSAL_LEGACY, RCA_METHOD_BARO):
            capability = _capability(method)
            minimum = capability["minimum_requirements"]
            capability.update(
                minimum_observations=minimum["observations"],
                minimum_features=minimum["features"],
                minimum_baseline_observations=minimum.get("baseline_observations", 0),
                minimum_post_change_observations=minimum.get("post_change_observations", 0),
            )
            state = initial_rca_state(
                validate_rca_configuration({"enabled": True, "method": method}),
                readiness,
                capability,
            )
            _, event = decide_trigger(state, cycle)
            events.append(event)
        self.assertEqual(events[0]["status"], "trigger")
        self.assertEqual(events[0], events[1])

    def test_causal_result_rejects_duplicate_ranking_rows(self):
        metadata = [
            {
                "column_index": 0,
                "feature_id": "cpu|pod=a",
                "feature_name": "cpu|pod=a",
                "metric_name": "cpu",
                "entity": None,
                "entity_type": None,
                "entity_id": None,
            },
            {
                "column_index": 1,
                "feature_id": "memory|pod=a",
                "feature_name": "memory|pod=a",
                "metric_name": "memory",
                "entity": None,
                "entity_type": None,
                "entity_id": None,
            },
        ]
        response = {
            "ranking": "metric,index,score\ncpu|pod=a,0,0.9\ncpu|pod=a,0,0.8\n"
        }
        with self.assertRaisesRegex(ValueError, "invalid ranking"):
            safe_monitoring_result(response, metadata)


class SelectableRcaPublicProjectionTests(unittest.TestCase):
    def test_run_projection_drops_private_nested_configuration(self):
        projected = MonitoringHistoryStore.public_rca_run(
            {
                "run_id": "rca_one",
                "method": RCA_METHOD_BARO,
                "algorithm": {
                    "id": RCA_METHOD_BARO,
                    "configuration_sha256": "a" * 64,
                    "configuration": {"secret": "hidden"},
                    "artifact_path": "/private/model",
                },
                "preprocessing": {
                    "contract": "cloudsentinel.baro-robust-scorer-raw/v1",
                    "configuration_sha256": "b" * 64,
                    "configuration": {"raw_matrix": [[1, 2]]},
                },
                "model_identity": {
                    "model_id": "model_one",
                    "artifact_id": "artifact_one",
                    "artifact_manifest_sha256": "c" * 64,
                    "internal_url": "http://private",
                },
                "failure": {
                    "type": "safe",
                    "message": "safe message",
                    "traceback": "private traceback",
                },
            }
        )
        encoded = str(projected)
        self.assertNotIn("secret", encoded)
        self.assertNotIn("artifact_path", encoded)
        self.assertNotIn("raw_matrix", encoded)
        self.assertNotIn("internal_url", encoded)
        self.assertNotIn("traceback", encoded)
        self.assertEqual(projected["algorithm"]["id"], RCA_METHOD_BARO)

    def test_result_projection_allowlists_ranking_and_change_point_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = MonitoringHistoryStore(temporary)
            private_result = {
                "ranking": [
                    {
                        "rank": 1,
                        "column_index": 0,
                        "feature_id": "cpu|pod=a",
                        "feature_name": "cpu|pod=a",
                        "metric_name": "cpu",
                        "score": 8.0,
                        "entity": {"type": "pod", "id": "a", "source": "explicit", "token": "secret"},
                        "matrix": [[1.0]],
                    }
                ],
                "graphs": [{"media_type": "image/png", "base64": "safe", "path": "/private"}],
                "method": RCA_METHOD_BARO,
                "change_point_evidence": {
                    "source": "stored_monitoring_cycle_result",
                    "cycle_id": "cycle_one",
                    "cycle_result_sha256": "d" * 64,
                    "detected_change_point": "2026-10-17T10:00:00Z",
                    "index": 2,
                    "baseline_observation_count": 2,
                    "anomalous_observation_count": 2,
                    "promql": "private query",
                },
            }
            reference, digest = store._publish_blob("rca_result", private_result)
            session = store._session_dir("monitor_one")
            (session / "rca_runs" / "rca_one").mkdir(parents=True)
            record = {
                "run_id": "rca_one",
                "session_id": "monitor_one",
                "cycle_id": "cycle_one",
                "status": "completed",
                "result_reference": reference,
                "result_sha256": digest,
            }
            from data_ingestion.monitoring_history import _atomic_json

            _atomic_json(session / "rca_runs" / "rca_one" / "run.json", record)
            detail = store.rca_run_detail("monitor_one", "rca_one")
            encoded = str(detail)
            for forbidden in ("matrix", "token", "path", "promql", "/private"):
                self.assertNotIn(forbidden, encoded)
            self.assertEqual(detail["result"]["ranking"][0]["feature_id"], "cpu|pod=a")


if __name__ == "__main__":
    unittest.main()
