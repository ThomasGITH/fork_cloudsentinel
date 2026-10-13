import json
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from anomaly_detection.crca.monitoring_result import (
    normalize_feature_metadata,
    safe_monitoring_result,
)

from data_ingestion.monitoring_history import MonitoringHistoryStore, canonical_sha256
from data_ingestion.monitoring_rca import (
    MonitoringRcaError,
    assess_recipe_compatibility,
    decide_trigger,
    initial_rca_state,
    mark_triggered,
    select_analysis_input,
    stable_run_id,
    validate_rca_configuration,
)


ROOT = Path(__file__).resolve().parents[1]


def compatible_recipe():
    return {
        "contract": "cloudsentinel.live-input/v1",
        "sampling_interval_seconds": 60,
        "alignment": {"grid": "inclusive_utc_range", "timestamp_match": "exact_microsecond", "off_grid_samples": "drop_with_warning"},
        "missing_data": {"offline_representation": "empty", "live_policy": "reject_window", "imputation": "none"},
        "feature_order": ["cpu|pod=checkout", "memory|pod=checkout"],
        "query_executions": [
            {"execution_id": "cpu", "feature_name": "cpu_usage", "identity_labels": ["pod"]},
            {"execution_id": "memory", "feature_name": "memory_usage", "identity_labels": ["pod"]},
        ],
        "series_mapping": [
            {"feature_id": "cpu|pod=checkout", "execution_id": "cpu", "metric_labels": {"pod": "checkout"}},
            {"feature_id": "memory|pod=checkout", "execution_id": "memory", "metric_labels": {"pod": "checkout"}},
        ],
    }


def cycle(percentage=75.0, count=20, at="2026-10-11T10:10:00Z"):
    return {
        "prediction_count": count,
        "anomaly_percentage": percentage,
        "completed_at": at,
    }


class MonitoringRcaPolicyTests(unittest.TestCase):
    def state(self, **configuration):
        config = validate_rca_configuration({"enabled": True, **configuration})
        return initial_rca_state(
            config,
            assess_recipe_compatibility(compatible_recipe()),
            {
                "algorithm_id": "cloudsentinel.causal-rca/legacy-v1",
                "config_sha256": "a" * 64,
                "minimum_observations": 2,
                "minimum_features": 2,
            },
        )

    def test_default_is_disabled_and_configuration_is_bounded(self):
        self.assertFalse(validate_rca_configuration(None)["enabled"])
        for value in (-0.1, 100.1):
            with self.assertRaises(MonitoringRcaError):
                validate_rca_configuration({"threshold_percentage": value})
        with self.assertRaises(MonitoringRcaError):
            validate_rca_configuration({"analysis_window_seconds": 10})
        with self.assertRaises(MonitoringRcaError):
            validate_rca_configuration({"cooldown_seconds": 604801})

    def test_compatibility_uses_explicit_recipe_metadata_and_order(self):
        ready = assess_recipe_compatibility(compatible_recipe())
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(
            [item["feature_id"] for item in ready["feature_metadata"]],
            compatible_recipe()["feature_order"],
        )
        changed = compatible_recipe()
        changed["series_mapping"][0]["metric_labels"] = {}
        result = assess_recipe_compatibility(changed)
        self.assertEqual(result["status"], "ready")
        self.assertIsNone(result["feature_metadata"][0]["entity"])

    def test_empty_identity_labels_use_explicit_series_pod(self):
        recipe = compatible_recipe()
        for execution in recipe["query_executions"]:
            execution["identity_labels"] = []
        ready = assess_recipe_compatibility(recipe)
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["feature_metadata"][0]["entity"], {
            "type": "pod", "id": "checkout", "source": "prometheus_series_label"
        })

    def test_unique_features_without_entity_metadata_are_ready(self):
        recipe = compatible_recipe()
        for execution in recipe["query_executions"]:
            execution["identity_labels"] = []
        for mapping in recipe["series_mapping"]:
            mapping["metric_labels"] = {"namespace": "boutique"}
        ready = assess_recipe_compatibility(recipe)
        self.assertEqual(ready["status"], "ready")
        self.assertTrue(all(item["entity"] is None for item in ready["feature_metadata"]))

    def test_same_metric_across_pods_keeps_unique_feature_columns(self):
        recipe = compatible_recipe()
        recipe["feature_order"] = ["cpu_usage|pod=cart", "cpu_usage|pod=checkout"]
        recipe["query_executions"] = [
            {"execution_id": "cpu", "feature_name": "cpu_usage", "identity_labels": []}
        ]
        recipe["series_mapping"] = [
            {"feature_id": "cpu_usage|pod=cart", "execution_id": "cpu", "metric_labels": {"pod": "cart"}},
            {"feature_id": "cpu_usage|pod=checkout", "execution_id": "cpu", "metric_labels": {"pod": "checkout"}},
        ]
        ready = assess_recipe_compatibility(recipe)
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(
            [item["feature_name"] for item in ready["feature_metadata"]],
            recipe["feature_order"],
        )

    def test_explicit_resolved_pod_target_is_used_and_conflict_is_rejected(self):
        recipe = compatible_recipe()
        recipe["query_executions"][0].update({
            "execution_mode": "per_target",
            "target_type": "pod",
            "resolved_target": {"pod": "checkout"},
            "identity_labels": [],
        })
        recipe["series_mapping"][0]["metric_labels"] = {}
        ready = assess_recipe_compatibility(recipe)
        self.assertEqual(ready["feature_metadata"][0]["entity"]["source"], "resolved_query_target")
        recipe["series_mapping"][0]["metric_labels"] = {"pod": "cart"}
        conflict = assess_recipe_compatibility(recipe)
        self.assertEqual(conflict["status"], "incompatible")
        self.assertEqual(conflict["reason_code"], "conflicting_entity_metadata")

    def test_precise_mapping_and_execution_failures(self):
        duplicate = compatible_recipe()
        duplicate["series_mapping"][1]["feature_id"] = duplicate["series_mapping"][0]["feature_id"]
        self.assertEqual(assess_recipe_compatibility(duplicate)["reason_code"], "invalid_series_mapping")
        duplicate_feature = compatible_recipe()
        duplicate_feature["feature_order"][1] = duplicate_feature["feature_order"][0]
        self.assertEqual(assess_recipe_compatibility(duplicate_feature)["reason_code"], "duplicate_feature_id")
        unknown = compatible_recipe()
        unknown["series_mapping"][0]["execution_id"] = "unknown"
        self.assertEqual(assess_recipe_compatibility(unknown)["reason_code"], "unknown_execution_id")
        missing_name = compatible_recipe()
        missing_name["query_executions"][0].pop("feature_name")
        self.assertEqual(assess_recipe_compatibility(missing_name)["reason_code"], "missing_query_feature_name")

    def test_strict_threshold_and_no_new_data_do_not_trigger(self):
        state = self.state(threshold_percentage=50)
        _, equal = decide_trigger(state, cycle(50))
        _, no_data = decide_trigger(state, cycle(None, 0))
        _, above = decide_trigger(state, cycle(50.01))
        self.assertEqual(equal["status"], "below_or_equal")
        self.assertEqual(no_data["status"], "no_new_data")
        self.assertEqual(above["status"], "trigger")

    def test_episode_recovery_cooldown_and_persistent_high_do_not_storm(self):
        state = self.state(threshold_percentage=50, cooldown_seconds=3600)
        state = mark_triggered(state, "rca_one", "2026-10-11T10:00:00Z")
        state, high = decide_trigger(state, cycle(100, at="2026-10-11T10:05:00Z"))
        self.assertEqual(high["status"], "suppressed_episode")
        state, no_data = decide_trigger(state, cycle(None, 0, "2026-10-11T10:06:00Z"))
        self.assertEqual(no_data["status"], "no_new_data")
        self.assertEqual(state["episode_state"], "triggered")
        state, recovered = decide_trigger(state, cycle(20, at="2026-10-11T10:10:00Z"))
        self.assertEqual(recovered["status"], "recovered")
        self.assertEqual(state["episode_state"], "recovered_cooldown")
        state, suppressed = decide_trigger(state, cycle(80, at="2026-10-11T10:20:00Z"))
        self.assertEqual(suppressed["status"], "suppressed_cooldown")
        self.assertEqual(state["episode_state"], "cooldown_episode")
        state, _ = decide_trigger(state, cycle(10, at="2026-10-11T11:01:00Z"))
        self.assertEqual(state["episode_state"], "armed")
        _, retrigger = decide_trigger(state, cycle(80, at="2026-10-11T11:02:00Z"))
        self.assertEqual(retrigger["status"], "trigger")

    def test_analysis_window_uses_immutable_rows_and_rejects_bad_values(self):
        readiness = assess_recipe_compatibility(compatible_recipe())
        payload = {
            "feature_order": compatible_recipe()["feature_order"],
            "feature_order_sha256": "f" * 64,
            "timestamps": ["2026-10-11T10:00:00Z", "2026-10-11T10:05:00Z", "2026-10-11T10:10:00Z"],
            "matrix": [[1, 2], [3, 4], [5, 6]],
            "input_reference": "input:sha256:" + "b" * 64,
            "input_sha256": "b" * 64,
        }
        selected = select_analysis_input(
            payload,
            end_timestamp="2026-10-11T10:10:00Z",
            window_seconds=300,
            feature_metadata=readiness["feature_metadata"],
            sampling_interval_seconds=60,
        )
        self.assertEqual(selected["timestamps"], payload["timestamps"][1:])
        self.assertEqual(selected["matrix"], [[3.0, 4.0], [5.0, 6.0]])
        short = dict(payload, timestamps=payload["timestamps"][1:], matrix=payload["matrix"][1:])
        with self.assertRaisesRegex(MonitoringRcaError, "cover the requested") as caught:
            select_analysis_input(
                short,
                end_timestamp="2026-10-11T10:10:00Z",
                window_seconds=600,
                feature_metadata=readiness["feature_metadata"],
                sampling_interval_seconds=60,
            )
        self.assertEqual(caught.exception.reason_code, "incomplete_analysis_window")
        payload["matrix"][2][0] = float("nan")
        with self.assertRaisesRegex(MonitoringRcaError, "finite") as caught:
            select_analysis_input(
                payload,
                end_timestamp="2026-10-11T10:10:00Z",
                window_seconds=300,
                feature_metadata=readiness["feature_metadata"],
                sampling_interval_seconds=60,
            )
        self.assertEqual(caught.exception.reason_code, "non_finite_runtime_input")

    def test_insufficient_context_is_a_runtime_reason_not_static_incompatibility(self):
        readiness = assess_recipe_compatibility(compatible_recipe())
        self.assertEqual(readiness["status"], "ready")
        with self.assertRaises(MonitoringRcaError) as caught:
            select_analysis_input(
                {
                    "feature_order": compatible_recipe()["feature_order"],
                    "feature_order_sha256": "f" * 64,
                    "timestamps": ["2026-10-11T10:10:00Z"],
                    "matrix": [[1, 2]],
                },
                end_timestamp="2026-10-11T10:10:00Z",
                window_seconds=60,
                feature_metadata=readiness["feature_metadata"],
                sampling_interval_seconds=60,
            )
        self.assertEqual(caught.exception.reason_code, "insufficient_runtime_context")

    def test_stable_identity_is_idempotent(self):
        identity = {"algorithm": "v1"}
        first = stable_run_id("monitor_one", "cycle_one", "a" * 64, identity)
        self.assertEqual(first, stable_run_id("monitor_one", "cycle_one", "a" * 64, identity))
        self.assertNotEqual(first, stable_run_id("monitor_one", "cycle_two", "a" * 64, identity))

    def test_result_normalization_is_bounded_and_column_traceable(self):
        metadata = assess_recipe_compatibility(compatible_recipe())["feature_metadata"]
        png = __import__("base64").b64encode(b"\x89PNG\r\n\x1a\nsmall").decode()
        result = safe_monitoring_result(
            {"ranking": "metric,index,score\ncpu|pod=checkout,0,0.9\nmemory|pod=checkout,1,0.1\n", "graph_image": [png]},
            metadata,
        )
        self.assertEqual(result["ranking"][0]["column_index"], 0)
        self.assertEqual(result["ranking"][0]["feature_id"], "cpu|pod=checkout")
        self.assertEqual(result["ranking"][0]["entity"]["source"], "prometheus_series_label")
        self.assertEqual(result["ranking"][0]["entity_id"], "checkout")
        self.assertEqual(result["graphs"][0]["media_type"], "image/png")
        public_json = json.dumps(result)
        self.assertNotIn("metric_labels", public_json)
        self.assertNotIn("resolved_query", public_json)
        with self.assertRaisesRegex(ValueError, "unknown feature"):
            safe_monitoring_result({"ranking": "metric,index,score\nunknown,0,1\n"}, metadata)

    def test_result_roundtrip_allows_unavailable_entity_and_legacy_flat_metadata(self):
        recipe = compatible_recipe()
        for mapping in recipe["series_mapping"]:
            mapping["metric_labels"] = {}
        metadata = assess_recipe_compatibility(recipe)["feature_metadata"]
        response = {"ranking": "metric,index,score\ncpu|pod=checkout,0,0.9\nmemory|pod=checkout,1,0.1\n"}
        result = safe_monitoring_result(response, metadata)
        self.assertIsNone(result["ranking"][0]["entity"])
        self.assertIsNone(result["ranking"][0]["entity_id"])

        legacy = [dict(item, entity=None, entity_type="pod", entity_id="checkout") for item in metadata]
        normalized = normalize_feature_metadata(legacy)
        self.assertEqual(normalized[0]["entity"]["source"], "legacy_explicit")

    def test_result_rejects_conflicting_or_reordered_column_metadata(self):
        metadata = assess_recipe_compatibility(compatible_recipe())["feature_metadata"]
        conflicting = [dict(item) for item in metadata]
        conflicting[0]["entity"] = {"type": "pod", "id": "cart", "source": "prometheus_series_label"}
        with self.assertRaisesRegex(ValueError, "conflicts"):
            normalize_feature_metadata(conflicting)
        reordered = [dict(item) for item in metadata]
        reordered[0]["column_index"] = 1
        with self.assertRaisesRegex(ValueError, "column order"):
            normalize_feature_metadata(reordered)


class MonitoringRcaPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = MonitoringHistoryStore(self.temporary.name)
        recipe = compatible_recipe()
        recipe.update({"schema_version": 1, "contract": "cloudsentinel.live-input/v1", "sampling_interval_seconds": 60, "feature_order_sha256": "f" * 64})
        recipe["recipe_sha256"] = canonical_sha256(recipe)
        state = initial_rca_state(validate_rca_configuration({"enabled": True}), assess_recipe_compatibility(recipe), {"config_sha256": "a" * 64})
        self.store.create_session(
            {"session_id": "monitor_one", "model_id": "model_one", "status": "active", "created_at": "2026-10-11T10:00:00Z", "rca": state}, recipe
        )
        self.store.commit_cycle(
            "monitor_one", "cycle_one",
            input_payload={"feature_order": recipe["feature_order"], "feature_order_sha256": "f" * 64, "timestamps": ["2026-10-11T10:00:00Z", "2026-10-11T10:01:00Z"], "matrix": [[1, 2], [2, 3]]},
            result_payload={"binary_predictions": [1, 1], "anomaly_scores": [1, 2]},
            cycle={"status": "completed", "completed_at": "2026-10-11T10:01:01Z", "prediction_count": 2, "anomaly_count": 2, "anomaly_percentage": 100.0},
            next_watermark="2026-10-11T10:01:00Z", session_status="active",
        )

    def test_run_input_result_and_history_survive_reopen(self):
        rca_input = {"matrix": [[1.0, 2.0], [2.0, 3.0]], "timestamps": ["a", "b"], "feature_order": ["a", "b"]}
        state = self.store.read_session("monitor_one")["rca"]
        self.store.create_rca_run(
            "monitor_one", "cycle_one",
            {"run_id": "rca_one", "status": "queued", "created_at": "2026-10-11T10:02:00Z", "dispatch_status": "pending"},
            input_payload=rca_input, event={"status": "started", "threshold_percentage": 50}, rca_state=state,
        )
        self.assertIsNotNone(self.store.claim_rca_run("monitor_one", "rca_one"))
        result = {"ranking": [{"rank": 1, "feature_name": "a", "score": 0.9}], "graphs": []}
        self.store.complete_rca_run("monitor_one", "rca_one", result)
        reopened = MonitoringHistoryStore(self.temporary.name)
        detail = reopened.rca_run_detail("monitor_one", "rca_one")
        self.assertEqual(detail["run"]["status"], "completed")
        self.assertEqual(detail["result"]["ranking"][0]["feature_name"], "a")
        encoded = json.dumps(reopened.session_detail("monitor_one"))
        self.assertNotIn('"matrix"', encoded)
        self.assertNotIn("input_reference", encoded)

    def test_duplicate_run_and_claim_are_idempotent(self):
        run = {"run_id": "rca_one", "status": "queued", "created_at": "2026-10-11T10:02:00Z"}
        state = self.store.read_session("monitor_one")["rca"]
        first = self.store.create_rca_run("monitor_one", "cycle_one", run, input_payload={"matrix": [], "timestamps": [], "feature_order": []}, event={"status": "started"}, rca_state=state)
        second = self.store.create_rca_run("monitor_one", "cycle_one", run, input_payload={"changed": True}, event={"status": "started"}, rca_state=state)
        self.assertEqual(first["input_sha256"], second["input_sha256"])
        self.assertIsNotNone(self.store.claim_rca_run("monitor_one", "rca_one"))
        self.assertIsNone(self.store.claim_rca_run("monitor_one", "rca_one"))

    def test_rca_failure_does_not_fail_monitoring_and_submitted_run_can_finish_after_stop(self):
        state = self.store.read_session("monitor_one")["rca"]
        self.store.create_rca_run(
            "monitor_one", "cycle_one",
            {"run_id": "rca_failed", "status": "queued", "created_at": "2026-10-11T10:02:00Z"},
            input_payload={"matrix": [[1, 2], [2, 3]], "timestamps": ["a", "b"], "feature_order": ["a", "b"]},
            event={"status": "started"}, rca_state=state,
        )
        self.store.claim_rca_run("monitor_one", "rca_failed")
        self.store.fail_rca_run("monitor_one", "rca_failed", "safe failure")
        self.assertEqual(self.store.read_session("monitor_one")["status"], "active")
        self.store.commit_cycle(
            "monitor_one", "cycle_two",
            input_payload={"feature_order": ["a", "b"], "timestamps": ["c", "d"], "matrix": [[1, 2], [2, 3]]},
            result_payload={"binary_predictions": [1, 1], "anomaly_scores": [1, 2]},
            cycle={"status": "completed", "completed_at": "2026-10-11T10:04:00Z", "prediction_count": 2, "anomaly_count": 2, "anomaly_percentage": 100.0},
            next_watermark="d", session_status="active",
        )
        self.store.create_rca_run(
            "monitor_one", "cycle_two",
            {"run_id": "rca_finishes", "status": "queued", "created_at": "2026-10-11T10:04:00Z"},
            input_payload={"matrix": [[1, 2], [2, 3]], "timestamps": ["c", "d"], "feature_order": ["a", "b"]},
            event={"status": "started"}, rca_state=state,
        )
        self.store.claim_rca_run("monitor_one", "rca_finishes")
        self.store.stop_session("monitor_one", "2026-10-11T10:03:00Z")
        self.store.complete_rca_run("monitor_one", "rca_finishes", {"ranking": [], "graphs": []})
        self.assertEqual(self.store.read_session("monitor_one")["status"], "stopped")
        self.assertEqual(self.store.read_rca_run("monitor_one", "rca_finishes")["status"], "completed")

    def test_committed_cycle_remains_reconcilable_until_event_is_atomic(self):
        self.assertEqual(self.store.pending_rca_cycles("monitor_one"), ["cycle_one"])
        state = self.store.read_session("monitor_one")["rca"]
        self.store.record_rca_event(
            "monitor_one", "cycle_one", {"status": "disabled"}, rca_state=state
        )
        self.assertEqual(self.store.pending_rca_cycles("monitor_one"), [])

    def test_worker_columns_and_ranking_roundtrip_exact_feature_ids_without_entity(self):
        crca_module = types.ModuleType("anomaly_detection.crca.crca")
        crca_module.run_crca = lambda *_args, **_kwargs: None
        config_module = types.ModuleType("anomaly_detection.crca.config")
        config_module.get_config = lambda: {}
        sys.modules.pop("anomaly_detection.crca.tasks", None)
        with patch.dict(sys.modules, {
            "anomaly_detection.crca.crca": crca_module,
            "anomaly_detection.crca.config": config_module,
        }):
            tasks = importlib.import_module("anomaly_detection.crca.tasks")

        recipe = compatible_recipe()
        for mapping in recipe["series_mapping"]:
            mapping["metric_labels"] = {}
        metadata = assess_recipe_compatibility(recipe)["feature_metadata"]
        configuration = {"alpha": 0.1}
        self.store.create_rca_run(
            "monitor_one",
            "cycle_one",
            {
                "run_id": "rca_worker",
                "status": "queued",
                "created_at": "2026-10-11T10:02:00Z",
                "algorithm": {"configuration_sha256": canonical_sha256(configuration)},
            },
            input_payload={
                "matrix": [[1.0, 3.0], [2.0, 4.0]],
                "timestamps": ["2026-10-11T10:00:00Z", "2026-10-11T10:01:00Z"],
                "feature_order": recipe["feature_order"],
                "feature_metadata": metadata,
            },
            event={"status": "started"},
            rca_state=self.store.read_session("monitor_one")["rca"],
        )
        ranking = (
            "metric,index,score\n"
            "cpu|pod=checkout,0,0.9\n"
            "memory|pod=checkout,1,0.1\n"
        )
        with (
            patch.object(tasks, "_history_store", return_value=self.store),
            patch.object(tasks, "get_config", return_value=configuration),
            patch.object(tasks, "run_crca", return_value={"ranking": ranking}) as run,
        ):
            response = tasks.run_monitoring_rca_task.run("monitor_one", "rca_worker")
        self.assertEqual(response["status"], "completed")
        frame = run.call_args.args[0]
        self.assertEqual(list(frame.columns), recipe["feature_order"])
        detail = self.store.rca_run_detail("monitor_one", "rca_worker")
        self.assertEqual(detail["result"]["ranking"][0]["column_index"], 0)
        self.assertEqual(detail["result"]["ranking"][0]["feature_id"], recipe["feature_order"][0])
        self.assertIsNone(detail["result"]["ranking"][0]["entity"])


class MonitoringRcaStaticBoundaryTests(unittest.TestCase):
    def test_generic_monitoring_has_no_detector_dispatch_or_legacy_refetch(self):
        source = (ROOT / "data_ingestion/app.py").read_text()
        start = source.index("def _process_cycle_rca")
        section = source[start:source.index("@celery.task(bind=True)", start)]
        self.assertNotIn("collect_crca_data", section)
        self.assertNotIn("detector_id ==", section)
        self.assertIn("read_artifact(cycle[\"input_reference\"])", section)

    def test_crca_naming_keeps_explicit_and_legacy_boundaries(self):
        source = (ROOT / "anomaly_detection/crca/crca.py").read_text()
        self.assertIn("explicit_names = crca_info.get('data', {}).get('feature_names')", source)
        self.assertIn("crca_info['data']['crca_pods']", source)
        self.assertIn("labels=labels", source)

    def test_ui_has_threshold_line_and_diagnostic_caveat(self):
        source = (ROOT / "user_interface/monitoring_project/config_app/static/js/monitoring_history.js").read_text()
        self.assertIn("Automatic RCA threshold (%)", source)
        self.assertIn("not proof of the actual root cause", source)
        self.assertIn('["Rank","Feature","Entity","Metric","Score"]', source)
        self.assertIn(': "Unavailable"', source)
        self.assertIn("item.feature_id||item.feature_name", source)
        self.assertNotIn('["Rank","Feature","Pod","Metric","Score"]', source)
        self.assertIn("RCA reanalysis started", source)
        self.assertIn("this episode was already analysed", source)
        self.assertIn("View earlier analysis", source)
        self.assertIn("episode budget reached", source)


if __name__ == "__main__":
    unittest.main()
