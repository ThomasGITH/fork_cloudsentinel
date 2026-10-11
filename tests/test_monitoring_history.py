import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data_ingestion.live_monitoring import (
    LiveMonitoringError,
    align_new_predictions,
    merge_buffer_with_watermark,
    merge_execution_projection,
)
from data_ingestion.monitoring_history import (
    MonitoringHistoryError,
    MonitoringHistoryStore,
    MonitoringSessionStopped,
    canonical_sha256,
)


def recipe():
    value = {
        "schema_version": 1,
        "contract": "cloudsentinel.live-input/v1",
        "prometheus_source_id": "cluster-default",
        "sampling_interval_seconds": 60,
        "feature_order": ["cpu|pod=checkout"],
        "feature_order_sha256": "f" * 64,
        "query_executions": [{"execution_id": "q1", "resolved_query": "private query"}],
        "series_mapping": [{"feature_id": "cpu|pod=checkout", "execution_id": "q1", "metric_labels": {"pod": "checkout"}}],
        "alignment": {"grid": "inclusive_utc_range", "timestamp_match": "exact_microsecond", "off_grid_samples": "drop_with_warning"},
        "missing_data": {"offline_representation": "empty", "live_policy": "reject_window", "imputation": "none"},
    }
    value["recipe_sha256"] = canonical_sha256(value)
    return value


def session():
    return {
        "session_id": "monitor_one",
        "display_name": "Checkout baseline",
        "model_id": "model_one",
        "detector_id": "local-outlier-factor",
        "detector_version": "1.0.0",
        "artifact_manifest_sha256": "a" * 64,
        "artifact_id": "artifact_model_one",
        "inference_contract": "cloudsentinel.inference/v1",
        "artifact_format": "sklearn-pipeline-joblib-v1",
        "recipe_contract": "cloudsentinel.live-input/v1",
        "recipe_sha256": recipe()["recipe_sha256"],
        "feature_order_sha256": "f" * 64,
        "sampling_interval_seconds": 60,
        "window_seconds": 600,
        "poll_interval_seconds": 60,
        "status": "queued",
        "created_at": "2026-10-11T10:00:00Z",
        "started_at": "2026-10-11T10:00:00Z",
    }


class PredictionAlignmentTests(unittest.TestCase):
    def test_first_cycle_and_if_lof_binary_semantics(self):
        result = align_new_predictions(
            ["2026-10-11T10:00:00Z", "2026-10-11T10:01:00Z"],
            {"input_observation_count": 2, "warmup_observations": 0, "binary_predictions": [0, 1], "anomaly_scores": [-0.4, 0.8]},
            None,
        )
        self.assertEqual(result["prediction_count"], 2)
        self.assertEqual(result["anomaly_count"], 1)
        self.assertEqual(result["anomaly_percentage"], 50.0)
        self.assertEqual(result["next_watermark"], "2026-10-11T10:01:00Z")

    def test_cgnn_warmup_aligns_to_later_timestamps(self):
        timestamps = [f"2026-10-11T10:0{minute}:00Z" for minute in range(4)]
        result = align_new_predictions(
            timestamps,
            {"input_observation_count": 4, "warmup_observations": 2, "binary_predictions": [1, 0], "anomaly_scores": [2.0, 0.1]},
            None,
        )
        self.assertEqual(result["prediction_timestamps"], timestamps[2:])
        self.assertEqual(result["prediction_count"], 2)

    def test_overlap_uses_watermark_and_zero_eligible_is_unavailable(self):
        timestamps = ["2026-10-11T10:00:00Z", "2026-10-11T10:01:00Z"]
        result = align_new_predictions(
            timestamps,
            {"input_observation_count": 2, "warmup_observations": 0, "binary_predictions": [1, 1], "anomaly_scores": [1.0, 2.0]},
            timestamps[-1],
        )
        self.assertEqual(result["prediction_count"], 0)
        self.assertIsNone(result["anomaly_percentage"])
        self.assertEqual(result["next_watermark"], timestamps[-1])

    def test_late_samples_do_not_rewrite_committed_context(self):
        existing = [{"timestamp": "2026-10-11T10:00:00Z", "values": [1.0]}]
        merged, late = merge_buffer_with_watermark(
            existing,
            ["2026-10-11T09:59:00Z", "2026-10-11T10:00:00Z", "2026-10-11T10:01:00Z"],
            [[9.0], [10.0], [2.0]],
            10,
            "2026-10-11T10:00:00Z",
        )
        self.assertEqual(late, 2)
        self.assertEqual(merged, [existing[0], {"timestamp": "2026-10-11T10:01:00Z", "values": [2.0]}])

    def test_invalid_lengths_order_and_values_are_rejected(self):
        base = {"input_observation_count": 2, "warmup_observations": 0, "binary_predictions": [0, 1], "anomaly_scores": [0.1, 0.2]}
        with self.assertRaisesRegex(LiveMonitoringError, "unique and ordered"):
            align_new_predictions(["b", "a"], base, None)
        changed = dict(base, binary_predictions=[0])
        with self.assertRaisesRegex(LiveMonitoringError, "aligned"):
            align_new_predictions(["a", "b"], changed, None)
        changed = dict(base, anomaly_scores=[0.1, math.inf])
        with self.assertRaisesRegex(LiveMonitoringError, "scores"):
            align_new_predictions(["a", "b"], changed, None)

    def test_durable_stop_cannot_be_reactivated_by_stale_execution_state(self):
        merged = merge_execution_projection(
            {"session_id": "monitor_one", "status": "stopped"},
            {"session_id": "monitor_one", "status": "active"},
        )
        self.assertEqual(merged["status"], "stopped")
        self.assertEqual(merged["execution_state"], "unavailable")


class MonitoringHistoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = MonitoringHistoryStore(self.temporary.name, max_observations=10, max_features=5)
        self.store.create_session(session(), recipe())

    def payloads(self):
        input_payload = {
            "schema_version": 1, "session_id": "monitor_one", "cycle_id": "cycle_one", "model_id": "model_one",
            "feature_order": ["cpu|pod=checkout"], "timestamps": ["2026-10-11T10:01:00Z"], "matrix": [[1.25]],
        }
        result_payload = {
            "schema_version": 1, "session_id": "monitor_one", "cycle_id": "cycle_one",
            "prediction_timestamps": ["2026-10-11T10:01:00Z"], "binary_predictions": [1], "anomaly_scores": [0.9],
        }
        cycle = {
            "status": "completed", "artifact_identity": {"artifact_id": "artifact_model_one", "artifact_manifest_sha256": "a" * 64}, "query_window": {"start": "2026-10-11T10:00:00Z", "end": "2026-10-11T10:01:00Z"},
            "inference_window": {"start": "2026-10-11T10:01:00Z", "end": "2026-10-11T10:01:00Z", "observation_count": 1},
            "prediction_window": {"start": "2026-10-11T10:01:00Z", "end": "2026-10-11T10:01:00Z"},
            "started_at": "2026-10-11T10:01:01Z", "completed_at": "2026-10-11T10:01:02Z",
            "prediction_count": 1, "anomaly_count": 1, "anomaly_percentage": 100.0, "warmup_observations": 0, "warnings": [],
        }
        return input_payload, result_payload, cycle

    def commit(self):
        input_payload, result_payload, cycle = self.payloads()
        return self.store.commit_cycle(
            "monitor_one", "cycle_one", input_payload=input_payload, result_payload=result_payload,
            cycle=cycle, next_watermark="2026-10-11T10:01:00Z", session_status="active",
        )

    def test_exact_private_artifacts_checksums_and_safe_projection(self):
        committed = self.commit()
        directory = Path(self.temporary.name) / "sessions" / "monitor_one" / "cycles" / "cycle_one"
        stored_input = self.store.read_artifact(committed["input_reference"])
        stored_result = self.store.read_artifact(committed["result_reference"])
        self.assertEqual(stored_input["matrix"], [[1.25]])
        self.assertEqual(stored_result["binary_predictions"], [1])
        self.assertEqual(committed["input_sha256"], canonical_sha256(stored_input))
        self.assertEqual(committed["result_sha256"], canonical_sha256(stored_result))
        public = self.store.session_detail("monitor_one")
        self.assertEqual(
            public["cycles"]["items"][0]["artifact_identity"]["artifact_id"],
            "artifact_model_one",
        )
        encoded = json.dumps(public)
        self.assertNotIn("private query", encoded)
        self.assertNotIn("matrix", encoded)
        self.assertNotIn("binary_predictions", encoded)
        self.assertNotIn(str(Path(self.temporary.name)), encoded)

    def test_retry_is_idempotent_and_watermark_advances_only_after_commit(self):
        first = self.commit()
        second = self.commit()
        self.assertEqual(first["cycle_id"], second["cycle_id"])
        reopened = MonitoringHistoryStore(self.temporary.name)
        stored = reopened.read_session("monitor_one")
        self.assertEqual(stored["cycle_count"], 1)
        self.assertEqual(stored["prediction_watermark"], "2026-10-11T10:01:00Z")

    def test_retry_cannot_change_committed_cycle_summary(self):
        self.commit()
        input_payload, result_payload, cycle = self.payloads()
        cycle["anomaly_count"] = 0
        with self.assertRaisesRegex(MonitoringHistoryError, "identity conflict"):
            self.store.commit_cycle(
                "monitor_one", "cycle_one", input_payload=input_payload,
                result_payload=result_payload, cycle=cycle,
                next_watermark="2026-10-11T10:01:00Z", session_status="active",
            )

    def test_partial_stage_is_ignored_after_reopen(self):
        cycles = Path(self.temporary.name) / "sessions" / "monitor_one" / "cycles"
        partial = cycles / ".cycle_partial.dead"
        partial.mkdir(parents=True)
        (partial / "input.json").write_text("{}")
        reopened = MonitoringHistoryStore(self.temporary.name)
        self.assertEqual(reopened.list_cycles("monitor_one")["total"], 0)

    def test_failed_publication_does_not_advance_watermark(self):
        input_payload, result_payload, cycle = self.payloads()
        with patch.object(self.store, "_publish_blob", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                self.store.commit_cycle(
                    "monitor_one", "cycle_one", input_payload=input_payload, result_payload=result_payload,
                    cycle=cycle, next_watermark="2026-10-11T10:01:00Z", session_status="active",
                )
        self.assertIsNone(self.store.read_session("monitor_one")["prediction_watermark"])
        self.assertEqual(self.store.list_cycles("monitor_one")["total"], 0)

    def test_stop_blocks_late_cycle_and_history_survives_execution_loss(self):
        self.store.stop_session("monitor_one", "2026-10-11T11:00:00Z")
        input_payload, result_payload, cycle = self.payloads()
        with self.assertRaises(MonitoringSessionStopped):
            self.store.commit_cycle(
                "monitor_one", "cycle_one", input_payload=input_payload, result_payload=result_payload,
                cycle=cycle, next_watermark="2026-10-11T10:01:00Z", session_status="active",
            )
        reopened = MonitoringHistoryStore(self.temporary.name)
        self.assertEqual(reopened.read_session("monitor_one")["status"], "stopped")

    def test_history_is_paginated(self):
        self.commit()
        sessions = self.store.list_sessions(page=1, page_size=1)
        cycles = self.store.list_cycles("monitor_one", page=1, page_size=1)
        self.assertEqual(sessions["page_size"], 1)
        self.assertEqual(cycles["total"], 1)
        with self.assertRaisesRegex(Exception, "safe limits"):
            self.store.list_sessions(page=1, page_size=101)

    def test_sessions_are_isolated(self):
        second = session()
        second["session_id"] = "monitor_two"
        second["display_name"] = "Second"
        self.store.create_session(second, recipe())
        self.commit()
        self.assertEqual(self.store.list_cycles("monitor_two")["total"], 0)


if __name__ == "__main__":
    unittest.main()
