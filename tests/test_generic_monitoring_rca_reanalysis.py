from copy import deepcopy
import tempfile
import unittest

from data_ingestion.monitoring_history import (
    MonitoringHistoryError,
    MonitoringHistoryStore,
    canonical_sha256,
)
from data_ingestion.monitoring_rca import (
    MonitoringRcaError,
    analysis_input_identity,
    assess_recipe_compatibility,
    decide_trigger,
    evaluate_reanalysis,
    initial_rca_state,
    mark_triggered,
    validate_rca_configuration,
)


def recipe():
    return {
        "schema_version": 1,
        "contract": "cloudsentinel.live-input/v1",
        "sampling_interval_seconds": 60,
        "alignment": {
            "grid": "inclusive_utc_range",
            "timestamp_match": "exact_microsecond",
            "off_grid_samples": "drop_with_warning",
        },
        "missing_data": {
            "offline_representation": "empty",
            "live_policy": "reject_window",
            "imputation": "none",
        },
        "feature_order": ["cpu|pod=checkout", "memory|pod=checkout"],
        "feature_order_sha256": "f" * 64,
        "query_executions": [
            {"execution_id": "cpu", "feature_name": "cpu_usage", "identity_labels": []},
            {"execution_id": "memory", "feature_name": "memory_usage", "identity_labels": []},
        ],
        "series_mapping": [
            {"feature_id": "cpu|pod=checkout", "execution_id": "cpu", "metric_labels": {"pod": "checkout"}},
            {"feature_id": "memory|pod=checkout", "execution_id": "memory", "metric_labels": {"pod": "checkout"}},
        ],
    }


def cycle(percentage=100.0, count=20, at="2026-10-13T10:00:00Z"):
    return {
        "prediction_count": count,
        "anomaly_percentage": percentage,
        "completed_at": at,
    }


def selected(*timestamps, source="input:sha256:first"):
    return {
        "contract": "cloudsentinel.crca-input/v1",
        "source_cycle_input_reference": source,
        "analysis_window": {"start": timestamps[0], "end": timestamps[-1], "requested_seconds": 600},
        "feature_order": ["cpu|pod=checkout", "memory|pod=checkout"],
        "feature_order_sha256": "f" * 64,
        "feature_metadata_sha256": "m" * 64,
        "timestamps": list(timestamps),
        "matrix": [[float(index), float(index + 1)] for index, _ in enumerate(timestamps)],
    }


class ReanalysisPolicyTests(unittest.TestCase):
    def state(self, **values):
        configuration = validate_rca_configuration({"enabled": True, **values})
        return initial_rca_state(
            configuration,
            assess_recipe_compatibility(recipe()),
            {"minimum_observations": 2, "minimum_features": 2},
        )

    def test_defaults_preserve_one_run_policy(self):
        configuration = validate_rca_configuration({"enabled": True})
        self.assertFalse(configuration["reanalyze_within_episode"])
        self.assertEqual(configuration["reanalysis_min_interval_seconds"], 900)
        self.assertEqual(configuration["reanalysis_min_new_observations"], 20)
        self.assertEqual(configuration["max_attempts_per_episode"], 3)
        state = mark_triggered(self.state(), "rca_first", "2026-10-13T10:00:00Z")
        _, decision = decide_trigger(state, cycle(at="2026-10-13T10:05:00Z"))
        self.assertEqual(decision["status"], "suppressed_episode")
        self.assertEqual(decision["reason_code"], "episode_already_analyzed")
        self.assertEqual(decision["related_run_id"], "rca_first")

    def test_configuration_rejects_boolean_integer_and_out_of_range_values(self):
        invalid = (
            {"reanalyze_within_episode": 1},
            {"reanalysis_min_interval_seconds": True},
            {"reanalysis_min_interval_seconds": 59},
            {"reanalysis_min_new_observations": 0},
            {"max_attempts_per_episode": 0},
            {"max_attempts_per_episode": 21},
        )
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(MonitoringRcaError):
                validate_rca_configuration(payload)

    def test_reanalysis_requires_interval_and_unique_new_timestamps(self):
        state = self.state(
            reanalyze_within_episode=True,
            reanalysis_min_interval_seconds=900,
            reanalysis_min_new_observations=2,
        )
        first = selected("2026-10-13T09:59:00Z", "2026-10-13T10:00:00Z")
        state = mark_triggered(
            state,
            "rca_first",
            "2026-10-13T10:00:01Z",
            intent_at="2026-10-13T10:00:02Z",
            analysis_end="2026-10-13T10:00:00Z",
            input_sha256=analysis_input_identity(first),
        )
        later = selected(
            "2026-10-13T09:59:00Z",
            "2026-10-13T10:00:00Z",
            "2026-10-13T10:01:00Z",
            "2026-10-13T10:02:00Z",
        )
        too_soon = evaluate_reanalysis(
            state,
            later,
            analysis_input_identity(later),
            decision_time="2026-10-13T10:14:59Z",
        )
        self.assertEqual(too_soon["status"], "suppressed_interval")
        ready = evaluate_reanalysis(
            state,
            later,
            analysis_input_identity(later),
            decision_time="2026-10-13T10:15:02Z",
        )
        self.assertEqual(ready["status"], "reanalysis")
        self.assertEqual(ready["new_observation_count"], 2)

        only_late_and_overlap = selected(
            "2026-10-13T09:58:00Z",
            "2026-10-13T09:59:00Z",
            "2026-10-13T10:00:00Z",
            "2026-10-13T10:01:00Z",
        )
        insufficient = evaluate_reanalysis(
            state,
            only_late_and_overlap,
            analysis_input_identity(only_late_and_overlap),
            decision_time="2026-10-13T10:15:02Z",
        )
        self.assertEqual(insufficient["status"], "suppressed_new_observations")
        self.assertEqual(insufficient["new_observation_count"], 1)

    def test_budget_includes_failed_accepted_runs_but_not_dispatch_retry(self):
        state = self.state(
            reanalyze_within_episode=True,
            reanalysis_min_interval_seconds=60,
            reanalysis_min_new_observations=1,
            max_attempts_per_episode=2,
        )
        state = mark_triggered(
            state,
            "rca_first",
            "2026-10-13T10:00:00Z",
            intent_at="2026-10-13T10:00:00Z",
            analysis_end="2026-10-13T10:00:00Z",
            input_sha256="a" * 64,
        )
        # A worker failure or dispatch retry does not mutate policy state or consume again.
        self.assertEqual(state["attempt_count"], 1)
        state = mark_triggered(
            state,
            "rca_second",
            "2026-10-13T10:02:00Z",
            intent_at="2026-10-13T10:02:00Z",
            analysis_end="2026-10-13T10:02:00Z",
            input_sha256="b" * 64,
        )
        blocked = evaluate_reanalysis(
            state,
            selected("2026-10-13T10:03:00Z"),
            "c" * 64,
            decision_time="2026-10-13T11:00:00Z",
        )
        self.assertEqual(blocked["status"], "suppressed_budget")
        self.assertEqual(state["attempt_count"], 2)

    def test_identical_content_ignores_cycle_artifact_reference(self):
        first = selected("2026-10-13T10:00:00Z", "2026-10-13T10:01:00Z", source="input:sha256:first")
        retry = deepcopy(first)
        retry["source_cycle_input_reference"] = "input:sha256:second"
        self.assertEqual(analysis_input_identity(first), analysis_input_identity(retry))

    def test_no_data_does_not_recover_and_valid_recovery_resets_budget(self):
        state = mark_triggered(self.state(reanalyze_within_episode=True), "rca_first", "2026-10-13T10:00:00Z")
        state, no_data = decide_trigger(state, cycle(None, 0, "2026-10-13T10:05:00Z"))
        self.assertEqual(no_data["status"], "no_new_data")
        self.assertEqual(state["attempt_count"], 1)
        state, recovered = decide_trigger(state, cycle(20, 20, "2026-10-13T11:01:00Z"))
        self.assertEqual(recovered["status"], "recovered")
        self.assertEqual(state["attempt_count"], 0)
        self.assertIsNone(state["last_run_id"])

    def test_preflight_skip_state_can_retry_without_consuming_budget(self):
        state = self.state()
        unchanged, first = decide_trigger(state, cycle())
        self.assertEqual(first["status"], "trigger")
        # A preflight selection failure is recorded without mark_triggered.
        unchanged, later = decide_trigger(unchanged, cycle(at="2026-10-13T10:05:00Z"))
        self.assertEqual(later["status"], "trigger")
        self.assertEqual(unchanged["attempt_count"], 0)


class ReanalysisPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = MonitoringHistoryStore(self.temporary.name)
        value = recipe()
        value["recipe_sha256"] = canonical_sha256(value)
        state = initial_rca_state(
            validate_rca_configuration(
                {
                    "enabled": True,
                    "reanalyze_within_episode": True,
                    "reanalysis_min_interval_seconds": 900,
                    "reanalysis_min_new_observations": 20,
                    "max_attempts_per_episode": 3,
                }
            ),
            assess_recipe_compatibility(value),
            {},
        )
        self.store.create_session(
            {
                "session_id": "monitor_reanalysis",
                "model_id": "model_one",
                "status": "active",
                "created_at": "2026-10-13T10:00:00Z",
                "rca": state,
            },
            value,
        )
        for number in (1, 2):
            self.store.commit_cycle(
                "monitor_reanalysis",
                f"cycle_{number}",
                input_payload={
                    "feature_order": value["feature_order"],
                    "timestamps": [f"2026-10-13T10:0{number}:00Z"],
                    "matrix": [[1.0, 2.0]],
                },
                result_payload={"binary_predictions": [1], "anomaly_scores": [1.0]},
                cycle={
                    "status": "completed",
                    "completed_at": f"2026-10-13T10:0{number}:01Z",
                    "prediction_count": 1,
                    "anomaly_count": 1,
                    "anomaly_percentage": 100.0,
                },
                next_watermark=f"2026-10-13T10:0{number}:00Z",
                session_status="active",
            )

    def test_policy_state_and_run_provenance_survive_reopen(self):
        current = self.store.read_session("monitor_reanalysis")["rca"]
        updated = mark_triggered(
            current,
            "rca_first",
            "2026-10-13T10:01:01Z",
            intent_at="2026-10-13T10:01:02Z",
            analysis_end="2026-10-13T10:01:00Z",
            input_sha256="a" * 64,
        )
        self.store.create_rca_run(
            "monitor_reanalysis",
            "cycle_1",
            {
                "run_id": "rca_first",
                "status": "queued",
                "created_at": "2026-10-13T10:01:02Z",
                "analysis_kind": "first_analysis",
                "execution_attempt_number": 1,
                "episode_attempt_budget": 3,
                "previous_run_id": None,
                "analysis_input_sha256": "a" * 64,
                "policy_version": updated["policy_version"],
            },
            input_payload={"matrix": [[1.0, 2.0]], "timestamps": ["2026-10-13T10:01:00Z"], "feature_order": recipe()["feature_order"]},
            event={"status": "started", "analysis_kind": "first_analysis"},
            rca_state=updated,
        )
        reopened = MonitoringHistoryStore(self.temporary.name)
        session = reopened.read_session("monitor_reanalysis")
        run = reopened.list_rca_runs("monitor_reanalysis")["items"][0]
        self.assertEqual(session["rca"]["attempt_count"], 1)
        self.assertEqual(run["analysis_kind"], "first_analysis")
        self.assertEqual(run["analysis_input_sha256"], "a" * 64)
        self.assertNotIn("input_reference", run)

    def test_atomic_reservation_rejects_stale_or_parallel_intent(self):
        current = self.store.read_session("monitor_reanalysis")["rca"]
        updated = mark_triggered(current, "rca_first", "2026-10-13T10:01:01Z")
        self.store.create_rca_run(
            "monitor_reanalysis",
            "cycle_1",
            {"run_id": "rca_first", "status": "queued", "created_at": "2026-10-13T10:01:02Z", "execution_attempt_number": 1, "previous_run_id": None},
            input_payload={"matrix": [[1.0, 2.0]], "timestamps": ["2026-10-13T10:01:00Z"], "feature_order": recipe()["feature_order"]},
            event={"status": "started"},
            rca_state=updated,
        )
        with self.assertRaises(MonitoringHistoryError):
            self.store.create_rca_run(
                "monitor_reanalysis",
                "cycle_2",
                {"run_id": "rca_parallel", "status": "queued", "created_at": "2026-10-13T10:02:02Z", "execution_attempt_number": 1, "previous_run_id": None},
                input_payload={"matrix": [[2.0, 3.0]], "timestamps": ["2026-10-13T10:02:00Z"], "feature_order": recipe()["feature_order"]},
                event={"status": "reanalysis_started"},
                rca_state=updated,
            )

    def test_idempotent_run_retry_does_not_roll_back_newer_policy_state(self):
        current = self.store.read_session("monitor_reanalysis")["rca"]
        first_state = mark_triggered(current, "rca_first", "2026-10-13T10:01:01Z")
        run = {"run_id": "rca_first", "status": "queued", "created_at": "2026-10-13T10:01:02Z", "execution_attempt_number": 1, "previous_run_id": None}
        payload = {"matrix": [[1.0, 2.0]], "timestamps": ["2026-10-13T10:01:00Z"], "feature_order": recipe()["feature_order"]}
        self.store.create_rca_run(
            "monitor_reanalysis", "cycle_1", run,
            input_payload=payload, event={"status": "started"}, rca_state=first_state,
        )
        newer = dict(first_state, attempt_count=2, last_run_id="rca_second")
        self.store.update_session(
            "monitor_reanalysis", lambda record: record.update(rca=newer)
        )
        self.store.create_rca_run(
            "monitor_reanalysis", "cycle_1", run,
            input_payload={"changed": True}, event={"status": "started"}, rca_state=first_state,
        )
        self.assertEqual(
            self.store.read_session("monitor_reanalysis")["rca"]["last_run_id"],
            "rca_second",
        )


if __name__ == "__main__":
    unittest.main()
