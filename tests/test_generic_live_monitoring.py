import copy
import json
import unittest

from data_ingestion.catalogue.fetching import FetchLimits, assemble_time_series
from data_ingestion.catalogue.live_input import (
    LiveInputRecipeError,
    build_live_input_recipe,
    validate_live_input_recipe,
)
from data_ingestion.live_monitoring import (
    LiveMonitoringError,
    collect_recipe_window,
    merge_bounded_buffer,
    safe_session_projection,
    stopped_session,
    validate_resolved_live_model,
)
from learning_adaptation.catalogue_training import (
    CatalogueBundleError,
    _verify_live_input_recipe,
)


def limits():
    return FetchLimits(10, 20_000, 86_400, 1, 100_000, 100, 1_000_000, 100, 1, 1, 0)


def config():
    return {
        "CATALOGUE_MAX_QUERY_COUNT": 10,
        "CATALOGUE_MAX_QUERY_LENGTH": 20_000,
        "CATALOGUE_MAX_WINDOW_SECONDS": 86_400,
        "CATALOGUE_MIN_SAMPLING_INTERVAL_SECONDS": 1,
        "CATALOGUE_MAX_THEORETICAL_SAMPLES": 100_000,
        "CATALOGUE_MAX_SERIES": 100,
        "CATALOGUE_MAX_RESPONSE_BYTES": 1_000_000,
        "CATALOGUE_MAX_PREVIEW_ROWS": 100,
        "CATALOGUE_CONNECT_TIMEOUT_SECONDS": 1,
        "CATALOGUE_READ_TIMEOUT_SECONDS": 1,
        "CATALOGUE_MAX_RETRIES": 0,
        "CATALOGUE_PROMETHEUS_SOURCES": {"cluster-default": "http://prometheus.invalid"},
    }


def execution():
    return {
        "execution_id": "q0000-t0000",
        "query_id": "cpu",
        "display_name": "CPU",
        "feature_name": "cpu",
        "required": True,
        "execution_mode": "single",
        "target_type": None,
        "identity_labels": ["pod"],
        "resolved_target": None,
        "query_template": "sum by (pod) (rate(cpu[1m]))",
        "resolved_query": "sum by (pod) (rate(cpu[1m]))",
    }


def response(values=None, labels=None):
    return {
        "payload": {
            "status": "success",
            "data": {"result": [{"metric": labels or {"pod": "checkout"}, "values": values or [[100, "1"], [160, "2"]]}]},
        },
        "warnings": [],
        "response_bytes": 100,
    }


class FakeHTTPResponse:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self._payload = payload
        self.content = json.dumps(payload).encode()

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=65536):
        yield self.content

    def close(self):
        return None


class GenericLiveMonitoringTests(unittest.TestCase):
    def recipe(self):
        version = {"source": {"prometheus_source_id": "cluster-default", "start_time": "1970-01-01T00:01:40Z", "end_time": "1970-01-01T00:02:40Z", "sampling_interval_seconds": 60}}
        assembled = assemble_time_series(version, [{"execution": execution(), "response": response()}], limits())
        return build_live_input_recipe(version, [execution()], assembled), assembled

    def test_offline_and_live_feature_construction_are_equivalent(self):
        recipe, offline = self.recipe()
        payload = response()["payload"]
        live = collect_recipe_window(recipe, 100, 160, config(), request_get=lambda *args, **kwargs: FakeHTTPResponse(payload))
        self.assertEqual(live["feature_order"], offline["features"])
        self.assertEqual(live["rows"], offline["rows"])
        self.assertEqual(live["timestamps"], offline["timestamps"])

    def test_recipe_and_series_identity_mismatches_are_rejected(self):
        recipe, _offline = self.recipe()
        changed = copy.deepcopy(recipe)
        changed["sampling_interval_seconds"] = 30
        with self.assertRaisesRegex(LiveInputRecipeError, "checksum"):
            validate_live_input_recipe(changed)
        payload = response(labels={"pod": "other"})["payload"]
        with self.assertRaisesRegex(LiveMonitoringError, "feature order|series identity"):
            collect_recipe_window(recipe, 100, 160, config(), request_get=lambda *args, **kwargs: FakeHTTPResponse(payload))

    def test_training_bundle_recipe_hash_mismatch_is_rejected(self):
        recipe, _offline = self.recipe()
        manifest = {
            "live_input_recipe": recipe,
            "live_input_recipe_sha256": "0" * 64,
        }
        with self.assertRaisesRegex(CatalogueBundleError, "checksum"):
            _verify_live_input_recipe(manifest, recipe["feature_order"])

    def test_session_start_requires_matching_artifact_recipe_and_features(self):
        recipe, _offline = self.recipe()
        resolved = {
            "model_id": "run_one-detector",
            "artifact_manifest_sha256": "a" * 64,
            "feature_identity": {
                "feature_order": recipe["feature_order"],
                "feature_order_sha256": recipe["feature_order_sha256"],
            },
            "recipe": recipe,
        }
        self.assertEqual(
            validate_resolved_live_model(resolved, "run_one-detector")["recipe"]["recipe_sha256"],
            recipe["recipe_sha256"],
        )
        changed = copy.deepcopy(resolved)
        changed["feature_identity"]["feature_order_sha256"] = "b" * 64
        with self.assertRaisesRegex(LiveMonitoringError, "feature-order checksum"):
            validate_resolved_live_model(changed, "run_one-detector")

    def test_missing_live_values_are_not_imputed(self):
        recipe, _offline = self.recipe()
        payload = response(values=[[100, "1"]])["payload"]
        with self.assertRaisesRegex(LiveMonitoringError, "missing values"):
            collect_recipe_window(recipe, 100, 160, config(), request_get=lambda *args, **kwargs: FakeHTTPResponse(payload))

    def test_buffers_are_bounded_deduplicated_and_session_local(self):
        first = merge_bounded_buffer([], ["a", "b"], [[1.0], [2.0]], 2)
        second = merge_bounded_buffer(first, ["b", "c"], [[20.0], [3.0]], 2)
        other = merge_bounded_buffer([], ["x"], [[9.0]], 2)
        self.assertEqual(second, [{"timestamp": "b", "values": [20.0]}, {"timestamp": "c", "values": [3.0]}])
        self.assertEqual(other, [{"timestamp": "x", "values": [9.0]}])

    def test_public_session_hides_recipe_buffer_and_task_ids(self):
        public = safe_session_projection({"session_id": "monitor_1", "model_id": "model_1", "status": "warmup", "warmup_observations": 5, "recipe": {"resolved_query": "secret"}, "buffer": [[1]], "current_task_id": "task-secret"})
        self.assertEqual(public["warmup_observations"], 5)
        self.assertNotIn("recipe", public)
        self.assertNotIn("buffer", public)
        self.assertNotIn("current_task_id", public)

    def test_stop_is_exact_and_clears_only_that_session_buffer(self):
        revoked = []
        session = stopped_session(
            {"session_id": "monitor-one", "status": "active", "current_task_id": "task-one", "next_task_id": "task-two", "buffer": [{"timestamp": "a", "values": [1]}]},
            revoked.append,
            stopped_at="2026-10-09T12:00:00Z",
        )
        self.assertEqual(revoked, ["task-one", "task-two"])
        self.assertEqual(session["status"], "stopped")
        self.assertNotIn("buffer", session)
        self.assertEqual(session["session_id"], "monitor-one")


if __name__ == "__main__":
    unittest.main()
