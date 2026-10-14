import importlib
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

from anomaly_detection.crca.baro_adapter import (
    BARO_INPUT_CONTRACT,
    BARO_METHOD_ID,
    BARO_PREPROCESSING_CONTRACT,
    BaroAdapterError,
    BaroRobustScorerAdapter,
    capability,
    configuration_sha256,
)
from data_ingestion.monitoring_rca import (
    MonitoringRcaError,
    analysis_input_identity,
    attach_detected_change_point,
)


ROOT = Path(__file__).resolve().parents[1]


def metadata():
    return [
        {
            "column_index": 0,
            "feature_id": "cpu_usage|pod=checkout",
            "feature_name": "cpu_usage|pod=checkout",
            "metric_name": "cpu_usage",
            "entity": {"type": "pod", "id": "checkout", "source": "prometheus_series_label"},
        },
        {
            "column_index": 1,
            "feature_id": "mem_usage|pod=cart",
            "feature_name": "mem_usage|pod=cart",
            "metric_name": "mem_usage",
            "entity": {"type": "pod", "id": "cart", "source": "prometheus_series_label"},
        },
    ]


def selected_input():
    timestamps = [f"2026-10-14T10:0{minute}:00Z" for minute in range(5)]
    return {
        "schema_version": 1,
        "contract": "cloudsentinel.crca-input/v1",
        "analysis_window": {"start": timestamps[0], "end": timestamps[-1]},
        "feature_order": [item["feature_id"] for item in metadata()],
        "feature_order_sha256": "f" * 64,
        "feature_metadata": metadata(),
        "feature_metadata_sha256": "e" * 64,
        "timestamps": timestamps,
        "matrix": [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [20.0, 12.5], [24.0, 13.0]],
    }


def cycle_result(predictions=None):
    timestamps = selected_input()["timestamps"]
    return {
        "prediction_timestamps": timestamps,
        "binary_predictions": predictions if predictions is not None else [0, 0, 1, 1, 1],
        "anomaly_scores": [0.0, 0.1, 2.0, 3.0, 4.0],
    }


def official_like_robust_scorer(frame, anomalies, **_kwargs):
    index = anomalies[0]
    scores = {}
    for column in frame.columns:
        baseline = frame.iloc[:index][[column]]
        anomalous = frame.iloc[index:][[column]]
        if baseline[column].nunique() <= 1 or anomalous[column].nunique() <= 1:
            continue
        transformed = RobustScaler().fit(baseline).transform(anomalous)[:, 0]
        scores[column] = float(max(transformed))
    return {"node_names": list(scores), "ranks": sorted(scores, key=scores.get, reverse=True)}


class BaroChangePointTests(unittest.TestCase):
    def attach(self, selected=None, result=None, **kwargs):
        return attach_detected_change_point(
            selected or selected_input(),
            result or cycle_result(),
            cycle_id="cycle_one",
            cycle_result_sha256="a" * 64,
            **kwargs,
        )

    def test_first_positive_prediction_is_pinned_as_change_point(self):
        value = self.attach()
        evidence = value["change_point"]
        self.assertEqual(value["contract"], BARO_INPUT_CONTRACT)
        self.assertEqual(evidence["detected_change_point"], "2026-10-14T10:02:00Z")
        self.assertEqual(evidence["index"], 2)
        self.assertEqual(evidence["baseline_observation_count"], 2)
        self.assertEqual(evidence["anomalous_observation_count"], 3)
        self.assertEqual(evidence["source"], "stored_monitoring_cycle_result")

    def test_no_positive_and_insufficient_segments_are_targeted_skips(self):
        with self.assertRaises(MonitoringRcaError) as no_positive:
            self.attach(result=cycle_result([0, 0, 0, 0, 0]))
        self.assertEqual(no_positive.exception.reason_code, "baro_change_point_unavailable")
        with self.assertRaises(MonitoringRcaError) as baseline:
            self.attach(result=cycle_result([0, 1, 1, 1, 1]))
        self.assertEqual(baseline.exception.reason_code, "insufficient_baro_baseline")
        with self.assertRaises(MonitoringRcaError) as post:
            self.attach(result=cycle_result([0, 0, 0, 0, 1]))
        self.assertEqual(post.exception.reason_code, "insufficient_baro_post_change")

    def test_change_point_is_part_of_analysis_identity(self):
        first = self.attach()
        later = self.attach(result=cycle_result([0, 0, 0, 1, 1]))
        self.assertNotEqual(analysis_input_identity(first), analysis_input_identity(later))


class BaroAdapterTests(unittest.TestCase):
    def payload(self):
        return attach_detected_change_point(
            selected_input(), cycle_result(), cycle_id="cycle_one", cycle_result_sha256="a" * 64
        )

    def test_synthetic_shift_ranks_changed_feature_first_and_preserves_identity(self):
        result = BaroRobustScorerAdapter(official_like_robust_scorer).run(self.payload())
        self.assertEqual(result["method"], BARO_METHOD_ID)
        self.assertEqual(result["ranking"][0]["feature_id"], "cpu_usage|pod=checkout")
        self.assertEqual(result["ranking"][0]["feature_name"], "cpu_usage|pod=checkout")
        self.assertEqual(result["ranking"][0]["entity_id"], "checkout")
        self.assertEqual(result["graphs"], [])
        self.assertGreater(result["ranking"][0]["score"], result["ranking"][1]["score"])

    @unittest.skipUnless(
        capability().get("status") == "available",
        "fse-baro 0.2.2 is not installed in the local test interpreter",
    )
    def test_official_fse_baro_022_interface_on_synthetic_shift(self):
        result = BaroRobustScorerAdapter().run(self.payload())
        self.assertEqual(result["ranking"][0]["feature_id"], "cpu_usage|pod=checkout")
        self.assertEqual(result["method"], BARO_METHOD_ID)

    def test_adapter_passes_raw_values_and_official_change_index(self):
        observed = {}

        def scorer(frame, anomalies, **kwargs):
            observed["matrix"] = frame.to_numpy().tolist()
            observed["anomalies"] = anomalies
            return official_like_robust_scorer(frame, anomalies, **kwargs)

        BaroRobustScorerAdapter(scorer).run(self.payload())
        self.assertEqual(observed["matrix"], selected_input()["matrix"])
        self.assertEqual(observed["anomalies"], [2])

    def test_non_finite_input_and_invalid_rankings_are_rejected(self):
        payload = self.payload()
        payload["matrix"][0][0] = math.inf
        with self.assertRaises(BaroAdapterError):
            BaroRobustScorerAdapter(official_like_robust_scorer).run(payload)
        with self.assertRaises(BaroAdapterError):
            BaroRobustScorerAdapter(
                lambda *_args, **_kwargs: {"ranks": ["unknown_feature"]}
            ).run(self.payload())
        with self.assertRaises(BaroAdapterError):
            BaroRobustScorerAdapter(
                lambda *_args, **_kwargs: {"ranks": ["cs_feature_0000", "cs_feature_0000"]}
            ).run(self.payload())

        class NonFiniteScaler:
            def fit(self, _values):
                return self

            def transform(self, values):
                return np.full((len(values), 1), math.inf)

        with (
            patch("anomaly_detection.crca.baro_adapter.RobustScaler", return_value=NonFiniteScaler()),
            self.assertRaises(BaroAdapterError),
        ):
            BaroRobustScorerAdapter(official_like_robust_scorer).run(self.payload())

    def test_method_specific_preprocessing_contracts_remain_independent(self):
        tasks_source = (ROOT / "anomaly_detection/crca/tasks.py").read_text()
        self.assertIn("MinMaxScaler().fit_transform", tasks_source)
        self.assertIn("BARO_PREPROCESSING_CONTRACT", tasks_source)
        self.assertIn('"input": "raw_finite_numeric"', tasks_source)
        self.assertEqual(BARO_PREPROCESSING_CONTRACT, "cloudsentinel.baro-robust-scorer-raw/v1")

    def test_worker_failure_is_terminal_for_run_only(self):
        original_modules = set(sys.modules)
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

        class Store:
            failed = None

            def claim_rca_run(self, _session, _run):
                return {
                    "method": BARO_METHOD_ID,
                    "algorithm": {"id": BARO_METHOD_ID, "configuration_sha256": configuration_sha256()},
                    "input_reference": "rca_input/sha256_deadbeef",
                }

            def fail_rca_run(self, session_id, run_id, message):
                self.failed = (session_id, run_id, message)

            def read_artifact(self, _reference):
                return self_input

        self_input = self.payload()
        store = Store()
        try:
            with (
                patch.object(tasks, "_history_store", return_value=store),
                patch.object(tasks, "BaroRobustScorerAdapter", side_effect=RuntimeError("private detail")),
            ):
                result = tasks.run_monitoring_rca_task.run("monitor_one", "rca_one", BARO_METHOD_ID)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(store.failed, ("monitor_one", "rca_one", "RCA execution failed."))
        finally:
            for module_name in set(sys.modules) - original_modules:
                sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
