import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from detectors.contracts import DetectorAdapter
from detectors.registry import discover_detectors, get_adapter


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DETECTOR_ROOT = REPOSITORY_ROOT / "detectors"


def synthetic_data():
    rng = np.random.default_rng(42)
    train = rng.normal(0.0, 0.25, size=(80, 2))
    normal_test = rng.normal(0.0, 0.25, size=(20, 2))
    anomalous_test = rng.normal(8.0, 0.1, size=(5, 2))
    test = np.vstack((normal_test, anomalous_test))
    labels = np.concatenate((np.zeros(20, dtype=int), np.ones(5, dtype=int)))
    return train, test, labels


class IsolationForestDiscoveryTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("detectors.isolation_forest.adapter", None)

    def test_manifest_is_discovered_alongside_cgnn(self):
        manifests = discover_detectors(DETECTOR_ROOT)
        self.assertEqual(
            [manifest["id"] for manifest in manifests],
            ["cgnn", "isolation-forest"],
        )
        manifest = next(item for item in manifests if item["id"] == "isolation-forest")
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["supported_modalities"], ["metrics"])
        self.assertEqual(manifest["training_parameters"]["contamination"]["default"], "auto")

    def test_registry_and_endpoint_stay_free_of_isolation_forest_runtime_imports(self):
        script = f"""
import sys
from flask import Flask
from detectors.api import create_detectors_blueprint
from detectors.registry import discover_detectors, get_adapter

root = {str(DETECTOR_ROOT)!r}
discover_detectors(root)
app = Flask(__name__)
app.register_blueprint(create_detectors_blueprint(root))
assert app.test_client().get('/detectors').status_code == 200
assert 'detectors.isolation_forest.adapter' not in sys.modules
assert 'detectors.isolation_forest.implementation' not in sys.modules
assert 'sklearn' not in sys.modules
assert 'joblib' not in sys.modules
adapter = get_adapter('isolation-forest', root)
assert type(adapter).__name__ == 'IsolationForestAdapter'
assert 'detectors.isolation_forest.adapter' in sys.modules
assert 'detectors.isolation_forest.implementation' not in sys.modules
assert 'sklearn' not in sys.modules
assert 'joblib' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_get_adapter_returns_contract_compatible_instance(self):
        adapter = get_adapter("isolation-forest", DETECTOR_ROOT)
        self.assertEqual(type(adapter).__name__, "IsolationForestAdapter")
        self.assertNotIsInstance(adapter, type)
        self.assertIsInstance(adapter, DetectorAdapter)

    def test_adapter_methods_load_implementation_lazily(self):
        adapter_module = importlib.import_module("detectors.isolation_forest.adapter")
        implementation = types.SimpleNamespace(
            train_model=Mock(return_value={"trained": True}),
            evaluate_model=Mock(return_value={"f1": 1.0}),
            predict_with_model=Mock(return_value={"anomaly_percentage": 0.0}),
        )
        with patch.object(adapter_module, "import_module", return_value=implementation) as loader:
            adapter = adapter_module.IsolationForestAdapter()
            self.assertEqual(adapter.train("train", "artifacts"), {"trained": True})
            self.assertEqual(adapter.evaluate("test", "labels", "artifacts"), {"f1": 1.0})
            self.assertEqual(
                adapter.predict("test", "artifacts"), {"anomaly_percentage": 0.0}
            )
        self.assertEqual(loader.call_count, 3)
        loader.assert_called_with("detectors.isolation_forest.implementation")


class IsolationForestRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = get_adapter("isolation-forest", DETECTOR_ROOT)
        self.train, self.test, self.labels = synthetic_data()

    def test_real_training_evaluation_prediction_and_joblib_roundtrip(self):
        import joblib

        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            metadata = self.adapter.train(
                self.train,
                artifact_dir,
                training_parameters={"n_estimators": 25, "random_state": 42},
                model_id="test-model",
            )

            self.assertTrue((artifact_dir / "model.joblib").is_file())
            self.assertTrue((artifact_dir / "model_metadata.json").is_file())
            stored_metadata = json.loads(
                (artifact_dir / "model_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata, stored_metadata)
            self.assertEqual(metadata["model_id"], "test-model")
            self.assertEqual(metadata["n_features"], 2)

            first_prediction = self.adapter.predict(self.test, artifact_dir)
            pipeline = joblib.load(artifact_dir / "model.joblib")
            roundtrip_path = artifact_dir / "roundtrip.joblib"
            joblib.dump(pipeline, roundtrip_path)
            reloaded = joblib.load(roundtrip_path)
            np.testing.assert_array_equal(
                pipeline.predict(self.test), reloaded.predict(self.test)
            )

            evaluation = self.adapter.evaluate(self.test, self.labels.reshape(-1, 1), artifact_dir)
            self.assertTrue((artifact_dir / "model_evaluation.json").is_file())
            self.assertEqual(evaluation["test_observations"], 25)
            self.assertEqual(evaluation["actual_anomalies"], 5)
            self.assertEqual(
                evaluation["predicted_anomalies"],
                sum(first_prediction["binary_predictions"]),
            )
            predictions = np.asarray(first_prediction["binary_predictions"])
            expected_counts = {
                "true_positive": int(np.sum((self.labels == 1) & (predictions == 1))),
                "true_negative": int(np.sum((self.labels == 0) & (predictions == 0))),
                "false_positive": int(np.sum((self.labels == 0) & (predictions == 1))),
                "false_negative": int(np.sum((self.labels == 1) & (predictions == 0))),
            }
            for name, expected in expected_counts.items():
                self.assertEqual(evaluation[name], expected)
            self.assertEqual(evaluation["true_positive"], 5)
            self.assertEqual(evaluation["false_negative"], 0)
            self.assertAlmostEqual(
                evaluation["precision"],
                evaluation["true_positive"]
                / (evaluation["true_positive"] + evaluation["false_positive"]),
            )
            self.assertAlmostEqual(evaluation["recall"], 1.0)
            expected_f1 = (
                2
                * evaluation["precision"]
                * evaluation["recall"]
                / (evaluation["precision"] + evaluation["recall"])
            )
            self.assertAlmostEqual(evaluation["f1"], expected_f1)
            self.assertEqual(
                evaluation["true_positive"]
                + evaluation["true_negative"]
                + evaluation["false_positive"]
                + evaluation["false_negative"],
                25,
            )
            self.assertGreaterEqual(first_prediction["anomaly_percentage"], 0.0)
            self.assertLessEqual(first_prediction["anomaly_percentage"], 100.0)
            self.assertEqual(len(first_prediction["anomaly_scores"]), 25)

    def test_sklearn_minus_one_predictions_map_to_binary_anomalies(self):
        implementation = importlib.import_module("detectors.isolation_forest.implementation")
        model = types.SimpleNamespace(
            predict=Mock(return_value=np.array([1, -1, 1, -1])),
            decision_function=Mock(return_value=np.array([0.4, -0.2, 0.1, -0.8])),
        )
        predictions, scores = implementation._predictions(model, np.ones((4, 2)))
        np.testing.assert_array_equal(predictions, np.array([0, 1, 0, 1]))
        np.testing.assert_allclose(scores, np.array([-0.4, 0.2, -0.1, 0.8]))

    def test_invalid_inputs_raise_targeted_domain_errors(self):
        implementation = importlib.import_module("detectors.isolation_forest.implementation")
        error = implementation.IsolationForestInputError
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            with self.assertRaisesRegex(error, "two-dimensional"):
                self.adapter.train(np.array([1.0, 2.0]), artifact_dir)
            with self.assertRaisesRegex(error, "more than one observation"):
                self.adapter.train(np.array([[1.0, 2.0]]), artifact_dir)
            with self.assertRaisesRegex(error, "finite"):
                self.adapter.train(np.array([[1.0, np.nan], [2.0, 3.0]]), artifact_dir)
            with self.assertRaisesRegex(error, "numeric values without coercion"):
                self.adapter.train(np.array([["1", "2"], ["3", "4"]]), artifact_dir)

            self.adapter.train(self.train, artifact_dir, training_parameters={"n_estimators": 10})
            with self.assertRaisesRegex(error, "stored model expects 2"):
                self.adapter.predict(np.ones((3, 3)), artifact_dir)
            with self.assertRaisesRegex(error, "same number of observations"):
                self.adapter.evaluate(self.test, self.labels[:-1], artifact_dir)
            invalid_labels = self.labels.copy()
            invalid_labels[-1] = 2
            with self.assertRaisesRegex(error, "only 0 and 1"):
                self.adapter.evaluate(self.test, invalid_labels, artifact_dir)
            with self.assertRaisesRegex(error, "one-dimensional or a single-row"):
                self.adapter.evaluate(self.test, np.zeros((5, 5)), artifact_dir)
            with self.assertRaisesRegex(error, "contamination.*'auto'"):
                self.adapter.train(
                    self.train,
                    artifact_dir,
                    training_parameters={"contamination": 0.1},
                )


if __name__ == "__main__":
    unittest.main()
