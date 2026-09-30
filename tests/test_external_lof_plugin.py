import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from flask import Flask
import joblib
import numpy as np

from detectors import registry
from detectors.contracts import (
    DetectorCompatibilityError,
    TrainingContext,
    TrainingData,
)
from detectors.plugin_repository import PluginRepository
from learning_adaptation.model_catalogue import ModelCatalogueStore
from learning_adaptation.plugin_training import execute_detector_plugin_training
from learning_adaptation.training_runs_api import (
    configure_training_run_defaults,
    create_training_runs_blueprint,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOF_SOURCE = REPOSITORY_ROOT / "examples" / "external_plugins" / "local_outlier_factor"
LOF_ID = "local-outlier-factor"


class ExternalLocalOutlierFactorPluginTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = PluginRepository((self.root / "plugin-repository").resolve())
        self.release = self.repository.install(LOF_SOURCE)
        self.repository.activate(LOF_ID, "1.0.0")
        self._clear_external_modules()

        self.dataset_root = self.root / "datasets"
        self.snapshot_root = self.root / "snapshots"
        self.run_root = self.root / "runs"
        self.model_root = self.root / "models"
        self.artifact_root = self.root / "artifacts"
        self.published_artifact_root = self.root / "published-model-artifacts"
        self.train, self.test, self.labels = self._write_dataset()

        self.task = Mock()
        self.task.apply_async.side_effect = lambda *, args: types.SimpleNamespace(
            id=f"generic-task-{args[0]['detector_id']}"
        )
        self.app = Flask(__name__)
        configure_training_run_defaults(self.app)
        self.app.config.update(
            TESTING=True,
            EXISTING_DATASETS_ROOT=str(self.dataset_root),
            DATASET_SNAPSHOT_STORAGE_ROOT=str(self.snapshot_root),
            TRAINING_RUN_STORAGE_ROOT=str(self.run_root),
            MODEL_CATALOGUE_STORAGE_ROOT=str(self.model_root),
        )
        self.app.register_blueprint(
            create_training_runs_blueprint(self.task, status_reader=lambda _task_id: None)
        )
        self.environment = patch.dict(
            os.environ,
            {
                "DETECTOR_PLUGIN_ROOTS": str(self.repository.active),
                "DATASET_SNAPSHOT_STORAGE_ROOT": str(self.snapshot_root),
                "TRAINING_RUN_STORAGE_ROOT": str(self.run_root),
                "MODEL_CATALOGUE_STORAGE_ROOT": str(self.model_root),
                "TRAINED_MODELS_TEMP_ROOT": str(self.artifact_root),
                "MODEL_ARTIFACT_STORAGE_ROOT": str(self.published_artifact_root),
            },
            clear=False,
        )
        self.environment.start()
        self.client = self.app.test_client()

    def tearDown(self):
        self.environment.stop()
        self._clear_external_modules()
        for current_root, directory_names, file_names in os.walk(
            self.root, topdown=False, followlinks=False
        ):
            for name in file_names:
                path = Path(current_root) / name
                if not path.is_symlink():
                    path.chmod(0o600)
            for name in directory_names:
                path = Path(current_root) / name
                if not path.is_symlink():
                    path.chmod(0o700)
        self.temporary.cleanup()

    @staticmethod
    def _clear_external_modules():
        for name in tuple(sys.modules):
            if name.startswith("_cloudsentinel_external_plugins"):
                sys.modules.pop(name, None)

    def _write_dataset(self):
        random = np.random.default_rng(145)
        train = random.normal(0.0, 0.35, size=(80, 2))
        normal_test = random.normal(0.0, 0.35, size=(20, 2))
        anomalies = np.array([[6.0, 6.0], [-7.0, 7.0], [8.0, -8.0], [9.0, 9.0], [-9.0, -8.0]])
        test = np.vstack([normal_test, anomalies])
        labels = np.concatenate(
            [np.zeros(normal_test.shape[0], dtype=int), np.ones(anomalies.shape[0], dtype=int)]
        )
        dataset_id = "lof-fixture"
        directory = self.dataset_root / dataset_id
        directory.mkdir(parents=True)
        np.savetxt(directory / f"train_{dataset_id}", train, delimiter=",")
        np.savetxt(directory / f"test_{dataset_id}", test, delimiter=",")
        np.savetxt(directory / f"test_label_{dataset_id}", labels, fmt="%d", delimiter=",")
        (directory / "details.json").write_text(
            json.dumps({"containers": ["service"], "metrics": ["cpu", "memory"]}),
            encoding="utf-8",
        )
        return train, test, labels

    def _create_run(self, parameters=None):
        response = self.client.post(
            "/training_runs",
            json={
                "run_name": "External LOF acceptance",
                "dataset": {"source": "existing", "dataset_id": "lof-fixture"},
                "detectors": [
                    {"detector_id": LOF_ID, "parameters": parameters or {}}
                ],
            },
        )
        self.assertEqual(response.status_code, 202, response.get_json())
        context_payload = self.task.apply_async.call_args.kwargs["args"][0]
        return response.get_json(), context_payload

    def _implementation(self):
        adapter = registry.get_training_adapter(LOF_ID)
        package_name = adapter.__class__.__module__.rsplit(".", 1)[0]
        return adapter, importlib.import_module(f"{package_name}.implementation")

    def test_package_is_external_valid_lazy_and_not_builtin(self):
        self.assertFalse((REPOSITORY_ROOT / "detectors" / "local_outlier_factor").exists())
        builtins = registry.scan_detectors(REPOSITORY_ROOT / "detectors")
        self.assertNotIn(LOF_ID, [item["id"] for item in builtins["detectors"]])

        self.assertFalse(
            any(
                name.startswith("_cloudsentinel_external_plugins")
                for name in sys.modules
            )
        )
        report = registry.scan_detectors()
        item = next(item for item in report["detectors"] if item["id"] == LOF_ID)
        self.assertEqual(item["source"], "external")
        self.assertEqual(item["training_runtime_status"], "ready")
        self.assertEqual(
            item["capabilities"]["training"]["input_profile"],
            "metrics-partition-v1",
        )
        self.assertNotIn("entry_point", item)
        self.assertFalse(
            any(
                name.startswith("_cloudsentinel_external_plugins")
                for name in sys.modules
            )
        )
        adapter, implementation = self._implementation()
        self.assertEqual(adapter.__class__.__name__, "LocalOutlierFactorAdapter")
        self.assertEqual(implementation.DETECTOR_ID, LOF_ID)

        manifest = registry.load_manifest(LOF_SOURCE / "manifest.yaml")
        self.assertEqual(manifest["entry_point"], "adapter:LocalOutlierFactorAdapter")
        self.assertNotIn(".", manifest["entry_point"].split(":", 1)[0])
        self.assertEqual(self.release["training_runtime_status"], "ready")

    def test_real_pipeline_generic_training_evaluation_and_model_record(self):
        run, context_payload = self._create_run()
        self.assertEqual(context_payload["detector_id"], LOF_ID)
        self.assertEqual(context_payload["plugin"]["source"], "external")
        progress = []
        result = execute_detector_plugin_training(
            context_payload, lambda state, detail: progress.append((state, detail))
        )
        self.assertEqual(result["status"], "completed")
        self.assertIn("TRAINING", [state for state, _detail in progress])
        self.assertIn("EVALUATING", [state for state, _detail in progress])

        model_id = context_payload["model_id"]
        artifact_dir = self.artifact_root / "plugins" / LOF_ID / model_id
        expected_files = {
            "model.joblib",
            "model_metadata.json",
            "model_evaluation.json",
        }
        self.assertEqual({path.name for path in artifact_dir.iterdir()}, expected_files)
        pipeline = joblib.load(artifact_dir / "model.joblib")
        self.assertTrue(pipeline.named_steps["detector"].novelty)
        np.testing.assert_allclose(
            pipeline.named_steps["scaler"].mean_, self.train.mean(axis=0)
        )

        adapter, implementation = self._implementation()
        self.assertNotIn("labels", inspect.signature(implementation.train_and_persist).parameters)
        predictions, scores = implementation.predict(pipeline, self.test)
        raw = pipeline.predict(self.test)
        np.testing.assert_array_equal(predictions, (raw == -1).astype(np.int8))
        np.testing.assert_allclose(scores, -pipeline.decision_function(self.test))
        self.assertGreater(scores[-5:].mean(), scores[:20].mean())

        metadata = json.loads(
            (artifact_dir / "model_metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["detector_id"], LOF_ID)
        self.assertEqual(metadata["detector_version"], "1.0.0")
        self.assertEqual(metadata["plugin_version"], "1.0.0")
        self.assertEqual(metadata["artifact_format"], "joblib")
        self.assertEqual(metadata["sklearn_version"], "1.5.0")
        self.assertEqual(metadata["feature_identity"]["feature_order"], ["service_cpu", "service_memory"])
        self.assertRegex(
            metadata["feature_identity"]["feature_order_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertEqual(metadata["training_parameters"]["n_neighbors"], 20)
        self.assertEqual(metadata["train_observations"], 80)
        self.assertEqual(metadata["test_observations"], 25)
        self.assertEqual(metadata["label_source"], {"type": "snapshot_labels"})
        self.assertEqual(metadata["label_semantics"], {"0": "normal", "1": "anomaly"})
        self.assertTrue(metadata["score_semantics"]["higher_is_more_anomalous"])
        self.assertEqual(metadata["preprocessing"]["fit_source"], "train.csv")

        evaluation = json.loads(
            (artifact_dir / "model_evaluation.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(evaluation["binary_predictions"]), self.test.shape[0])
        self.assertEqual(len(evaluation["anomaly_scores"]), self.test.shape[0])
        self.assertEqual(evaluation["test_observations"], self.test.shape[0])

        stored_run = self.client.get(f"/training_runs/{run['run_id']}").get_json()
        child = stored_run["children"][0]
        self.assertEqual(child["status"], "completed")
        self.assertEqual(child["model_status"], "available")
        model = self.client.get(f"/models/{model_id}").get_json()
        self.assertEqual(model["detector_id"], LOF_ID)
        self.assertEqual(model["plugin"]["source"], "external")
        self.assertNotIn("package_sha256", model["plugin"])
        self.assertNotIn("manifest_sha256", model["plugin"])
        stored_model = ModelCatalogueStore(self.model_root).read(model_id)
        self.assertEqual(stored_model["plugin"], context_payload["plugin"])
        self.assertEqual(model["artifact"]["format"], "joblib")
        self.assertEqual(model["promotion"]["status"], "not_applicable")
        self.assertEqual(model["inference"]["status"], "ready")
        self.assertEqual(model["feature_identity"]["feature_order"], ["service_cpu", "service_memory"])
        self.assertNotIn("binary_predictions", model["evaluation"])
        self.assertNotIn("anomaly_scores", model["evaluation"])
        self.assertIn("f1", model["evaluation"])

    def _context_for_arrays(
        self, train: np.ndarray, test: np.ndarray, labels: np.ndarray, parameters: dict
    ) -> TrainingContext:
        directory = self.root / f"context-{hashlib.sha256(train.tobytes() + test.tobytes()).hexdigest()[:12]}"
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()
        train_path = directory / "train.csv"
        test_path = directory / "test.csv"
        labels_path = directory / "labels.csv"
        np.savetxt(train_path, train, delimiter=",")
        np.savetxt(test_path, test, delimiter=",")
        np.savetxt(labels_path, labels, fmt="%d", delimiter=",")
        return TrainingContext(
            run_id="run-lof-test",
            child_id="child-lof-test",
            model_id="model-lof-test",
            detector_id=LOF_ID,
            detector_version="1.0.0",
            parameters=parameters,
            dataset={"source": "existing", "dataset_id": "fixture"},
            snapshot={},
            feature_identity={"feature_order": ["first", "second"]},
            data=TrainingData(
                train_path=train_path,
                test_path=test_path,
                labels_path=labels_path,
                train_observations=train.shape[0],
                test_observations=test.shape[0],
                feature_count=train.shape[1],
            ),
            artifact_workspace=self.artifact_root,
            suggested_artifact_directory=self.artifact_root / "manual" / "model-lof-test",
            model_record_context={},
        )

    def test_targeted_neighbor_and_feature_compatibility_failures(self):
        adapter, _implementation = self._implementation()
        parameters = {
            "n_neighbors": 3,
            "metric": "minkowski",
            "contamination": "auto",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
        }
        too_small = self._context_for_arrays(
            np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]),
            np.array([[0.0, 0.0], [4.0, 4.0]]),
            np.array([0, 1]),
            parameters,
        )
        with self.assertRaisesRegex(DetectorCompatibilityError, "smaller than"):
            adapter.validate_training(too_small)

        mismatched = self._context_for_arrays(
            np.arange(12, dtype=float).reshape(6, 2),
            np.arange(9, dtype=float).reshape(3, 3),
            np.array([0, 0, 1]),
            {**parameters, "n_neighbors": 2},
        )
        with self.assertRaisesRegex(DetectorCompatibilityError, "same feature count"):
            adapter.validate_training(mismatched)

    def test_no_lof_specific_central_task_dispatch_or_kubernetes_resource(self):
        for root_name in ("detectors", "learning_adaptation"):
            for path in (REPOSITORY_ROOT / root_name).rglob("*.py"):
                self.assertNotIn(LOF_ID, path.read_text(encoding="utf-8"), str(path))
        for path in (REPOSITORY_ROOT / "k8s").glob("*.yml"):
            self.assertNotIn(LOF_ID, path.read_text(encoding="utf-8"), str(path))
        tasks = (REPOSITORY_ROOT / "learning_adaptation" / "tasks.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def train_detector_plugin_task", tasks)
        self.assertNotIn("train_local_outlier_factor", tasks)


if __name__ == "__main__":
    unittest.main()
