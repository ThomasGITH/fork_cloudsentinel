import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from data_processing.cgnn_preprocess import (
    GenericDetectionCallerError,
    handle_cgnn_request,
)
from detectors.cgnn.adapter import CGNNAdapter
from detectors.contracts import ProgressReporter, TrainingContext, TrainingData
from detectors.isolation_forest.adapter import IsolationForestAdapter


ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = json.dumps(self._payload).encode("utf-8")
        self.headers = {"Content-Type": "application/json"}

    def json(self):
        return self._payload


def training_context(root: Path, detector_id: str) -> TrainingContext:
    data_root = root / "snapshot"
    data_root.mkdir()
    train_path = data_root / "train.csv"
    test_path = data_root / "test.csv"
    labels_path = data_root / "labels.csv"
    np.savetxt(train_path, np.zeros((12, 2)), delimiter=",")
    np.savetxt(test_path, np.zeros((4, 2)), delimiter=",")
    np.savetxt(labels_path, np.zeros(4), delimiter=",")
    model_id = f"model-{detector_id}"
    return TrainingContext(
        run_id="run-migration",
        child_id=f"run-migration-{detector_id}",
        model_id=model_id,
        detector_id=detector_id,
        detector_version="1.0.0",
        parameters={},
        dataset={"source": "existing", "dataset_id": "dataset-migration"},
        snapshot={"dataset_details": {"containers": [], "metrics": []}},
        feature_identity={
            "feature_order": ["cpu", "memory"],
            "feature_order_sha256": "a" * 64,
        },
        data=TrainingData(
            train_path=train_path,
            test_path=test_path,
            labels_path=labels_path,
            train_observations=12,
            test_observations=4,
            feature_count=2,
        ),
        artifact_workspace=root / "artifacts",
        suggested_artifact_directory=root / "artifacts" / detector_id / model_id,
        model_record_context={},
    )


class GenericTrainingPublicationMigrationTests(unittest.TestCase):
    def test_isolation_forest_training_produces_complete_generic_artifact_without_promotion(self):
        rng = np.random.default_rng(71)
        train = rng.normal(size=(30, 2))
        test = np.vstack((rng.normal(size=(5, 2)), [[8.0, 8.0]]))
        labels = np.array([0, 0, 0, 0, 0, 1])
        with tempfile.TemporaryDirectory() as directory:
            context = training_context(Path(directory), "isolation-forest")
            context = TrainingContext(
                **{
                    **context.__dict__,
                    "parameters": {
                        "n_estimators": 10,
                        "max_samples": "auto",
                        "contamination": "auto",
                        "max_features": 1.0,
                        "bootstrap": False,
                        "random_state": 42,
                        "n_jobs": 1,
                    },
                }
            )
            adapter = IsolationForestAdapter()
            with patch.object(
                adapter, "_load_training_data", return_value=(train, test, labels)
            ), patch(
                "requests.post", side_effect=AssertionError("legacy promotion must not run")
            ):
                result = adapter.run_training(
                    context, ProgressReporter(lambda *_args: None)
                )

            self.assertEqual(result.promotion.status, "not_applicable")
            self.assertEqual(
                set(result.artifact.required_files),
                {"model.joblib", "model_metadata.json", "model_evaluation.json"},
            )
            for filename in result.artifact.required_files:
                self.assertTrue((result.artifact.directory / filename).is_file())

    def test_cgnn_training_produces_complete_generic_artifact_without_service_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = training_context(root, "cgnn")
            parameters = {
                "lookback": 2,
                "feature_importance": False,
            }
            context = TrainingContext(**{**context.__dict__, "parameters": parameters})
            adapter = CGNNAdapter()
            train = np.arange(24, dtype=np.float32).reshape(12, 2)
            test = np.arange(8, dtype=np.float32).reshape(4, 2)
            labels = np.array([0, 0, 1, 1], dtype=np.float32)

            def fake_train(dataset_config, *_args, **_kwargs):
                artifact = root / "artifacts" / "dataset-migration_fixture"
                artifact.mkdir(parents=True)
                (artifact / "model.pt").write_bytes(b"state-dict")
                (artifact / "model_config.json").write_text("{}", encoding="utf-8")
                (artifact / "model_evaluation.json").write_text(
                    json.dumps({"epsilon_result": {"threshold": 0.5}}),
                    encoding="utf-8",
                )
                return {
                    "dataset": "dataset-migration",
                    "id": "fixture",
                    "feature_importance": False,
                }, None

            with patch.object(
                adapter, "_load_training_data", return_value=(train, test, labels)
            ), patch.object(adapter, "train", side_effect=fake_train), patch.object(
                adapter,
                "evaluate",
                return_value={"epsilon_result": {"threshold": 0.5}},
            ), patch(
                "requests.post", side_effect=AssertionError("legacy service must not run")
            ):
                result = adapter.run_training(
                    context, ProgressReporter(lambda *_args: None)
                )

            self.assertEqual(result.promotion.status, "not_applicable")
            self.assertEqual(
                set(result.artifact.required_files),
                {
                    "model.pt",
                    "model_config.json",
                    "model_evaluation.json",
                    "model_params.json",
                    "scaler.joblib",
                },
            )
            for filename in result.artifact.required_files:
                self.assertTrue((result.artifact.directory / filename).is_file())

    def test_generic_training_sources_do_not_reference_legacy_detection_services(self):
        sources = [
            ROOT / "learning_adaptation" / "plugin_training.py",
            ROOT / "detectors" / "cgnn" / "adapter.py",
            ROOT / "detectors" / "isolation_forest" / "adapter.py",
            ROOT / "examples" / "external_plugins" / "local_outlier_factor" / "adapter.py",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
        for forbidden in (
            "cgnn-anomaly-detection-service",
            "isolation-forest-anomaly-detection-service",
            "/save_model",
            "promote_isolation_forest_model",
        ):
            self.assertNotIn(forbidden, combined)

        retired_cgnn_promotion = (
            ROOT / "learning_adaptation" / "app.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("API_CGNN_ANOMALY_DETECTION_URL", retired_cgnn_promotion)
        self.assertNotIn("/save_model", retired_cgnn_promotion)
        task_results = (
            ROOT
            / "user_interface"
            / "monitoring_project"
            / "config_app"
            / "views"
            / "task_results.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("/save_to_detection_module", task_results)

        # CGNN remains deployed for continuous monitoring. Live parity has
        # already allowed the detector-specific IF resources to be retired.
        cgnn_manifest = (
            ROOT / "k8s" / "cgnn_anomaly_detection-deployment.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("cgnn-anomaly-detection-deployment", cgnn_manifest)
        self.assertIn("cgnn-anomaly-detection-service", cgnn_manifest)
        self.assertFalse(
            (
                ROOT
                / "k8s"
                / "isolation_forest_anomaly_detection-deployment.yml"
            ).exists()
        )


class GenericDetectionCallerMigrationTests(unittest.TestCase):
    def model(self, *, status="ready"):
        return {
            "model_id": "model-cgnn-ready",
            "inference": {"status": status},
            "feature_identity": {
                "feature_order": ["cpu", "memory"],
                "feature_order_sha256": "b" * 64,
            },
        }

    def test_caller_uses_server_configured_generic_url_model_id_and_feature_identity(self):
        calls = []

        def post(url, **kwargs):
            calls.append((url, kwargs))
            if url.endswith("/activate"):
                return FakeResponse(payload={"status": "ready"})
            return FakeResponse(
                payload={
                    "status": "success",
                    "model_id": "model-cgnn-ready",
                    "prediction_count": 2,
                }
            )

        with patch.dict(
            os.environ,
            {
                "API_GENERIC_ANOMALY_DETECTION_URL": "http://generic.test",
                "API_LEARNING_ADAPTATION_URL": "http://learning.test",
            },
        ), patch(
            "data_processing.cgnn_preprocess.requests.get",
            return_value=FakeResponse(payload=self.model()),
        ) as model_request, patch(
            "data_processing.cgnn_preprocess.requests.post", side_effect=post
        ):
            response = handle_cgnn_request(
                io.StringIO("10,20\n30,40\n"),
                {
                    "settings": {
                        "API_CGNN_ANOMALY_DETECTION_URL": "http://legacy.invalid"
                    },
                    "data": {"model_id": "model-cgnn-ready", "iteration": 3},
                },
            )

        self.assertEqual(response.status_code, 200)
        model_request.assert_called_once_with(
            "http://learning.test/models/model-cgnn-ready", timeout=(5.0, 120.0)
        )
        self.assertEqual(
            [call[0] for call in calls],
            [
                "http://generic.test/models/model-cgnn-ready/activate",
                "http://generic.test/detect",
            ],
        )
        metadata = json.loads(calls[1][1]["data"]["metadata"])
        self.assertEqual(metadata["model_id"], "model-cgnn-ready")
        self.assertEqual(metadata["feature_order"], ["cpu", "memory"])
        self.assertEqual(metadata["context"], {"iteration": 3})
        self.assertEqual(calls[1][1]["files"]["matrix"][1], "10.0,20.0\n30.0,40.0\n")
        self.assertNotIn("legacy.invalid", json.dumps(calls))

    def test_not_ready_model_is_rejected_before_runtime_call(self):
        with patch.dict(
            os.environ,
            {
                "API_GENERIC_ANOMALY_DETECTION_URL": "http://generic.test",
                "API_LEARNING_ADAPTATION_URL": "http://learning.test",
            },
        ), patch(
            "data_processing.cgnn_preprocess.requests.get",
            return_value=FakeResponse(payload=self.model(status="legacy_external")),
        ), patch("data_processing.cgnn_preprocess.requests.post") as runtime_request:
            with self.assertRaisesRegex(GenericDetectionCallerError, "not ready"):
                handle_cgnn_request(
                    io.StringIO("1,2\n3,4\n"),
                    {"data": {"model_id": "model-cgnn-ready"}},
                )
        runtime_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
