import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests

from anomaly_detection.isolation_forest.app import create_app
from detectors.registry import get_adapter


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def synthetic_data():
    rng = np.random.default_rng(42)
    train = rng.normal(0.0, 0.25, size=(80, 2))
    normal_test = rng.normal(0.0, 0.25, size=(20, 2))
    anomalous_test = rng.normal(8.0, 0.1, size=(5, 2))
    return train, normal_test, anomalous_test


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def csv_file(array, filename="test.csv"):
    output = io.StringIO()
    np.savetxt(output, np.asarray(array), delimiter=",")
    return io.BytesIO(output.getvalue().encode("utf-8")), filename


class IsolationForestDetectionServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_directory = tempfile.TemporaryDirectory()
        cls.train, cls.normal_test, cls.anomalous_test = synthetic_data()
        source_path = Path(cls.source_directory.name)
        adapter = get_adapter("isolation-forest")
        adapter.train(
            cls.train,
            source_path,
            training_parameters={"n_estimators": 25, "random_state": 42},
            model_id="model-1",
        )
        adapter.evaluate(
            np.vstack((cls.normal_test, cls.anomalous_test)),
            np.concatenate((np.zeros(20), np.ones(5))),
            source_path,
        )
        cls.artifacts = {
            "model.joblib": (source_path / "model.joblib").read_bytes(),
            "model_metadata.json": (source_path / "model_metadata.json").read_bytes(),
            "model_evaluation.json": (source_path / "model_evaluation.json").read_bytes(),
        }

    @classmethod
    def tearDownClass(cls):
        cls.source_directory.cleanup()

    def setUp(self):
        self.storage = tempfile.TemporaryDirectory()
        storage_root = Path(self.storage.name)
        self.model_root = storage_root / "models"
        self.results_root = storage_root / "results"
        self.app = create_app(
            {
                "TESTING": True,
                "IF_MODEL_STORAGE_ROOT": str(self.model_root),
                "IF_RESULTS_STORAGE_ROOT": str(self.results_root),
                "MAX_CONTENT_LENGTH": 10 * 1024 * 1024,
            }
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.storage.cleanup()

    def upload_data(self, *, model_id="model-1", artifacts=None, **overrides):
        values = dict(self.artifacts if artifacts is None else artifacts)
        data = {
            "detector_id": "isolation-forest",
            "model_id": model_id,
            "model_sha256": digest(values["model.joblib"]),
            "metadata_sha256": digest(values["model_metadata.json"]),
            "evaluation_sha256": digest(values["model_evaluation.json"]),
            "model": (io.BytesIO(values["model.joblib"]), "model.joblib"),
            "metadata": (
                io.BytesIO(values["model_metadata.json"]),
                "model_metadata.json",
            ),
            "evaluation": (
                io.BytesIO(values["model_evaluation.json"]),
                "model_evaluation.json",
            ),
        }
        data.update(overrides)
        return data

    def upload_model(self):
        return self.client.post(
            "/save_model",
            data=self.upload_data(),
            content_type="multipart/form-data",
        )

    def detect(
        self,
        matrix,
        model_id="model-1",
        iteration="0",
        task_id="monitor-1",
        **data_overrides,
    ):
        data = {
            "detector_id": "isolation-forest",
            "model": model_id,
            "iteration": iteration,
            "start_time": 1718738837,
            "end_time": 1718739437,
            "containers": ["service-a"],
            "metrics": ["cpu", "memory"],
            "data_interval": 60,
            "crca_threshold": 100.0,
            "crca_pods": ["service-a-pod"],
        }
        data.update(data_overrides)
        return self.client.post(
            "/detect_anomalies",
            data={
                "test_array": csv_file(matrix),
                "test_info": json.dumps(
                    {
                        "task_id": task_id,
                        "settings": {
                            "API_DATA_INGESTION_URL": "http://data-ingestion.test"
                        },
                        "data": data,
                    }
                ),
            },
            content_type="multipart/form-data",
        )

    def test_healthcheck_does_not_load_ml_runtime_or_adapter(self):
        script = """
import sys
from anomaly_detection.isolation_forest.app import create_app
app = create_app({'TESTING': True})
response = app.test_client().get('/healthz')
assert response.status_code == 200
assert response.get_json()['detector_id'] == 'isolation-forest'
assert 'detectors.isolation_forest.adapter' not in sys.modules
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

    def test_successful_upload_stores_exactly_three_identical_artifacts(self):
        response = self.upload_model()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {
                "status": "available",
                "detector_id": "isolation-forest",
                "model_id": "model-1",
                "idempotent": False,
            },
        )
        model_directory = self.model_root / "isolation-forest" / "model-1"
        self.assertEqual(
            sorted(path.name for path in model_directory.iterdir()),
            ["model.joblib", "model_evaluation.json", "model_metadata.json"],
        )
        for filename, expected in self.artifacts.items():
            self.assertEqual((model_directory / filename).read_bytes(), expected)

    def test_checksum_mismatch_is_rejected_without_active_model(self):
        response = self.client.post(
            "/save_model",
            data=self.upload_data(model_sha256="0" * 64),
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("checksum mismatch", response.get_json()["error"])
        self.assertFalse((self.model_root / "isolation-forest" / "model-1").exists())

    def test_missing_files_and_form_fields_are_rejected(self):
        missing_file_data = self.upload_data()
        del missing_file_data["evaluation"]
        response = self.client.post(
            "/save_model", data=missing_file_data, content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing required file", response.get_json()["error"])

        missing_field_data = self.upload_data()
        del missing_field_data["metadata_sha256"]
        response = self.client.post(
            "/save_model", data=missing_field_data, content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("metadata_sha256", response.get_json()["error"])

    def test_unsafe_model_id_is_rejected(self):
        response = self.client.post(
            "/save_model",
            data=self.upload_data(model_id="../model"),
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("model_id", response.get_json()["error"])

    def test_invalid_metadata_and_identity_mismatches_are_rejected(self):
        base_metadata = json.loads(self.artifacts["model_metadata.json"])
        invalid_documents = (
            b"not-json",
            json.dumps({**base_metadata, "schema_version": 2}).encode(),
            json.dumps({**base_metadata, "detector_id": "cgnn"}).encode(),
            json.dumps({**base_metadata, "model_id": "another-model"}).encode(),
            json.dumps({**base_metadata, "artifact_format": "pickle"}).encode(),
            json.dumps({**base_metadata, "n_features": 0}).encode(),
            json.dumps({**base_metadata, "feature_identity": "named"}).encode(),
            json.dumps({**base_metadata, "preprocessing": ""}).encode(),
            json.dumps({**base_metadata, "score_semantics": {}}).encode(),
        )
        for metadata in invalid_documents:
            with self.subTest(metadata=metadata[:30]):
                artifacts = {**self.artifacts, "model_metadata.json": metadata}
                response = self.client.post(
                    "/save_model",
                    data=self.upload_data(artifacts=artifacts),
                    content_type="multipart/form-data",
                )
                self.assertEqual(response.status_code, 400)
                self.assertFalse(
                    (self.model_root / "isolation-forest" / "model-1").exists()
                )

    def test_repeated_identical_upload_is_idempotent(self):
        self.assertEqual(self.upload_model().status_code, 200)
        response = self.upload_model()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["idempotent"])

    def test_existing_model_with_different_content_returns_conflict(self):
        self.assertEqual(self.upload_model().status_code, 200)
        changed_evaluation = json.dumps({"f1": 0.123}).encode()
        artifacts = {
            **self.artifacts,
            "model_evaluation.json": changed_evaluation,
        }
        response = self.client.post(
            "/save_model",
            data=self.upload_data(artifacts=artifacts),
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 409)
        self.assertIn("different artifacts", response.get_json()["error"])

    def test_real_uploaded_pipeline_predicts_via_registry_adapter(self):
        self.assertEqual(self.upload_model().status_code, 200)
        from anomaly_detection.isolation_forest import app as service_module

        with patch.object(
            service_module.registry,
            "get_adapter",
            wraps=service_module.registry.get_adapter,
        ) as resolver:
            normal_response = self.detect(self.normal_test, iteration="0")
            anomalous_response = self.detect(self.anomalous_test, iteration="1")

        self.assertEqual(normal_response.status_code, 200)
        self.assertEqual(anomalous_response.status_code, 200)
        normal_result = normal_response.get_json()
        anomalous_result = anomalous_response.get_json()
        self.assertEqual(normal_result["iteration"], "0")
        self.assertEqual(anomalous_result["iteration"], "1")
        self.assertGreaterEqual(normal_result["percentage"], 0.0)
        self.assertLessEqual(normal_result["percentage"], 100.0)
        self.assertGreaterEqual(anomalous_result["percentage"], normal_result["percentage"])
        self.assertLessEqual(anomalous_result["percentage"], 100.0)
        result_file = self.results_root / "monitor-1" / "isolation_forest_results.json"
        stored = json.loads(result_file.read_text(encoding="utf-8"))
        self.assertEqual(set(stored["results"]), {"0", "1"})
        self.assertEqual(stored["detector_id"], "isolation-forest")
        self.assertNotIn("binary_predictions", json.dumps(stored))
        self.assertNotIn("anomaly_scores", json.dumps(stored))
        self.assertNotIn("test_array", json.dumps(stored))
        self.assertEqual(resolver.call_count, 2)
        resolver.assert_called_with("isolation-forest")

    def test_prediction_input_and_model_errors_are_targeted(self):
        self.assertEqual(self.upload_model().status_code, 200)

        response = self.detect(np.ones((3, 3)))
        self.assertEqual(response.status_code, 400)
        self.assertIn("stored model expects 2", response.get_json()["error"])

        response = self.detect(np.array([[1.0, np.nan]]))
        self.assertEqual(response.status_code, 400)
        self.assertIn("finite", response.get_json()["error"])

        response = self.detect(np.ones((3, 2)), model_id="unknown-model")
        self.assertEqual(response.status_code, 404)
        self.assertIn("unknown Isolation Forest model", response.get_json()["error"])

    def test_prediction_revalidates_stored_metadata(self):
        self.assertEqual(self.upload_model().status_code, 200)
        metadata_path = (
            self.model_root
            / "isolation-forest"
            / "model-1"
            / "model_metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["detector_id"] = "cgnn"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        response = self.detect(self.normal_test)
        self.assertEqual(response.status_code, 400)
        self.assertIn("metadata detector_id", response.get_json()["error"])

    def test_detection_at_threshold_stores_compact_result_without_rca(self):
        self.assertEqual(self.upload_model().status_code, 200)
        from anomaly_detection.isolation_forest import app as service_module

        adapter = Mock()
        adapter.predict.return_value = {
            "binary_predictions": [1, 0],
            "anomaly_scores": [0.8, -0.2],
            "anomaly_percentage": 5.0,
        }
        with patch.object(
            service_module.registry, "get_adapter", return_value=adapter
        ), patch(
            "anomaly_detection.isolation_forest.rca.requests.post"
        ) as rca_post:
            response = self.detect(
                [[8.0, 8.0], [0.0, 0.0]],
                crca_threshold=5.0,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["crca_task_id"], None)
        rca_post.assert_not_called()
        result_path = self.results_root / "monitor-1" / "isolation_forest_results.json"
        stored = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(
            stored,
            {
                "detector_id": "isolation-forest",
                "model": "model-1",
                "start_time": 1718738837,
                "containers": ["service-a"],
                "metrics": ["cpu", "memory"],
                "step": 60,
                "crca_threshold": 5.0,
                "results": {
                    "0": {
                        "start_time": 1718738837,
                        "end_time": 1718739437,
                        "percentage": 5.0,
                        "anomaly_count": 1,
                        "observation_count": 2,
                        "crca_task_id": None,
                    }
                },
            },
        )
        serialized = json.dumps(stored)
        for forbidden in ("binary_predictions", "anomaly_scores", "test_array"):
            self.assertNotIn(forbidden, serialized)

    def test_detection_above_threshold_triggers_compatible_rca_payload(self):
        self.assertEqual(self.upload_model().status_code, 200)
        from anomaly_detection.isolation_forest import app as service_module

        adapter = Mock()
        adapter.predict.return_value = {
            "binary_predictions": [1, 0],
            "anomaly_scores": [0.8, -0.2],
            "anomaly_percentage": 20.0,
        }
        rca_response = Mock(status_code=202, text="")
        rca_response.json.return_value = {"task_id": "rca-task-1"}
        with patch.object(
            service_module.registry, "get_adapter", return_value=adapter
        ), patch(
            "anomaly_detection.isolation_forest.rca.requests.post",
            return_value=rca_response,
        ) as rca_post:
            response = self.detect(
                [[8.0, 8.0], [0.0, 0.0]],
                crca_threshold=5.0,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["crca_task_id"], "rca-task-1")
        rca_post.assert_called_once()
        call = rca_post.call_args
        self.assertEqual(
            call.args[0], "http://data-ingestion.test/anomaly_rca"
        )
        self.assertEqual(call.kwargs["timeout"], (5.0, 60.0))
        self.assertEqual(
            json.loads(call.kwargs["data"]["crca_data"]),
            {
                "settings": {
                    "API_DATA_INGESTION_URL": "http://data-ingestion.test"
                },
                "data": {
                    "task_id": "monitor-1",
                    "start_time": 1718738837,
                    "end_time": 1718739437,
                    "crca_pods": ["service-a-pod"],
                    "metrics": ["cpu", "memory"],
                    "step": 60,
                },
            },
        )
        stored = json.loads(
            (
                self.results_root
                / "monitor-1"
                / "isolation_forest_results.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(stored["results"]["0"]["crca_task_id"], "rca-task-1")

    def test_rca_failure_is_stored_and_returned_without_losing_prediction(self):
        self.assertEqual(self.upload_model().status_code, 200)
        from anomaly_detection.isolation_forest import app as service_module

        adapter = Mock()
        adapter.predict.return_value = {
            "binary_predictions": [1, 0],
            "anomaly_scores": [0.8, -0.2],
            "anomaly_percentage": 20.0,
        }
        result_path = self.results_root / "monitor-1" / "isolation_forest_results.json"

        def fail_rca_after_result_was_stored(*_args, **_kwargs):
            self.assertTrue(result_path.is_file())
            preliminary = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(preliminary["results"]["0"]["percentage"], 20.0)
            raise requests.Timeout("RCA timed out")

        with patch.object(
            service_module.registry, "get_adapter", return_value=adapter
        ), patch(
            "anomaly_detection.isolation_forest.rca.requests.post",
            side_effect=fail_rca_after_result_was_stored,
        ):
            response = self.detect(
                [[8.0, 8.0], [0.0, 0.0]],
                crca_threshold=5.0,
            )

        self.assertEqual(response.status_code, 502)
        body = response.get_json()
        self.assertEqual(body["status"], "rca_failed")
        self.assertEqual(body["percentage"], 20.0)
        self.assertIsNone(body["crca_task_id"])
        self.assertIn("timed out", body["rca_error"])
        stored = json.loads(result_path.read_text(encoding="utf-8"))
        result = stored["results"]["0"]
        self.assertEqual(result["percentage"], 20.0)
        self.assertIsNone(result["crca_task_id"])
        self.assertIn("timed out", result["crca_error"])

    def test_iterations_append_and_duplicate_iteration_is_rejected(self):
        self.assertEqual(self.upload_model().status_code, 200)
        from anomaly_detection.isolation_forest import app as service_module

        adapter = Mock()
        adapter.predict.return_value = {
            "binary_predictions": [0, 0],
            "anomaly_scores": [-0.2, -0.1],
            "anomaly_percentage": 0.0,
        }
        with patch.object(
            service_module.registry, "get_adapter", return_value=adapter
        ):
            first = self.detect([[0.0, 0.0], [0.1, 0.1]], iteration="0")
            second = self.detect([[0.0, 0.0], [0.1, 0.1]], iteration="1")
            duplicate = self.detect([[0.0, 0.0], [0.1, 0.1]], iteration="1")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(duplicate.status_code, 409)
        self.assertIn("already exists", duplicate.get_json()["error"])
        stored = json.loads(
            (
                self.results_root
                / "monitor-1"
                / "isolation_forest_results.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(set(stored["results"]), {"0", "1"})
        self.assertEqual(adapter.predict.call_count, 2)

    def test_failed_atomic_write_leaves_no_partial_result_file(self):
        self.assertEqual(self.upload_model().status_code, 200)
        from anomaly_detection.isolation_forest import app as service_module

        adapter = Mock()
        adapter.predict.return_value = {
            "binary_predictions": [0],
            "anomaly_scores": [-0.2],
            "anomaly_percentage": 0.0,
        }
        with patch.object(
            service_module.registry, "get_adapter", return_value=adapter
        ), patch(
            "anomaly_detection.isolation_forest.results.os.replace",
            side_effect=OSError("disk write failed"),
        ):
            response = self.detect([[0.0, 0.0]])

        self.assertEqual(response.status_code, 500)
        self.assertIn("atomically store", response.get_json()["error"])
        task_directory = self.results_root / "monitor-1"
        self.assertFalse((task_directory / "isolation_forest_results.json").exists())
        self.assertEqual(list(task_directory.glob("*.tmp")), [])

    def test_extended_test_info_validation_rejects_unsafe_values(self):
        self.assertEqual(self.upload_model().status_code, 200)
        invalid_cases = (
            ({"iteration": "bad/iteration"}, "iteration"),
            ({"end_time": 1, "start_time": 2}, "end_time"),
            ({"data_interval": 0}, "data_interval"),
            ({"crca_threshold": 101}, "crca_threshold"),
            ({"containers": "service-a"}, "containers"),
            ({"metrics": [""]}, "metrics"),
            ({"crca_pods": [1]}, "crca_pods"),
        )
        for overrides, expected_error in invalid_cases:
            with self.subTest(overrides=overrides):
                task_id = overrides.get("task_id", "validation-task")
                data_overrides = {
                    key: value for key, value in overrides.items() if key != "task_id"
                }
                response = self.detect(
                    [[0.0, 0.0]],
                    task_id=task_id,
                    **data_overrides,
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn(expected_error, response.get_json()["error"])

    def test_task_id_path_safety_rejects_unsafe_directory_names(self):
        self.assertEqual(self.upload_model().status_code, 200)
        escaped_directory = self.results_root.parent / "escaped-task"
        unsafe_task_ids = (
            "",
            ".",
            "..",
            "../escaped-task",
            "safe/../../escaped-task",
            "nested/task",
            r"nested\task",
            str(escaped_directory),
        )
        for task_id in unsafe_task_ids:
            with self.subTest(task_id=task_id):
                response = self.detect([[0.0, 0.0]], task_id=task_id)
                self.assertEqual(response.status_code, 400)
                self.assertIn("task_id", response.get_json()["error"])
                self.assertFalse(escaped_directory.exists())

        valid_response = self.detect(
            [[0.0, 0.0]], task_id="safe.Task_1-2"
        )
        self.assertEqual(valid_response.status_code, 200)
        result_file = (
            self.results_root
            / "safe.Task_1-2"
            / "isolation_forest_results.json"
        )
        self.assertTrue(result_file.is_file())
        self.assertTrue(result_file.resolve().is_relative_to(self.results_root.resolve()))

    def test_task_directory_symlink_cannot_escape_results_root(self):
        self.assertEqual(self.upload_model().status_code, 200)
        outside_directory = self.results_root.parent / "outside-results"
        outside_directory.mkdir()
        self.results_root.mkdir()
        (self.results_root / "linked-task").symlink_to(
            outside_directory, target_is_directory=True
        )

        response = self.detect([[0.0, 0.0]], task_id="linked-task")

        self.assertEqual(response.status_code, 500)
        self.assertIn("outside IF_RESULTS_STORAGE_ROOT", response.get_json()["error"])
        self.assertEqual(list(outside_directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
