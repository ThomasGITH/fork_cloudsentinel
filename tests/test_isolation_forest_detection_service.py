import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

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

    def detect(self, matrix, model_id="model-1", iteration="0"):
        return self.client.post(
            "/detect_anomalies",
            data={
                "test_array": csv_file(matrix),
                "test_info": json.dumps(
                    {
                        "data": {
                            "detector_id": "isolation-forest",
                            "model": model_id,
                            "iteration": iteration,
                        }
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


if __name__ == "__main__":
    unittest.main()
