import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests

from learning_adaptation.isolation_forest_promotion import (
    ARTIFACT_FILES,
    IsolationForestPromotionError,
    promote_isolation_forest_model,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_data():
    rng = np.random.default_rng(42)
    train = rng.normal(0.0, 0.25, size=(80, 2))
    normal_test = rng.normal(0.0, 0.25, size=(20, 2))
    anomalous_test = rng.normal(8.0, 0.1, size=(5, 2))
    test = np.vstack((normal_test, anomalous_test))
    labels = np.concatenate((np.zeros(20, dtype=int), np.ones(5, dtype=int)))
    return train, test, labels


def csv_upload(array, filename):
    stream = io.StringIO()
    np.savetxt(stream, np.asarray(array), delimiter=",")
    return io.BytesIO(stream.getvalue().encode("utf-8")), filename


class FakeConfig:
    def update(self, *args, **kwargs):
        return None


class FakeCeleryApp:
    def __init__(self, *args, **kwargs):
        self.conf = FakeConfig()

    def task(self, *args, **kwargs):
        def decorate(function):
            function.celery_task_options = kwargs
            return function

        return decorate


def celery_stub():
    module = types.ModuleType("celery")
    module.Celery = FakeCeleryApp
    module.Task = object
    return module


def config_stubs():
    package = types.ModuleType("cgnn")
    package.__path__ = []
    config = types.ModuleType("cgnn.config")
    config.set_config = Mock()
    config.get_config = Mock(return_value={})
    config.set_initial_config = Mock()
    return package, config


class FakePromotionResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def available_response(model_id="model-1", **overrides):
    payload = {
        "status": "available",
        "detector_id": "isolation-forest",
        "model_id": model_id,
    }
    payload.update(overrides)
    return FakePromotionResponse(payload=payload)


def write_placeholder_artifacts(directory):
    directory.mkdir(parents=True, exist_ok=True)
    contents = {
        "model.joblib": b"binary-model-content",
        "model_metadata.json": b'{"detector_id":"isolation-forest"}',
        "model_evaluation.json": b'{"f1":1.0}',
    }
    for filename, content in contents.items():
        (directory / filename).write_bytes(content)
    return contents


class IsolationForestTrainingRouteTests(unittest.TestCase):
    def setUp(self):
        self.submitted_task = Mock()
        self.submitted_task.apply_async.return_value = types.SimpleNamespace(id="if-task-123")
        tasks = types.ModuleType("tasks")
        tasks.train_and_evaluate_task = Mock()
        tasks.train_and_evaluate_isolation_forest_task = self.submitted_task
        cgnn, config = config_stubs()
        stubs = {
            "celery": celery_stub(),
            "tasks": tasks,
            "cgnn": cgnn,
            "cgnn.config": config,
            "torch": types.ModuleType("torch"),
        }
        with patch.dict(sys.modules, stubs):
            self.module = load_module(
                "if_b_learning_app",
                REPOSITORY_ROOT / "learning_adaptation" / "app.py",
            )
        self.client = self.module.app.test_client()
        self.train, self.test, self.labels = synthetic_data()

    def tearDown(self):
        sys.modules.pop("if_b_learning_app", None)

    def request_data(self, **overrides):
        data = {
            "train_array": csv_upload(self.train, "train.csv"),
            "test_array": csv_upload(self.test, "test.csv"),
            "anomaly_label_array": csv_upload(self.labels, "labels.csv"),
            "model_id": "if-model-1",
            "training_parameters": json.dumps({"n_estimators": 25}),
        }
        data.update(overrides)
        return data

    def test_valid_request_returns_accepted_task_id_and_dispatches_validated_data(self):
        response = self.client.post(
            "/isolation_forest_train_model",
            data=self.request_data(),
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json(), {"task_id": "if-task-123"})
        self.submitted_task.apply_async.assert_called_once()
        arguments = self.submitted_task.apply_async.call_args.kwargs["args"]
        self.assertEqual(np.asarray(arguments[0]).shape, (80, 2))
        self.assertEqual(np.asarray(arguments[1]).shape, (25, 2))
        self.assertEqual(np.asarray(arguments[2]).shape, (25,))
        self.assertEqual(arguments[3]["n_estimators"], 25)
        self.assertEqual(arguments[3]["contamination"], "auto")
        self.assertEqual(arguments[3]["max_samples"], "auto")
        self.assertEqual(arguments[4], "if-model-1")

    def test_invalid_training_inputs_return_clear_client_errors(self):
        invalid_cases = (
            (
                {"train_array": csv_upload([[1.0, np.nan], [2.0, 3.0]], "train.csv")},
                "finite",
            ),
            (
                {"test_array": csv_upload(np.ones((25, 3)), "test.csv")},
                "same number of features",
            ),
            (
                {"anomaly_label_array": csv_upload(np.zeros(24), "labels.csv")},
                "same number of observations",
            ),
            (
                {"anomaly_label_array": csv_upload(np.full(25, 2), "labels.csv")},
                "only 0 and 1",
            ),
        )
        for overrides, expected_error in invalid_cases:
            with self.subTest(expected_error=expected_error):
                response = self.client.post(
                    "/isolation_forest_train_model",
                    data=self.request_data(**overrides),
                    content_type="multipart/form-data",
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn(expected_error, response.get_json()["error"])

        response = self.client.post(
            "/isolation_forest_train_model",
            data={"test_array": csv_upload(self.test, "test.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing required file", response.get_json()["error"])
        self.submitted_task.apply_async.assert_not_called()

    def test_unknown_and_unsupported_parameters_return_client_errors(self):
        invalid_parameters = (
            ({"unknown": 1}, "unknown training parameter"),
            ({"contamination": 0.1}, "contamination.*must be 'auto'"),
            ({"max_samples": 20}, "max_samples.*must be 'auto'"),
        )
        for parameters, expected_error in invalid_parameters:
            with self.subTest(parameters=parameters):
                response = self.client.post(
                    "/isolation_forest_train_model",
                    data=self.request_data(
                        training_parameters=json.dumps(parameters)
                    ),
                    content_type="multipart/form-data",
                )
                self.assertEqual(response.status_code, 400)
                self.assertRegex(response.get_json()["error"], expected_error)
        self.submitted_task.apply_async.assert_not_called()


class IsolationForestTrainingTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(sys.modules, {"celery": celery_stub()}):
            cls.module = load_module(
                "if_b_tasks",
                REPOSITORY_ROOT / "learning_adaptation" / "tasks.py",
            )

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("if_b_tasks", None)

    def test_task_resolves_one_adapter_and_uses_detector_namespace(self):
        metadata = {
            "training_parameters": {"n_estimators": 100},
            "n_features": 2,
        }
        evaluation = {
            key: index
            for index, key in enumerate(self.module.ISOLATION_FOREST_EVALUATION_KEYS)
        }
        def train(_train, artifact_dir, **_kwargs):
            write_placeholder_artifacts(artifact_dir)
            return metadata

        adapter = types.SimpleNamespace(
            train=Mock(side_effect=train),
            evaluate=Mock(return_value=evaluation),
        )
        task_context = types.SimpleNamespace(update_state=Mock())

        with tempfile.TemporaryDirectory() as directory, (
            patch.object(
                self.module,
                "ISOLATION_FOREST_ARTIFACT_ROOT",
                Path(directory) / "isolation-forest",
            )
        ), patch.object(
            self.module.registry, "get_adapter", return_value=adapter
        ) as resolver, patch.object(
            self.module,
            "promote_isolation_forest_model",
            return_value={
                "status": "available",
                "detector_id": "isolation-forest",
                "model_id": "model-1",
            },
        ) as promotion:
            result = self.module.train_and_evaluate_isolation_forest_task(
                task_context,
                [[0.0, 0.0], [0.1, 0.1]],
                [[2.0, 2.0]],
                [1],
                {"n_estimators": 100},
                "model-1",
            )

        resolver.assert_called_once_with("isolation-forest")
        adapter.train.assert_called_once()
        adapter.evaluate.assert_called_once()
        expected_dir = Path(directory) / "isolation-forest" / "model-1"
        self.assertEqual(adapter.train.call_args.args[1], expected_dir)
        self.assertEqual(adapter.evaluate.call_args.args[2], expected_dir)
        promotion.assert_called_once_with(expected_dir, "model-1")
        self.assertEqual(result["artifact_dir"], "isolation-forest/model-1")
        self.assertEqual(result["promotion"]["status"], "available")
        self.assertEqual(
            [call.kwargs["state"] for call in task_context.update_state.call_args_list],
            ["INITIATING", "TRAINING", "EVALUATING", "PROMOTING", "COMPLETED"],
        )

    def test_promotion_retry_reuses_artifacts_without_retraining(self):
        metadata = {
            "training_parameters": {"n_estimators": 100},
            "n_features": 2,
        }
        evaluation = {
            key: index
            for index, key in enumerate(self.module.ISOLATION_FOREST_EVALUATION_KEYS)
        }

        def train(_train, artifact_dir, **_kwargs):
            write_placeholder_artifacts(artifact_dir)
            return metadata

        adapter = types.SimpleNamespace(
            train=Mock(side_effect=train),
            evaluate=Mock(return_value=evaluation),
        )
        task_context = types.SimpleNamespace(update_state=Mock())
        attempts = []

        def lose_first_response(_url, *, data, files, timeout):
            attempts.append(
                {
                    "data": dict(data),
                    "files": {
                        field: (value[0], value[1].read())
                        for field, value in files.items()
                    },
                    "timeout": timeout,
                }
            )
            if len(attempts) == 1:
                raise requests.Timeout("response lost after installation")
            return available_response("model-1", idempotent=True)

        with tempfile.TemporaryDirectory() as directory, patch.object(
            self.module,
            "ISOLATION_FOREST_ARTIFACT_ROOT",
            Path(directory) / "isolation-forest",
        ), patch.object(
            self.module.registry, "get_adapter", return_value=adapter
        ), patch(
            "learning_adaptation.isolation_forest_promotion.requests.post",
            side_effect=lose_first_response,
        ):
            artifact_dir = Path(directory) / "isolation-forest" / "model-1"
            result = self.module.train_and_evaluate_isolation_forest_task(
                task_context,
                [[0.0, 0.0], [0.1, 0.1]],
                [[2.0, 2.0]],
                [1],
                {"n_estimators": 100},
                "model-1",
            )
            self.assertFalse(artifact_dir.exists())

        adapter.train.assert_called_once()
        adapter.evaluate.assert_called_once()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], attempts[1])
        self.assertEqual(result["promotion"]["status"], "available")

    def test_promotion_failure_is_excluded_from_full_celery_autoretry(self):
        excluded = self.module.train_and_evaluate_isolation_forest_task.celery_task_options[
            "dont_autoretry_for"
        ]
        self.assertIn(self.module.IsolationForestPromotionError, excluded)

    def test_real_task_run_writes_artifacts_and_returns_compact_json(self):
        train, test, labels = synthetic_data()
        task_context = types.SimpleNamespace(update_state=Mock())
        parameters = {
            "n_estimators": 25,
            "max_samples": "auto",
            "contamination": "auto",
            "max_features": 1.0,
            "bootstrap": False,
            "random_state": 42,
            "n_jobs": 1,
        }

        uploaded = {}

        def capture_upload(url, *, data, files, timeout):
            uploaded["url"] = url
            uploaded["data"] = dict(data)
            uploaded["timeout"] = timeout
            uploaded["files"] = {
                field: {
                    "filename": value[0],
                    "content": value[1].read(),
                    "content_type": value[2],
                }
                for field, value in files.items()
            }
            return available_response("real-model", idempotent=True)

        with tempfile.TemporaryDirectory() as directory, patch.object(
            self.module,
            "ISOLATION_FOREST_ARTIFACT_ROOT",
            Path(directory) / "isolation-forest",
        ), patch(
            "learning_adaptation.isolation_forest_promotion.requests.post",
            side_effect=capture_upload,
        ):
            artifact_dir = Path(directory) / "isolation-forest" / "real-model"
            with patch.dict(
                os.environ,
                {
                    "API_ISOLATION_FOREST_ANOMALY_DETECTION_URL":
                        "http://if-detection.test:5014/"
                },
            ):
                result = self.module.train_and_evaluate_isolation_forest_task(
                    task_context,
                    train.tolist(),
                    test.tolist(),
                    labels.tolist(),
                    parameters,
                    "real-model",
                )
            self.assertFalse(artifact_dir.exists())

        stored_evaluation = json.loads(
            uploaded["files"]["evaluation"]["content"].decode("utf-8")
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["detector_id"], "isolation-forest")
        self.assertEqual(result["model_id"], "real-model")
        self.assertEqual(result["artifact_dir"], "isolation-forest/real-model")
        self.assertEqual(result["training_parameters"], parameters)
        self.assertEqual(result["n_features"], 2)
        self.assertEqual(
            result["promotion"],
            {
                "status": "available",
                "detector_id": "isolation-forest",
                "model_id": "real-model",
            },
        )
        self.assertNotIn("binary_predictions", result["evaluation"])
        self.assertNotIn("anomaly_scores", result["evaluation"])
        self.assertIn("binary_predictions", stored_evaluation)
        self.assertIn("anomaly_scores", stored_evaluation)
        json.dumps(result)
        self.assertLess(len(json.dumps(result)), 2_000)
        self.assertEqual(
            uploaded["url"], "http://if-detection.test:5014/save_model"
        )
        self.assertEqual(uploaded["timeout"], (5.0, 60.0))
        self.assertEqual(
            {field: item["filename"] for field, item in uploaded["files"].items()},
            ARTIFACT_FILES,
        )
        self.assertTrue(uploaded["files"]["model"]["content"])
        for field in ARTIFACT_FILES:
            expected_digest = hashlib.sha256(
                uploaded["files"][field]["content"]
            ).hexdigest()
            self.assertEqual(uploaded["data"][f"{field}_sha256"], expected_digest)
        self.assertEqual(uploaded["data"]["detector_id"], "isolation-forest")
        self.assertEqual(uploaded["data"]["model_id"], "real-model")

    def test_task_does_not_hide_adapter_errors(self):
        failure = RuntimeError("training failed")
        adapter = types.SimpleNamespace(train=Mock(side_effect=failure))
        task_context = types.SimpleNamespace(update_state=Mock())
        with patch.object(
            self.module.registry, "get_adapter", return_value=adapter
        ):
            with self.assertRaises(RuntimeError) as raised:
                self.module.train_and_evaluate_isolation_forest_task(
                    task_context,
                    [[0.0], [1.0]],
                    [[2.0]],
                    [1],
                    {"n_estimators": 100},
                    "failed-model",
                )
        self.assertIs(raised.exception, failure)

    def test_failed_promotion_preserves_local_artifacts(self):
        metadata = {
            "training_parameters": {"n_estimators": 100},
            "n_features": 2,
        }
        evaluation = {
            key: index
            for index, key in enumerate(self.module.ISOLATION_FOREST_EVALUATION_KEYS)
        }

        def write_artifacts(_train, artifact_dir, **_kwargs):
            write_placeholder_artifacts(artifact_dir)
            return metadata

        adapter = types.SimpleNamespace(
            train=Mock(side_effect=write_artifacts),
            evaluate=Mock(return_value=evaluation),
        )
        task_context = types.SimpleNamespace(update_state=Mock())
        failure = IsolationForestPromotionError("remote unavailable")

        with tempfile.TemporaryDirectory() as directory, patch.object(
            self.module,
            "ISOLATION_FOREST_ARTIFACT_ROOT",
            Path(directory) / "isolation-forest",
        ), patch.object(
            self.module.registry, "get_adapter", return_value=adapter
        ), patch.object(
            self.module,
            "promote_isolation_forest_model",
            side_effect=failure,
        ):
            artifact_dir = Path(directory) / "isolation-forest" / "model-1"
            with self.assertRaises(IsolationForestPromotionError) as raised:
                self.module.train_and_evaluate_isolation_forest_task(
                    task_context,
                    [[0.0, 0.0], [0.1, 0.1]],
                    [[2.0, 2.0]],
                    [1],
                    {"n_estimators": 100},
                    "model-1",
                )
            self.assertTrue(artifact_dir.is_dir())
            self.assertEqual(
                sorted(path.name for path in artifact_dir.iterdir()),
                sorted(ARTIFACT_FILES.values()),
            )
        self.assertIs(raised.exception, failure)
        self.assertEqual(
            [call.kwargs["state"] for call in task_context.update_state.call_args_list],
            ["INITIATING", "TRAINING", "EVALUATING", "PROMOTING"],
        )

    def test_cleanup_failure_does_not_reverse_confirmed_promotion(self):
        metadata = {
            "training_parameters": {"n_estimators": 100},
            "n_features": 2,
        }
        evaluation = {
            key: index
            for index, key in enumerate(self.module.ISOLATION_FOREST_EVALUATION_KEYS)
        }
        adapter = types.SimpleNamespace(
            train=Mock(return_value=metadata),
            evaluate=Mock(return_value=evaluation),
        )
        task_context = types.SimpleNamespace(update_state=Mock())
        promotion = {
            "status": "available",
            "detector_id": "isolation-forest",
            "model_id": "model-1",
        }

        with patch.object(
            self.module.registry, "get_adapter", return_value=adapter
        ), patch.object(
            self.module,
            "promote_isolation_forest_model",
            return_value=promotion,
        ), patch.object(
            self.module.shutil,
            "rmtree",
            side_effect=OSError("cleanup denied"),
        ):
            with self.assertLogs(self.module.logger, level="WARNING") as logs:
                result = self.module.train_and_evaluate_isolation_forest_task(
                    task_context,
                    [[0.0, 0.0], [0.1, 0.1]],
                    [[2.0, 2.0]],
                    [1],
                    {"n_estimators": 100},
                    "model-1",
                )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["promotion"], promotion)
        self.assertIn("cleanup failed", logs.output[0])


class IsolationForestPromotionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.artifact_dir = Path(self.temporary_directory.name) / "model-1"
        self.contents = write_placeholder_artifacts(self.artifact_dir)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_binary_multipart_fields_checksums_and_idempotent_response(self):
        captured = {}

        def capture(url, *, data, files, timeout):
            captured["url"] = url
            captured["data"] = dict(data)
            captured["timeout"] = timeout
            captured["files"] = {
                field: (value[0], value[1].read(), value[2])
                for field, value in files.items()
            }
            return available_response("model-1", idempotent=True)

        with patch(
            "learning_adaptation.isolation_forest_promotion.requests.post",
            side_effect=capture,
        ):
            result = promote_isolation_forest_model(
                self.artifact_dir,
                "model-1",
                service_url="http://detector.test/base/",
            )

        self.assertEqual(
            result,
            {
                "status": "available",
                "detector_id": "isolation-forest",
                "model_id": "model-1",
            },
        )
        self.assertEqual(captured["url"], "http://detector.test/base/save_model")
        self.assertEqual(captured["timeout"], (5.0, 60.0))
        self.assertEqual(captured["data"]["detector_id"], "isolation-forest")
        self.assertEqual(captured["data"]["model_id"], "model-1")
        for field, filename in ARTIFACT_FILES.items():
            uploaded_name, uploaded_content, _content_type = captured["files"][field]
            self.assertEqual(uploaded_name, filename)
            self.assertEqual(uploaded_content, self.contents[filename])
            self.assertEqual(
                captured["data"][f"{field}_sha256"],
                hashlib.sha256(self.contents[filename]).hexdigest(),
            )

    def test_missing_artifact_is_a_targeted_promotion_failure(self):
        (self.artifact_dir / "model.joblib").unlink()
        with self.assertRaisesRegex(
            IsolationForestPromotionError, "missing model.joblib"
        ):
            promote_isolation_forest_model(
                self.artifact_dir, "model-1", service_url="http://detector.test"
            )

    def test_http_errors_are_promotion_failures_and_preserve_artifacts(self):
        for status_code in (400, 409, 500):
            with self.subTest(status_code=status_code), patch(
                "learning_adaptation.isolation_forest_promotion.requests.post",
                return_value=FakePromotionResponse(
                    status_code=status_code, payload={"error": "rejected"}, text="rejected"
                ),
            ):
                with self.assertRaisesRegex(
                    IsolationForestPromotionError, f"HTTP {status_code}"
                ):
                    promote_isolation_forest_model(
                        self.artifact_dir,
                        "model-1",
                        service_url="http://detector.test",
                    )
                self.assertTrue(self.artifact_dir.is_dir())

    def test_invalid_json_is_a_promotion_failure(self):
        with patch(
            "learning_adaptation.isolation_forest_promotion.requests.post",
            return_value=FakePromotionResponse(payload=ValueError("not JSON")),
        ), self.assertRaisesRegex(IsolationForestPromotionError, "invalid JSON"):
            promote_isolation_forest_model(
                self.artifact_dir, "model-1", service_url="http://detector.test"
            )

    def test_timeout_and_network_failures_are_promotion_failures(self):
        for failure in (
            requests.Timeout("timed out"),
            requests.ConnectionError("connection refused"),
        ):
            with self.subTest(failure=type(failure).__name__), patch(
                "learning_adaptation.isolation_forest_promotion.requests.post",
                side_effect=failure,
            ) as transport, self.assertRaisesRegex(
                IsolationForestPromotionError, "promotion request failed"
            ):
                promote_isolation_forest_model(
                    self.artifact_dir,
                    "model-1",
                    service_url="http://detector.test",
                )
            self.assertEqual(transport.call_count, 2)
            self.assertTrue(self.artifact_dir.is_dir())

    def test_remote_identity_and_status_mismatches_are_rejected(self):
        invalid_payloads = (
            {
                "status": "pending",
                "detector_id": "isolation-forest",
                "model_id": "model-1",
            },
            {
                "status": "available",
                "detector_id": "cgnn",
                "model_id": "model-1",
            },
            {
                "status": "available",
                "detector_id": "isolation-forest",
                "model_id": "other-model",
            },
        )
        expected_errors = ("availability", "detector_id mismatch", "model_id mismatch")
        for payload, expected_error in zip(invalid_payloads, expected_errors):
            with self.subTest(payload=payload), patch(
                "learning_adaptation.isolation_forest_promotion.requests.post",
                return_value=FakePromotionResponse(payload=payload),
            ), self.assertRaisesRegex(IsolationForestPromotionError, expected_error):
                promote_isolation_forest_model(
                    self.artifact_dir,
                    "model-1",
                    service_url="http://detector.test",
                )


if __name__ == "__main__":
    unittest.main()
