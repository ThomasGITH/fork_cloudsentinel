import importlib.util
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
        return lambda function: function


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
        adapter = types.SimpleNamespace(
            train=Mock(return_value=metadata),
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
        ) as resolver:
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
        self.assertEqual(result["artifact_dir"], "isolation-forest/model-1")
        self.assertEqual(
            [call.kwargs["state"] for call in task_context.update_state.call_args_list],
            ["INITIATING", "TRAINING", "EVALUATING"],
        )

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

        with tempfile.TemporaryDirectory() as directory, patch.object(
            self.module,
            "ISOLATION_FOREST_ARTIFACT_ROOT",
            Path(directory) / "isolation-forest",
        ):
            result = self.module.train_and_evaluate_isolation_forest_task(
                task_context,
                train.tolist(),
                test.tolist(),
                labels.tolist(),
                parameters,
                "real-model",
            )
            artifact_dir = Path(directory) / "isolation-forest" / "real-model"
            self.assertTrue((artifact_dir / "model.joblib").is_file())
            self.assertTrue((artifact_dir / "model_metadata.json").is_file())
            self.assertTrue((artifact_dir / "model_evaluation.json").is_file())
            stored_evaluation = json.loads(
                (artifact_dir / "model_evaluation.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["detector_id"], "isolation-forest")
        self.assertEqual(result["model_id"], "real-model")
        self.assertEqual(result["artifact_dir"], "isolation-forest/real-model")
        self.assertEqual(result["training_parameters"], parameters)
        self.assertEqual(result["n_features"], 2)
        self.assertNotIn("binary_predictions", result["evaluation"])
        self.assertNotIn("anomaly_scores", result["evaluation"])
        self.assertIn("binary_predictions", stored_evaluation)
        self.assertIn("anomaly_scores", stored_evaluation)
        json.dumps(result)
        self.assertLess(len(json.dumps(result)), 2_000)

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


if __name__ == "__main__":
    unittest.main()
