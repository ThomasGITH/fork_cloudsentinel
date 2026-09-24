import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, mock_open, patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeConfig:
    def update(self, *args, **kwargs):
        return None


class FakeCeleryApp:
    def __init__(self, *args, **kwargs):
        self.conf = FakeConfig()

    def task(self, *args, **kwargs):
        return lambda function: function


class FakeFrame:
    def __init__(self, values):
        self.values = values

    def to_numpy(self, dtype=None):
        return self

    def tolist(self):
        return self.values


class CGNNAdapterContractTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("detectors.cgnn.adapter", None)

    def test_importing_adapter_does_not_import_legacy_runtime(self):
        blocked = ("torch", "celery", "cgnn.train", "cgnn.evaluate_prediction", "predict")
        adapter_module = importlib.import_module("detectors.cgnn.adapter")

        self.assertIsNotNone(adapter_module.CGNNAdapter)
        self.assertEqual(
            [name for name in blocked if name in sys.modules],
            [],
        )

    def test_train_delegates_arguments_and_return_value(self):
        adapter_module = importlib.import_module("detectors.cgnn.adapter")
        implementation = Mock(return_value=("config", "importance"))
        runtime_module = types.SimpleNamespace(train=implementation)
        callback = Mock()

        with patch.object(adapter_module, "import_module", return_value=runtime_module) as importer:
            result = adapter_module.CGNNAdapter.train(
                {"dataset": "sample"}, "train", "test", "labels", progress_callback=callback
            )

        importer.assert_called_once_with("cgnn.train")
        implementation.assert_called_once_with(
            {"dataset": "sample"}, "train", "test", "labels", progress_callback=callback
        )
        self.assertEqual(result, ("config", "importance"))

    def test_evaluate_delegates_arguments_and_return_value(self):
        adapter_module = importlib.import_module("detectors.cgnn.adapter")
        implementation = Mock(return_value="evaluation")
        runtime_module = types.SimpleNamespace(predict_and_evaluate=implementation)
        callback = Mock()

        with patch.object(adapter_module, "import_module", return_value=runtime_module) as importer:
            result = adapter_module.CGNNAdapter.evaluate(
                {"id": "model"},
                "train",
                "test",
                "labels",
                progress_callback=callback,
                save_output=False,
            )

        importer.assert_called_once_with("cgnn.evaluate_prediction")
        implementation.assert_called_once_with(
            {"id": "model"},
            "train",
            "test",
            "labels",
            progress_callback=callback,
            save_output=False,
        )
        self.assertEqual(result, "evaluation")

    def test_predict_delegates_arguments_and_return_value(self):
        adapter_module = importlib.import_module("detectors.cgnn.adapter")
        implementation = Mock(return_value=12.5)
        runtime_module = types.SimpleNamespace(load_model_and_predict=implementation)

        with patch.object(adapter_module, "import_module", return_value=runtime_module) as importer:
            result = adapter_module.CGNNAdapter.predict("test", "model", save_output=False)

        importer.assert_called_once_with("predict")
        implementation.assert_called_once_with("test", "model", save_output=False)
        self.assertEqual(result, 12.5)

    def test_adapter_does_not_translate_runtime_errors(self):
        adapter_module = importlib.import_module("detectors.cgnn.adapter")
        error = RuntimeError("legacy failure")
        runtime_module = types.SimpleNamespace(train=Mock(side_effect=error))

        with patch.object(adapter_module, "import_module", return_value=runtime_module):
            with self.assertRaises(RuntimeError) as raised:
                adapter_module.CGNNAdapter.train({}, "train", "test", "labels")

        self.assertIs(raised.exception, error)


class CGNNServiceIntegrationTests(unittest.TestCase):
    def tearDown(self):
        for name in (
            "phase_b_learning_app",
            "phase_b_tasks",
            "phase_b_detection_app",
            "detectors.cgnn.adapter",
        ):
            sys.modules.pop(name, None)

    def common_service_stubs(self):
        fake_numpy = types.ModuleType("numpy")
        fake_numpy.float32 = object()
        fake_pandas = types.ModuleType("pandas")
        fake_pandas.read_csv = Mock(return_value=FakeFrame([[1.0, 2.0]]))
        fake_cors = types.ModuleType("flask_cors")
        fake_cors.CORS = lambda *args, **kwargs: None
        fake_requests = types.ModuleType("requests")
        fake_torch = types.ModuleType("torch")
        return {
            "numpy": fake_numpy,
            "pandas": fake_pandas,
            "flask_cors": fake_cors,
            "requests": fake_requests,
            "torch": fake_torch,
        }

    def test_training_request_keeps_accepted_task_id_flow(self):
        submitted_task = Mock()
        submitted_task.apply_async.return_value = types.SimpleNamespace(id="task-123")
        tasks_stub = types.ModuleType("tasks")
        tasks_stub.train_and_evaluate_task = submitted_task
        celery_stub = types.ModuleType("celery")
        celery_stub.Celery = FakeCeleryApp
        config_stub = types.ModuleType("cgnn.config")
        config_stub.set_config = Mock()
        config_stub.get_config = Mock(return_value={})
        config_stub.set_initial_config = Mock()
        cgnn_stub = types.ModuleType("cgnn")
        cgnn_stub.__path__ = []
        stubs = self.common_service_stubs()
        stubs.update(
            {
                "celery": celery_stub,
                "tasks": tasks_stub,
                "cgnn": cgnn_stub,
                "cgnn.config": config_stub,
            }
        )

        with patch.dict(sys.modules, stubs):
            module = load_module(
                "phase_b_learning_app", REPOSITORY_ROOT / "learning_adaptation" / "app.py"
            )
            response = module.app.test_client().post(
                "/cgnn_train_model",
                data={
                    "train_array": (io.BytesIO(b"1,2\n"), "train.csv"),
                    "test_array": (io.BytesIO(b"1,2\n"), "test.csv"),
                    "anomaly_label_array": (io.BytesIO(b"0\n"), "labels.csv"),
                    "train_info": json.dumps({"data": {"dataset": "sample"}}),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json(), {"task_id": "task-123"})
        submitted_task.apply_async.assert_called_once_with(
            args=[[[1.0, 2.0]], [[1.0, 2.0]], [[1.0, 2.0]], {"data": {"dataset": "sample"}}]
        )

    def test_celery_training_task_uses_adapter_for_train_and_evaluate(self):
        celery_stub = types.ModuleType("celery")
        celery_stub.Celery = FakeCeleryApp
        celery_stub.Task = object
        fake_numpy = types.ModuleType("numpy")
        fake_numpy.float32 = object()
        fake_numpy.array = lambda value, dtype=None: ("converted", value)

        with patch.dict(sys.modules, {"celery": celery_stub, "numpy": fake_numpy}):
            module = load_module(
                "phase_b_tasks", REPOSITORY_ROOT / "learning_adaptation" / "tasks.py"
            )
            self.assertNotIn("detectors.cgnn.adapter", sys.modules)
            task_context = types.SimpleNamespace(update_state=Mock(), retry=Mock())
            model_config = {"dataset": "sample", "id": "run-1", "feature_importance": False}
            train_result = (model_config, None)
            adapter = types.SimpleNamespace(
                train=Mock(return_value=train_result),
                evaluate=Mock(return_value="evaluation"),
            )
            with (
                patch.object(module.registry, "get_adapter", return_value=adapter) as get_adapter,
                patch.object(module.os, "makedirs"),
                patch("builtins.open", mock_open()),
            ):
                result = module.train_and_evaluate_task(
                    task_context,
                    [[1.0]],
                    [[2.0]],
                    [[0.0]],
                    {"data": {"dataset": "sample"}},
                )

        self.assertEqual(result, "Training Successful")
        get_adapter.assert_called_once_with("cgnn")
        adapter.train.assert_called_once()
        adapter.evaluate.assert_called_once()
        self.assertEqual(adapter.train.call_args.args[0], {"dataset": "sample"})
        self.assertIs(adapter.evaluate.call_args.args[0], model_config)
        self.assertIs(
            adapter.train.call_args.kwargs["progress_callback"],
            adapter.evaluate.call_args.kwargs["progress_callback"],
        )

    def test_detection_flow_calls_adapter_predict(self):
        config_stub = types.ModuleType("config")
        config_stub.set_config = Mock()
        config_stub.get_config = Mock(return_value={})
        config_stub.set_initial_config = Mock()
        stubs = self.common_service_stubs()
        stubs["config"] = config_stub
        expected_test_data = stubs["pandas"].read_csv.return_value

        with patch.dict(sys.modules, stubs), tempfile.TemporaryDirectory() as directory:
            module = load_module(
                "phase_b_detection_app", REPOSITORY_ROOT / "anomaly_detection" / "cgnn" / "app.py"
            )
            previous_directory = os.getcwd()
            os.chdir(directory)
            try:
                Path("results").mkdir()
                adapter = types.SimpleNamespace(predict=Mock(return_value=12.5))
                with patch.object(module.registry, "get_adapter", return_value=adapter) as get_adapter:
                    response = module.app.test_client().post(
                        "/detect_anomalies",
                        data={
                            "test_array": (io.BytesIO(b"1,2\n"), "test.csv"),
                            "test_info": json.dumps(
                                {
                                    "task_id": "monitor-1",
                                    "settings": {},
                                    "data": {
                                        "model": "saved-model",
                                        "crca_threshold": 50,
                                        "iteration": "0",
                                        "start_time": "start",
                                        "end_time": "end",
                                        "containers": ["service"],
                                        "metrics": ["cpu"],
                                        "data_interval": 60,
                                    },
                                }
                            ),
                        },
                        content_type="multipart/form-data",
                    )
                result_file = Path("results/monitor-1/cgnn_results.json")
                result_data = json.loads(result_file.read_text(encoding="utf-8"))
            finally:
                os.chdir(previous_directory)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "success")
        get_adapter.assert_called_once_with("cgnn")
        adapter.predict.assert_called_once_with(expected_test_data, "saved-model")
        self.assertEqual(result_data["results"]["0"]["percentage"], 12.5)


if __name__ == "__main__":
    unittest.main()
