import hashlib
import importlib.util
import json
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


class FakeConfig:
    def update(self, *args, **kwargs):
        return None


class FakeCelery:
    def __init__(self, *args, **kwargs):
        self.conf = FakeConfig()


def celery_stub():
    module = types.ModuleType("celery")
    module.Celery = FakeCelery
    return module


def config_stubs():
    package = types.ModuleType("cgnn")
    package.__path__ = []
    config = types.ModuleType("cgnn.config")
    config.set_config = Mock()
    config.get_config = Mock(return_value={})
    config.set_initial_config = Mock()
    return package, config


class TrainingRunApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset_root = self.root / "datasets"
        self.snapshot_root = self.root / "dataset_snapshots"
        self.run_root = self.root / "training_runs"
        self.dataset_id = "fixture-1"
        self.write_dataset()

        self.cgnn_task = Mock()
        self.cgnn_task.apply_async.return_value = types.SimpleNamespace(id="cgnn-task")
        self.if_task = Mock()
        self.if_task.apply_async.return_value = types.SimpleNamespace(id="if-task")
        tasks = types.ModuleType("tasks")
        tasks.train_and_evaluate_task = self.cgnn_task
        tasks.train_and_evaluate_isolation_forest_task = self.if_task
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
                "if_d2_learning_app",
                REPOSITORY_ROOT / "learning_adaptation" / "app.py",
            )
        self.module.app.config.update(
            TESTING=True,
            EXISTING_DATASETS_ROOT=str(self.dataset_root),
            DATASET_SNAPSHOT_STORAGE_ROOT=str(self.snapshot_root),
            TRAINING_RUN_STORAGE_ROOT=str(self.run_root),
        )
        self.client = self.module.app.test_client()

    def tearDown(self):
        sys.modules.pop("if_d2_learning_app", None)
        self.temporary_directory.cleanup()

    def write_dataset(self, *, labels=None):
        directory = self.dataset_root / self.dataset_id
        directory.mkdir(parents=True, exist_ok=True)
        train = np.array([[0.0, 1.0], [0.2, 1.2], [0.1, 0.8], [0.3, 1.1]])
        test = np.array([[0.1, 1.0], [5.0, 6.0], [0.2, 0.9]])
        labels = np.array([0, 1, 0]) if labels is None else np.asarray(labels)
        np.savetxt(directory / f"train_{self.dataset_id}", train, delimiter=",")
        np.savetxt(directory / f"test_{self.dataset_id}", test, delimiter=",")
        np.savetxt(directory / f"test_label_{self.dataset_id}", labels, delimiter=",")
        (directory / "details.json").write_text(
            json.dumps(
                {
                    "dataset": "Fixture 1",
                    "containers": ["service-a"],
                    "metrics": ["cpu", "memory"],
                    "step_size": 60,
                    "duration": 180,
                    "anomaly_sequence": False,
                    "data_entries": 3,
                }
            ),
            encoding="utf-8",
        )

    def payload(self, detectors):
        return {
            "dataset": {"source": "existing", "dataset_id": self.dataset_id},
            "detectors": detectors,
            "client_request_id": "ui-request-1",
        }

    def test_isolation_forest_run_snapshots_and_dispatches_one_child(self):
        response = self.client.post(
            "/training_runs",
            json=self.payload(
                [{"detector_id": "isolation-forest", "parameters": {"n_estimators": 25}}]
            ),
        )

        self.assertEqual(response.status_code, 202)
        body = response.get_json()
        self.assertRegex(body["run_id"], r"^run_[0-9a-f]{32}$")
        self.assertRegex(body["snapshot_id"], r"^snapshot_[0-9a-f]{32}$")
        child = body["children"][0]
        self.assertEqual(child["detector_id"], "isolation-forest")
        self.assertEqual(child["task_id"], "if-task")
        self.assertEqual(child["parameters"]["n_estimators"], 25)
        self.assertEqual(child["parameters"]["contamination"], "auto")
        self.assertFalse(body["physical_parallelism_guaranteed"])

        snapshot_directory = self.snapshot_root / body["snapshot_id"]
        self.assertEqual(
            sorted(path.name for path in snapshot_directory.iterdir()),
            ["labels.csv", "snapshot.json", "test.csv", "train.csv"],
        )
        metadata = json.loads((snapshot_directory / "snapshot.json").read_text())
        original_snapshot_train = (snapshot_directory / "train.csv").read_bytes()
        for logical_name in ("train", "test", "labels"):
            path = snapshot_directory / metadata["files"][logical_name]["filename"]
            self.assertEqual(
                metadata["files"][logical_name]["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        source_train = self.dataset_root / self.dataset_id / f"train_{self.dataset_id}"
        source_train.write_text("99,99\n", encoding="utf-8")
        self.assertEqual((snapshot_directory / "train.csv").read_bytes(), original_snapshot_train)
        self.assertTrue((self.run_root / body["run_id"] / "run.json").is_file())
        self.assertTrue(
            (self.run_root / body["run_id"] / "children" / "isolation-forest" / "input" / "train.csv").is_file()
        )
        self.if_task.apply_async.assert_called_once()
        self.cgnn_task.apply_async.assert_not_called()

    def test_cgnn_run_uses_safe_direct_dispatch_with_preprocessed_inputs(self):
        response = self.client.post(
            "/training_runs",
            json=self.payload([{"detector_id": "cgnn", "parameters": {"epochs": 3}}]),
        )

        self.assertEqual(response.status_code, 202)
        body = response.get_json()
        child = body["children"][0]
        self.assertEqual(child["task_id"], "cgnn-task")
        args = self.cgnn_task.apply_async.call_args.kwargs["args"]
        train = np.asarray(args[0])
        test = np.asarray(args[1])
        self.assertEqual(train.shape, (4, 2))
        self.assertEqual(test.shape, (3, 2))
        self.assertEqual(np.asarray(args[2]).shape, (3, 1))
        self.assertGreaterEqual(train.min(), 0.0)
        self.assertLessEqual(train.max(), 1.0)
        self.assertEqual(args[3]["data"]["epochs"], 3)
        self.assertEqual(args[3]["data"]["orchestration_model_id"], child["model_id"])
        self.if_task.apply_async.assert_not_called()

    def test_combined_run_keeps_parameters_and_child_inputs_separate(self):
        response = self.client.post(
            "/training_runs",
            json=self.payload(
                [
                    {"detector_id": "cgnn", "parameters": {"epochs": 2}},
                    {
                        "detector_id": "isolation-forest",
                        "parameters": {"n_estimators": 30},
                    },
                ]
            ),
        )

        self.assertEqual(response.status_code, 202)
        body = response.get_json()
        children = {child["detector_id"]: child for child in body["children"]}
        self.assertEqual(children["cgnn"]["parameters"]["epochs"], 2)
        self.assertNotIn("n_estimators", children["cgnn"]["parameters"])
        self.assertEqual(children["isolation-forest"]["parameters"]["n_estimators"], 30)
        self.assertNotIn("epochs", children["isolation-forest"]["parameters"])
        self.assertNotEqual(children["cgnn"]["model_id"], children["isolation-forest"]["model_id"])
        for detector_id in children:
            self.assertTrue(
                (self.run_root / body["run_id"] / "children" / detector_id / "input").is_dir()
            )
        self.cgnn_task.apply_async.assert_called_once()
        self.if_task.apply_async.assert_called_once()

    def test_run_snapshot_and_model_identifiers_are_unique(self):
        responses = [
            self.client.post(
                "/training_runs",
                json=self.payload([{"detector_id": "isolation-forest", "parameters": {}}]),
            ).get_json()
            for _ in range(2)
        ]
        self.assertNotEqual(responses[0]["run_id"], responses[1]["run_id"])
        self.assertNotEqual(responses[0]["snapshot_id"], responses[1]["snapshot_id"])
        self.assertNotEqual(
            responses[0]["children"][0]["model_id"],
            responses[1]["children"][0]["model_id"],
        )

    def test_invalid_detector_lists_and_parameters_dispatch_nothing(self):
        invalid_payloads = (
            self.payload([]),
            self.payload(
                [
                    {"detector_id": "cgnn", "parameters": {}},
                    {"detector_id": "cgnn", "parameters": {}},
                ]
            ),
            self.payload([{"detector_id": "unknown", "parameters": {}}]),
            self.payload([{"detector_id": "cgnn", "parameters": {"unknown": 1}}]),
            self.payload(
                [{"detector_id": "isolation-forest", "parameters": {"contamination": 0.1}}]
            ),
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.post("/training_runs", json=payload)
                self.assertEqual(response.status_code, 400)
        self.cgnn_task.apply_async.assert_not_called()
        self.if_task.apply_async.assert_not_called()

        malformed = self.client.post(
            "/training_runs", data="{", content_type="application/json"
        )
        self.assertEqual(malformed.status_code, 400)
        self.assertIn("error", malformed.get_json())

    def test_invalid_dataset_is_rejected_before_dispatch(self):
        payload = self.payload([{"detector_id": "cgnn", "parameters": {}}])
        payload["dataset"]["dataset_id"] = "../escape"
        response = self.client.post("/training_runs", json=payload)
        self.assertEqual(response.status_code, 400)

        payload["dataset"]["dataset_id"] = "missing"
        response = self.client.post("/training_runs", json=payload)
        self.assertEqual(response.status_code, 400)
        self.cgnn_task.apply_async.assert_not_called()
        self.if_task.apply_async.assert_not_called()

    def test_detector_data_validation_happens_before_any_dispatch(self):
        self.write_dataset(labels=[0, 1])
        response = self.client.post(
            "/training_runs",
            json=self.payload(
                [
                    {"detector_id": "cgnn", "parameters": {}},
                    {"detector_id": "isolation-forest", "parameters": {}},
                ]
            ),
        )
        self.assertEqual(response.status_code, 400)
        self.cgnn_task.apply_async.assert_not_called()
        self.if_task.apply_async.assert_not_called()

    def test_one_dispatch_failure_does_not_cancel_the_other_child(self):
        self.cgnn_task.apply_async.side_effect = RuntimeError("broker rejected CGNN")
        response = self.client.post(
            "/training_runs",
            json=self.payload(
                [
                    {"detector_id": "cgnn", "parameters": {}},
                    {"detector_id": "isolation-forest", "parameters": {}},
                ]
            ),
        )
        self.assertEqual(response.status_code, 202)
        children = {item["detector_id"]: item for item in response.get_json()["children"]}
        self.assertEqual(children["cgnn"]["status"], "dispatch_failed")
        self.assertIn("broker rejected", children["cgnn"]["dispatch_error"])
        self.assertEqual(children["isolation-forest"]["task_id"], "if-task")
        self.if_task.apply_async.assert_called_once()

    def test_no_successful_dispatch_returns_service_error_and_preserves_run(self):
        self.if_task.apply_async.side_effect = RuntimeError("broker unavailable")
        response = self.client.post(
            "/training_runs",
            json=self.payload([{"detector_id": "isolation-forest", "parameters": {}}]),
        )
        self.assertEqual(response.status_code, 503)
        body = response.get_json()
        self.assertEqual(body["status"], "failed")
        self.assertTrue((self.run_root / body["run_id"] / "run.json").is_file())

    def test_get_run_aggregates_child_success_and_failure(self):
        created = self.client.post(
            "/training_runs",
            json=self.payload(
                [
                    {"detector_id": "cgnn", "parameters": {}},
                    {"detector_id": "isolation-forest", "parameters": {}},
                ]
            ),
        ).get_json()
        tasks = {
            "cgnn-task": types.SimpleNamespace(state="SUCCESS", info="Training Successful"),
            "if-task": types.SimpleNamespace(state="FAILURE", info=RuntimeError("failed")),
        }
        self.module.app.config["TRAINING_RUN_STATUS_READER"] = tasks.__getitem__

        response = self.client.get(f"/training_runs/{created['run_id']}")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["status"], "partial_success")
        statuses = {child["detector_id"]: child["status"] for child in body["children"]}
        self.assertEqual(statuses, {"cgnn": "completed", "isolation-forest": "failed"})
        self.assertNotIn("dataset_contents", body)

    def test_get_run_reports_queued_running_completed_and_missing(self):
        created = self.client.post(
            "/training_runs",
            json=self.payload([{"detector_id": "isolation-forest", "parameters": {}}]),
        ).get_json()
        for celery_state, expected in (
            ("PENDING", "queued"),
            ("TRAINING", "running"),
            ("SUCCESS", "completed"),
        ):
            with self.subTest(celery_state=celery_state):
                self.module.app.config["TRAINING_RUN_STATUS_READER"] = lambda _task_id, state=celery_state: types.SimpleNamespace(state=state, info=None)
                body = self.client.get(f"/training_runs/{created['run_id']}").get_json()
                self.assertEqual(body["status"], expected)
                self.assertEqual(body["children"][0]["status"], expected)
        self.assertEqual(self.client.get("/training_runs/missing").status_code, 404)
        self.assertEqual(self.client.get("/training_runs/%2E%2E").status_code, 400)


if __name__ == "__main__":
    unittest.main()
