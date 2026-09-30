import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from flask import Flask

from detectors import registry
from detectors.api import create_detectors_blueprint
from detectors.contracts import (
    DetectorCompatibilityError,
    DetectorExecutionError,
    DetectorPromotionError,
)
from learning_adaptation.plugin_training import execute_detector_plugin_training
from learning_adaptation.training_runs_api import (
    configure_training_run_defaults,
    create_training_runs_blueprint,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PluginTrainingAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plugin_root = self.root / "detectors"
        self.dataset_root = self.root / "datasets"
        self.snapshot_root = self.root / "snapshots"
        self.run_root = self.root / "runs"
        self.model_root = self.root / "models"
        self.artifact_root = self.root / "artifacts"
        self.plugin_id = "fixture-metrics-detector"
        self.module_name = "fixture_metrics_plugin.adapter"
        self.write_plugin(self.plugin_id, self.module_name)
        self.write_invalid_plugin()
        self.write_dataset()

        self.task = Mock()
        self.task.apply_async.side_effect = lambda *, args: types.SimpleNamespace(
            id=f"task-{args[0]['detector_id']}"
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
        self.app.register_blueprint(create_detectors_blueprint())
        self.app.register_blueprint(
            create_training_runs_blueprint(self.task, status_reader=lambda _task_id: None)
        )
        self.environment = patch.dict(
            os.environ,
            {
                "DETECTOR_PLUGIN_ROOT": str(self.plugin_root),
                "DATASET_SNAPSHOT_STORAGE_ROOT": str(self.snapshot_root),
                "TRAINING_RUN_STORAGE_ROOT": str(self.run_root),
                "MODEL_CATALOGUE_STORAGE_ROOT": str(self.model_root),
                "TRAINED_MODELS_TEMP_ROOT": str(self.artifact_root),
            },
        )
        self.environment.start()
        sys.path.insert(0, str(self.plugin_root))
        self.client = self.app.test_client()

    def tearDown(self):
        self.environment.stop()
        if str(self.plugin_root) in sys.path:
            sys.path.remove(str(self.plugin_root))
        for name in tuple(sys.modules):
            if name.startswith("fixture_metrics_plugin") or name.startswith(
                "fixture_sibling_plugin"
            ):
                sys.modules.pop(name, None)
        self.temporary.cleanup()

    def manifest(self, detector_id, entry_point, *, protocol="cloudsentinel.training/v1"):
        return f"""schema_version: 1
id: {detector_id}
name: Fixture metrics detector
version: 2.1.0
description: A deterministic protocol acceptance fixture.
supported_modalities: [metrics]
capabilities:
  training:
    enabled: true
    protocol: {protocol}
    input_profile: metrics-partition-v1
input_requirements:
  training:
    labels_required: true
training_parameters:
  marker:
    type: integer
    default: 7
    minimum: 1
    description: Value written to the dummy artifact.
  failure_mode:
    type: string
    default: normal
    allowed_values: [normal, compatibility, runtime, promotion, outside, missing]
    advanced: true
    description: Test-only failure selection.
entry_point: {entry_point}
"""

    def adapter_source(self):
        return '''import json
from detectors.contracts import (
    ArtifactResult,
    DetectorCompatibilityError,
    DetectorExecutionError,
    DetectorPromotionError,
    PromotionResult,
    TrainingResult,
)

IMPORTED = True

class FixtureAdapter:
    def validate_training(self, context):
        if context.parameters["failure_mode"] == "compatibility":
            raise DetectorCompatibilityError("fixture data is incompatible")

    def run_training(self, context, progress):
        mode = context.parameters["failure_mode"]
        if mode == "runtime":
            raise DetectorExecutionError("fixture runtime failed")
        if mode == "promotion":
            raise DetectorPromotionError("fixture promotion failed")
        directory = context.suggested_artifact_directory
        if mode == "outside":
            directory = context.artifact_workspace.parent / "outside-artifact"
        directory.mkdir(parents=True, exist_ok=True)
        if mode != "missing":
            (directory / "dummy.model").write_text(
                json.dumps({"marker": context.parameters["marker"]}), encoding="utf-8"
            )
        progress.report("TRAINING", "fixture trained")
        return TrainingResult(
            status="completed",
            artifact=ArtifactResult(
                directory=directory,
                required_files=("dummy.model",),
                format="fixture-binary",
                safe_reference=context.model_id,
            ),
            evaluation={"f1": 0.75},
            promotion=PromotionResult(status="not_applicable"),
            model_metadata={"marker": context.parameters["marker"]},
        )
'''

    def write_plugin(self, detector_id, module_name, *, protocol="cloudsentinel.training/v1"):
        folder_name = module_name.split(".")[0]
        folder = self.plugin_root / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "manifest.yaml").write_text(
            self.manifest(
                detector_id,
                f"{module_name}:FixtureAdapter",
                protocol=protocol,
            ),
            encoding="utf-8",
        )
        (folder / "adapter.py").write_text(self.adapter_source(), encoding="utf-8")

    def write_invalid_plugin(self):
        folder = self.plugin_root / "broken_fixture"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "manifest.yaml").write_text(
            "schema_version: 1\nid: Broken ID\n", encoding="utf-8"
        )
        (folder / "adapter.py").write_text(
            "raise RuntimeError('must never import')\n", encoding="utf-8"
        )

    def write_dataset(self):
        dataset_id = "fixture-dataset"
        directory = self.dataset_root / dataset_id
        directory.mkdir(parents=True)
        np.savetxt(directory / f"train_{dataset_id}", [[0, 1], [1, 2]], delimiter=",")
        np.savetxt(directory / f"test_{dataset_id}", [[2, 3], [3, 4]], delimiter=",")
        np.savetxt(directory / f"test_label_{dataset_id}", [0, 1], delimiter=",")
        (directory / "details.json").write_text(
            json.dumps({"containers": ["svc"], "metrics": ["cpu", "memory"]}),
            encoding="utf-8",
        )

    def create_run(self, *, detector_id=None, parameters=None, run_name=None):
        payload = {
            "dataset": {"source": "existing", "dataset_id": "fixture-dataset"},
            "detectors": [
                {
                    "detector_id": detector_id or self.plugin_id,
                    "parameters": parameters or {},
                }
            ],
        }
        if run_name is not None:
            payload["run_name"] = run_name
        response = self.client.post("/training_runs", json=payload)
        self.assertEqual(response.status_code, 202, response.get_json())
        context_payload = self.task.apply_async.call_args.kwargs["args"][0]
        return response.get_json(), context_payload

    def test_third_plugin_is_lazy_discovered_dispatched_and_catalogued(self):
        self.assertNotIn(self.module_name, sys.modules)
        response = self.client.get("/detectors")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual([item["id"] for item in body["detectors"]], [self.plugin_id])
        self.assertEqual(body["discovery_errors"][0]["plugin"], "broken_fixture")
        self.assertNotIn(self.module_name, sys.modules)

        run, context_payload = self.create_run(
            run_name="Fixture protocol acceptance"
        )
        self.assertEqual(run["run_name"], "Fixture protocol acceptance")
        self.assertEqual(run["children"][0]["parameters"]["marker"], 7)
        self.assertEqual(context_payload["plugin"]["detector_id"], self.plugin_id)
        self.assertEqual(context_payload["plugin"]["source"], "external")
        self.assertNotIn(self.module_name, sys.modules)

        result = execute_detector_plugin_training(context_payload, lambda *_args: None)
        self.assertEqual(result["status"], "completed")
        self.assertIn(self.module_name, sys.modules)
        stored_run = self.client.get(f"/training_runs/{run['run_id']}").get_json()
        child = stored_run["children"][0]
        self.assertEqual(child["status"], "completed")
        self.assertEqual(child["model_status"], "available")
        self.assertEqual(stored_run["run_name"], "Fixture protocol acceptance")
        history = self.client.get("/training_runs").get_json()
        self.assertEqual(history["items"][0]["run_name"], "Fixture protocol acceptance")

        model = self.client.get(f"/models/{child['model_id']}").get_json()
        self.assertEqual(model["detector_id"], self.plugin_id)
        self.assertEqual(model["detector_version"], "2.1.0")
        self.assertEqual(model["dataset"]["dataset_id"], "fixture-dataset")
        self.assertEqual(model["snapshot"]["snapshot_id"], run["snapshot_id"])
        self.assertEqual(model["feature_identity"]["feature_order"], ["svc_cpu", "svc_memory"])
        self.assertEqual(model["training_parameters"]["marker"], 7)
        self.assertEqual(model["artifact"]["format"], "fixture-binary")
        self.assertEqual(model["plugin"]["detector_id"], self.plugin_id)
        self.assertEqual(model["plugin"]["source"], "external")

    def test_compatibility_and_runtime_failures_are_durable_without_models(self):
        exception_types = {
            "compatibility": DetectorCompatibilityError,
            "runtime": DetectorExecutionError,
            "promotion": DetectorPromotionError,
            "outside": DetectorExecutionError,
            "missing": DetectorExecutionError,
        }
        for mode, exception_type in exception_types.items():
            with self.subTest(mode=mode):
                self.task.reset_mock()
                run, payload = self.create_run(parameters={"failure_mode": mode})
                with self.assertRaises(exception_type):
                    execute_detector_plugin_training(payload, lambda *_args: None)
                stored = self.client.get(f"/training_runs/{run['run_id']}").get_json()
                expected = "validation_failed" if mode == "compatibility" else "failed"
                self.assertEqual(stored["children"][0]["status"], expected)
                self.assertEqual(self.client.get("/models").get_json()["total"], 0)

    def test_sibling_compatibility_failure_does_not_block_success(self):
        sibling_id = "fixture-sibling"
        self.write_plugin(sibling_id, "fixture_sibling_plugin.adapter")
        response = self.client.post(
            "/training_runs",
            json={
                "dataset": {"source": "existing", "dataset_id": "fixture-dataset"},
                "detectors": [
                    {"detector_id": self.plugin_id, "parameters": {"failure_mode": "compatibility"}},
                    {"detector_id": sibling_id, "parameters": {}},
                ],
            },
        )
        self.assertEqual(response.status_code, 202, response.get_json())
        payloads = [call.kwargs["args"][0] for call in self.task.apply_async.call_args_list]
        for payload in payloads:
            if payload["detector_id"] == self.plugin_id:
                with self.assertRaises(DetectorCompatibilityError):
                    execute_detector_plugin_training(payload, lambda *_args: None)
            else:
                execute_detector_plugin_training(payload, lambda *_args: None)
        stored = self.client.get(f"/training_runs/{response.get_json()['run_id']}").get_json()
        states = {child["detector_id"]: child["status"] for child in stored["children"]}
        self.assertEqual(states[self.plugin_id], "validation_failed")
        self.assertEqual(states[sibling_id], "completed")
        self.assertEqual(stored["status"], "partial_success")

    def test_duplicate_ids_and_unknown_protocol_are_isolated(self):
        self.write_plugin("unaffected-plugin", "unaffected_plugin.adapter")
        self.write_plugin(self.plugin_id, "fixture_sibling_plugin.adapter")
        report = registry.scan_detectors(self.plugin_root)
        self.assertEqual(
            [manifest["id"] for manifest in report["detectors"]],
            ["unaffected-plugin"],
        )
        self.assertEqual(
            {error["code"] for error in report["discovery_errors"]},
            {"duplicate_detector_id", "invalid_manifest"},
        )

        protocol_root = self.root / "protocol-detectors"
        old_root = self.plugin_root
        self.plugin_root = protocol_root
        try:
            self.write_plugin(
                "future-protocol", "future_protocol_plugin.adapter", protocol="cloudsentinel.training/v99"
            )
        finally:
            self.plugin_root = old_root
        manifest = registry.scan_detectors(protocol_root)["detectors"][0]
        self.assertEqual(manifest["id"], "future-protocol")
        with self.assertRaises(registry.UnsupportedTrainingProtocolError):
            registry.get_training_adapter("future-protocol", protocol_root)

    def test_adapter_import_and_contract_failures_are_targeted(self):
        broken_root = self.root / "adapter-errors"
        for folder_name, detector_id, entry_point in (
            ("missing_import", "missing-import", "module_that_does_not_exist:Adapter"),
            ("missing_contract", "missing-contract", "missing_contract.adapter:Adapter"),
        ):
            folder = broken_root / folder_name
            folder.mkdir(parents=True)
            (folder / "manifest.yaml").write_text(
                self.manifest(detector_id, entry_point), encoding="utf-8"
            )
        contract_package = self.plugin_root / "missing_contract"
        contract_package.mkdir(parents=True)
        (contract_package / "adapter.py").write_text(
            "class Adapter:\n    pass\n", encoding="utf-8"
        )
        with self.assertRaises(registry.AdapterImportError):
            registry.get_training_adapter("missing-import", broken_root)
        with self.assertRaises(registry.AdapterContractError):
            registry.get_training_adapter("missing-contract", broken_root)

    def test_production_code_has_no_fixture_detector_coupling(self):
        for root_name in ("detectors", "learning_adaptation"):
            for path in (REPOSITORY_ROOT / root_name).rglob("*.py"):
                self.assertNotIn(self.plugin_id, path.read_text(encoding="utf-8"), str(path))


if __name__ == "__main__":
    unittest.main()
