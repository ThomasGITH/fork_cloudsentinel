import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from anomaly_detection.generic.app import create_app
from anomaly_detection.generic.validation import DetectionRequestError, parse_metadata
from detectors import registry
from detectors.contracts import ArtifactResult, ModelLoadContext
from detectors.model_artifacts import ModelArtifactError, ModelArtifactStore
from detectors.plugin_repository import PluginRepository
from learning_adaptation.model_catalogue import ModelCatalogueStore
from learning_adaptation.training_lifecycle import public_model_record


ROOT = Path(__file__).resolve().parents[1]
LOF_SOURCE = ROOT / "examples" / "external_plugins" / "local_outlier_factor"


class GenericDetectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_root = self.root / "models"
        self.artifact_root = self.root / "artifacts"
        self.result_root = self.root / "results"
        self.models = ModelCatalogueStore(self.model_root)
        self.artifacts = ModelArtifactStore(self.artifact_root)

    def tearDown(self):
        for name in tuple(__import__("sys").modules):
            if name.startswith("_cloudsentinel_external_plugins"):
                __import__("sys").modules.pop(name, None)
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

    def test_live_monitoring_context_is_bounded_and_validated(self):
        metadata = parse_metadata(
            json.dumps(
                {
                    "model_id": "model_live",
                    "feature_order": ["cpu"],
                    "context": {
                        "live_monitoring_session_id": "monitor_123",
                        "recipe_sha256": "a" * 64,
                    },
                }
            )
        )
        self.assertEqual(
            metadata["context"]["live_monitoring_session_id"], "monitor_123"
        )
        with self.assertRaisesRegex(DetectionRequestError, "recipe_sha256"):
            parse_metadata(
                json.dumps(
                    {
                        "model_id": "model_live",
                        "feature_order": ["cpu"],
                        "context": {
                            "live_monitoring_session_id": "monitor_123",
                            "recipe_sha256": "not-a-digest",
                        },
                    }
                )
            )

    @staticmethod
    def feature_identity():
        order = ["cpu", "memory"]
        from anomaly_detection.generic.validation import feature_order_sha256

        return {
            "feature_order": order,
            "feature_order_sha256": feature_order_sha256(order),
        }

    def model_record(
        self,
        model_id,
        detector_id,
        detector_version,
        published,
        artifact_format,
        *,
        plugin=None,
        model_metadata=None,
        status="ready",
    ):
        inference = {"status": status}
        if status == "ready":
            inference.update(
                {
                    "artifact_id": published["artifact_id"],
                    "artifact_manifest_sha256": published["manifest_sha256"],
                    "contract": "cloudsentinel.inference/v1",
                    "artifact_format": artifact_format,
                }
            )
        record = {
            "schema_version": 1,
            "model_id": model_id,
            "detector_id": detector_id,
            "detector_version": detector_version,
            "status": "available",
            "created_at": "2026-09-30T00:00:00+00:00",
            "updated_at": "2026-09-30T00:00:00+00:00",
            "training_run_id": "run_generic_detection",
            "training_child_id": "child_generic_detection",
            "dataset": {"source": "catalogue", "dataset_id": "dataset-one", "version": 1, "partition_id": "partition-one"},
            "snapshot": {"snapshot_id": "snapshot-one", "snapshot_sha256": "a" * 64},
            "feature_identity": self.feature_identity(),
            "training_parameters": {},
            "evaluation": {"f1": 1.0},
            "promotion": {"status": "not_applicable", "promoted_at": None, "safe_reference": None},
            "artifact": {"format": artifact_format, "safe_reference": model_id},
            "model_metadata": model_metadata or {"n_features": 2},
            "inference": inference,
        }
        if plugin:
            record["plugin"] = plugin
        self.models.write(record)
        return record

    def publish(self, model_id, source, required, detector_id, version, fmt, plugin=None):
        return self.artifacts.publish(
            model_id,
            ArtifactResult(directory=source, required_files=tuple(required), format=fmt),
            detector_id=detector_id,
            detector_version=version,
            artifact_format=fmt,
            feature_identity=self.feature_identity(),
            plugin_reference=plugin,
            inference_contract="cloudsentinel.inference/v1",
        )

    def app(self, **extra):
        config = {
            "TESTING": True,
            "MODEL_CATALOGUE_STORAGE_ROOT": str(self.model_root),
            "MODEL_ARTIFACT_STORAGE_ROOT": str(self.artifact_root),
            "GENERIC_DETECTION_RESULTS_ROOT": str(self.result_root),
            "MODEL_CACHE_SIZE": 2,
            "MAX_CONTENT_LENGTH": 1024 * 1024,
            "MAX_OBSERVATIONS": 1000,
            "MAX_FEATURES": 20,
        }
        config.update(extra)
        return create_app(config)

    def multipart(self, model_id, matrix, *, order=None):
        output = io.StringIO()
        np.savetxt(output, matrix, delimiter=",")
        metadata = {
            "model_id": model_id,
            "feature_order": order or ["cpu", "memory"],
        }
        return {
            "matrix": (io.BytesIO(output.getvalue().encode()), "matrix.csv"),
            "metadata": json.dumps(metadata),
        }

    def multipart_metadata(self, matrix, metadata):
        output = io.StringIO()
        np.savetxt(output, matrix, delimiter=",")
        return {
            "matrix": (io.BytesIO(output.getvalue().encode()), "matrix.csv"),
            "metadata": json.dumps(metadata),
        }

    def create_if_model(self, model_id="model_if_generic"):
        from detectors.isolation_forest.implementation import train_model

        source = self.root / f"source-{model_id}"
        train = np.random.default_rng(41).normal(size=(60, 2))
        train_model(train, source, model_id=model_id)
        (source / "model_evaluation.json").write_text("{}\n", encoding="utf-8")
        plugin = registry.get_plugin_reference("isolation-forest")
        published = self.publish(
            model_id,
            source,
            ("model.joblib", "model_metadata.json", "model_evaluation.json"),
            "isolation-forest",
            "1.0.0",
            "sklearn-pipeline-joblib-v1",
            plugin,
        )
        self.model_record(
            model_id,
            "isolation-forest",
            "1.0.0",
            published,
            "sklearn-pipeline-joblib-v1",
            plugin=plugin,
            model_metadata={"n_features": 2},
        )
        return source, train

    def test_health_is_lazy_and_unknown_and_not_ready_statuses_are_safe(self):
        app = self.app()
        with patch("detectors.registry.get_inference_adapter", side_effect=AssertionError):
            response = app.test_client().get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(app.test_client().post("/models/missing/activate").status_code, 404)

        record = self.model_record(
            "model_legacy",
            "cgnn",
            "1.0.0",
            {},
            "cgnn-state-dict-v1",
            status="legacy_external",
        )
        self.assertEqual(app.test_client().post("/models/model_legacy/activate").status_code, 409)
        public = public_model_record(record)
        self.assertEqual(public["inference"]["status"], "legacy_external")
        self.assertNotIn("path", json.dumps(public).lower())

    def test_artifact_publication_is_atomic_immutable_and_detects_tampering(self):
        source = self.root / "artifact-source"
        source.mkdir()
        (source / "model.bin").write_bytes(b"stable-model")
        published = self.publish(
            "model_artifact_test", source, ("model.bin",), "fixture", "1.0.0", "fixture-v1"
        )
        self.assertTrue((published["directory"] / "artifact_manifest.json").is_file())
        self.assertEqual(self.artifacts.verify("model_artifact_test")["manifest_sha256"], published["manifest_sha256"])
        target = published["directory"] / "files" / "model.bin"
        target.chmod(0o600)
        target.write_bytes(b"tampered")
        with self.assertRaisesRegex(ModelArtifactError, "integrity"):
            self.artifacts.verify("model_artifact_test")

        unsafe = self.root / "unsafe-source"
        unsafe.mkdir()
        (unsafe / "real").write_text("x", encoding="utf-8")
        (unsafe / "link").symlink_to(unsafe / "real")
        with self.assertRaises(ModelArtifactError):
            self.publish("model_symlink", unsafe, ("link",), "fixture", "1", "fixture-v1")
        with self.assertRaises(ModelArtifactError):
            self.publish("model_missing", unsafe, ("missing",), "fixture", "1", "fixture-v1")

    def test_real_isolation_forest_activation_detection_cache_and_validation(self):
        source, train = self.create_if_model()
        app = self.app()
        client = app.test_client()
        self.assertEqual(client.post("/models/model_if_generic/activate").status_code, 200)
        self.assertEqual(client.post("/models/model_if_generic/activate").status_code, 200)
        status = client.get("/models/model_if_generic/runtime-status").get_json()
        self.assertTrue(status["activated"])

        test = np.vstack([train[:5], [[8.0, 8.0]]])
        with patch.object(StandardScaler, "fit", side_effect=AssertionError("must not refit")):
            response = client.post("/detect", data=self.multipart("model_if_generic", test))
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertEqual(body["detector_id"], "isolation-forest")
        self.assertEqual(body["prediction_count"], test.shape[0])
        legacy = registry.get_adapter("isolation-forest").predict(test, source)
        self.assertEqual(
            body["anomaly_count"], int(np.asarray(legacy["binary_predictions"]).sum())
        )
        self.assertAlmostEqual(body["anomaly_percentage"], legacy["anomaly_percentage"])
        self.assertNotIn("predictions", body)
        self.assertNotIn("scores", body)
        stored = json.loads(next(self.result_root.glob("*.json")).read_text(encoding="utf-8"))
        self.assertNotIn("binary_predictions", stored)
        self.assertNotIn("anomaly_scores", stored)

        mismatch = client.post(
            "/detect", data=self.multipart("model_if_generic", test, order=["memory", "cpu"])
        )
        self.assertEqual(mismatch.status_code, 400)
        url_context = client.post(
            "/detect",
            data=self.multipart_metadata(
                test,
                {
                    "model_id": "model_if_generic",
                    "feature_order": ["cpu", "memory"],
                    "context": {"service_url": "http://internal.example"},
                },
            ),
        )
        self.assertEqual(url_context.status_code, 400)
        hash_only_wrong_width = client.post(
            "/detect",
            data=self.multipart_metadata(
                np.column_stack([test, np.ones(test.shape[0])]),
                {
                    "model_id": "model_if_generic",
                    "feature_order_sha256": self.feature_identity()["feature_order_sha256"],
                },
            ),
        )
        self.assertEqual(hash_only_wrong_width.status_code, 400)
        non_finite = test.copy()
        non_finite[0, 0] = np.nan
        self.assertEqual(
            client.post("/detect", data=self.multipart("model_if_generic", non_finite)).status_code,
            400,
        )
        limited = self.app(MAX_CONTENT_LENGTH=100).test_client()
        self.assertEqual(
            limited.post("/detect", data=self.multipart("model_if_generic", test)).status_code,
            413,
        )

    def test_internal_evaluation_uses_activated_generic_model_and_is_bounded(self):
        _source, train = self.create_if_model("model_if_evaluation")
        client = self.app(MAX_EVALUATION_OUTPUTS=10).test_client()
        matrix = train[:5]
        before_activation = client.post(
            "/internal/evaluate",
            data=self.multipart("model_if_evaluation", matrix),
        )
        self.assertEqual(before_activation.status_code, 409)
        self.assertEqual(
            client.post("/models/model_if_evaluation/activate").status_code, 200
        )
        response = client.post(
            "/internal/evaluate",
            data=self.multipart_metadata(
                matrix,
                {
                    "model_id": "model_if_evaluation",
                    "feature_order": ["cpu", "memory"],
                    "timestamps": [
                        f"2026-10-01T10:0{index}:00Z" for index in range(5)
                    ],
                },
            ),
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertEqual(len(payload["binary_predictions"]), 5)
        self.assertEqual(len(payload["anomaly_scores"]), 5)
        self.assertGreaterEqual(payload["runtime_ms"], 0)
        self.assertEqual(len(payload["artifact_manifest_sha256"]), 64)
        limited = self.app(MAX_EVALUATION_OUTPUTS=2).test_client()
        self.assertEqual(
            limited.post(
                "/internal/evaluate",
                data=self.multipart("model_if_evaluation", matrix),
            ).status_code,
            400,
        )

    def test_unknown_inference_protocol_and_missing_contract_are_rejected(self):
        plugin_root = self.root / "protocol-fixtures"
        module_name = "generic_inference_protocol_fixture"
        module_file = plugin_root / f"{module_name}.py"
        plugin_root.mkdir()
        module_file.write_text(
            "class MissingInferenceMethods:\n"
            "    def validate_training(self, context): pass\n"
            "    def run_training(self, context, progress): pass\n",
            encoding="utf-8",
        )
        sys.path.insert(0, str(plugin_root))
        try:
            for detector_id, protocol in (
                ("unknown-inference-protocol", "cloudsentinel.inference/v999"),
                ("missing-inference-contract", "cloudsentinel.inference/v1"),
            ):
                directory = plugin_root / detector_id
                directory.mkdir()
                (directory / "manifest.yaml").write_text(
                    f"""schema_version: 1
id: {detector_id}
name: Inference fixture
version: 1.0.0
description: Inference protocol validation fixture.
supported_modalities: [metrics]
capabilities:
  training:
    enabled: true
    protocol: cloudsentinel.training/v1
    input_profile: metrics-partition-v1
  inference:
    enabled: true
    protocol: {protocol}
    input_profile: metrics-matrix/v1
    artifact_format: fixture-v1
input_requirements:
  training:
    labels_required: true
training_parameters: {{}}
entry_point: {module_name}:MissingInferenceMethods
""",
                    encoding="utf-8",
                )
            with self.assertRaises(registry.UnsupportedInferenceProtocolError):
                registry.get_inference_adapter(
                    "unknown-inference-protocol", detector_dir=plugin_root
                )
            with self.assertRaises(registry.AdapterContractError):
                registry.get_inference_adapter(
                    "missing-inference-contract", detector_dir=plugin_root
                )
        finally:
            sys.path.remove(str(plugin_root))
            sys.modules.pop(module_name, None)

    def test_external_lof_loads_by_pinned_reference_through_same_route(self):
        repository = PluginRepository((self.root / "plugin-repository").resolve())
        installed = repository.install(LOF_SOURCE)
        repository.activate("local-outlier-factor", "1.0.0")
        plugin = None
        environment = patch.dict(os.environ, {"DETECTOR_PLUGIN_ROOTS": str(repository.active)}, clear=False)
        environment.start()
        try:
            plugin = registry.get_plugin_reference("local-outlier-factor")
            adapter = registry.get_training_adapter("local-outlier-factor", plugin_reference=plugin)
            implementation = __import__(adapter.__class__.__module__.rsplit(".", 1)[0] + ".implementation", fromlist=["*"])
            source = self.root / "lof-source"
            train = np.random.default_rng(8).normal(0, 0.25, size=(50, 2))
            parameters = implementation.validate_parameters({"n_neighbors": 10})
            implementation.train_and_persist(
                train,
                source,
                parameters=parameters,
                model_id="model_lof_generic",
                feature_identity=self.feature_identity(),
                train_observations=50,
                test_observations=3,
                label_source={"type": "snapshot_labels"},
            )
            implementation.evaluate_and_persist(
                np.array([[0.0, 0.0], [6.0, 6.0], [-7.0, 7.0]]),
                np.array([0, 1, 1]),
                source,
            )
            published = self.publish(
                "model_lof_generic",
                source,
                ("model.joblib", "model_metadata.json", "model_evaluation.json"),
                "local-outlier-factor",
                "1.0.0",
                "sklearn-pipeline-joblib-v1",
                plugin,
            )
            self.model_record(
                "model_lof_generic",
                "local-outlier-factor",
                "1.0.0",
                published,
                "sklearn-pipeline-joblib-v1",
                plugin=plugin,
                model_metadata={"n_features": 2},
            )
            pipeline = joblib.load(source / "model.joblib")
            self.assertTrue(pipeline.named_steps["detector"].novelty)
            client = self.app().test_client()
            self.assertEqual(client.post("/models/model_lof_generic/activate").status_code, 200)
            response = client.post(
                "/detect",
                data=self.multipart(
                    "model_lof_generic", np.array([[0.0, 0.0], [8.0, 8.0]])
                ),
            )
            self.assertEqual(response.status_code, 200, response.get_json())
            self.assertEqual(response.get_json()["detector_id"], "local-outlier-factor")
        finally:
            environment.stop()

    def create_cgnn_model(self, model_id, threshold):
        import torch

        from learning_adaptation.cgnn.mtad_gat import MTAD_GAT

        source = self.root / f"cgnn-{model_id}"
        source.mkdir()
        config = {
            "lookback": 2,
            "kernel_size": 3,
            "use_gatv2": True,
            "feat_gat_embed_dim": None,
            "time_gat_embed_dim": None,
            "gru_n_layers": 1,
            "gru_hid_dim": 4,
            "fc_n_layers": 1,
            "fc_hid_dim": 4,
            "dropout": 0.0,
            "alpha": 0.2,
            "use_cuda": False,
        }
        model = MTAD_GAT(2, 2, 2, kernel_size=3, gru_hid_dim=4, forecast_n_layers=1, forecast_hid_dim=4, dropout=0.0)
        torch.save(model.state_dict(), source / "model.pt")
        (source / "model_config.json").write_text(json.dumps(config), encoding="utf-8")
        (source / "model_evaluation.json").write_text(
            json.dumps({"epsilon_result": {"threshold": threshold}}), encoding="utf-8"
        )
        (source / "model_params.json").write_text("{}", encoding="utf-8")
        joblib.dump(MinMaxScaler().fit(np.array([[0.0, 0.0], [1.0, 1.0]])), source / "scaler.joblib")
        plugin = registry.get_plugin_reference("cgnn")
        published = self.publish(
            model_id,
            source,
            ("model.pt", "model_config.json", "model_evaluation.json", "model_params.json", "scaler.joblib"),
            "cgnn",
            "1.0.0",
            "cgnn-state-dict-v1",
            plugin,
        )
        record = self.model_record(
            model_id, "cgnn", "1.0.0", published, "cgnn-state-dict-v1", plugin=plugin, model_metadata={"n_features": 2}
        )
        return record, published

    def test_small_real_cgnn_state_dict_lookback_and_model_isolation(self):
        record_one, published_one = self.create_cgnn_model("model_cgnn_one", 0.1)
        record_two, published_two = self.create_cgnn_model("model_cgnn_two", 10.0)
        adapter = registry.get_inference_adapter("cgnn")
        loaded = []
        for record, published in ((record_one, published_one), (record_two, published_two)):
            loaded.append(
                adapter.load_model(
                    ModelLoadContext(
                        model_id=record["model_id"],
                        detector_id="cgnn",
                        detector_version="1.0.0",
                        artifact_directory=published["directory"] / "files",
                        artifact_manifest=published["manifest"],
                        model_record=record,
                        feature_identity=record["feature_identity"],
                    )
                )
            )
        self.assertEqual(loaded[0]["threshold"], 0.1)
        self.assertEqual(loaded[1]["threshold"], 10.0)
        client = self.app().test_client()
        self.assertEqual(client.post("/models/model_cgnn_one/activate").status_code, 200)
        short = client.post(
            "/detect", data=self.multipart("model_cgnn_one", np.array([[0.0, 0.0], [0.2, 0.2]]))
        )
        self.assertEqual(short.status_code, 422)
        response = client.post(
            "/detect",
            data=self.multipart("model_cgnn_one", np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [2.0, 2.0]])),
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["warmup_observations"], 2)
        self.assertEqual(response.get_json()["prediction_count"], 2)

    def test_no_detector_specific_branch_task_or_deployment_in_generic_runtime(self):
        generic_source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "anomaly_detection" / "generic").glob("*.py")
        )
        for detector_id in ("cgnn", "isolation-forest", "local-outlier-factor"):
            self.assertNotIn(f'== "{detector_id}"', generic_source)
            self.assertFalse(any(detector_id in path.name for path in (ROOT / "k8s").glob("*lof*")))
        self.assertNotIn("train_local_outlier_factor", (ROOT / "learning_adaptation" / "tasks.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
