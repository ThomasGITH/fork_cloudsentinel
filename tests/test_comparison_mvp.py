import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from flask import Flask

from learning_adaptation.comparison_api import (
    _model_compatibility,
    create_comparisons_blueprint,
    public_comparison,
)
from learning_adaptation.comparison_execution import (
    detection_timing,
    execute_comparison,
    labelled_metrics,
)
from learning_adaptation.comparison_evaluation import (
    ComparisonEvaluationError,
    GenericEvaluationClient,
)
from learning_adaptation.comparison_storage import ComparisonStore
from learning_adaptation.model_catalogue import ModelCatalogueStore


FEATURE_HASH = "a" * 64
ARTIFACT_HASH = "b" * 64


def model_record(model_id="model-one", *, ready=True, feature_hash=FEATURE_HASH):
    return {
        "schema_version": 1,
        "model_id": model_id,
        "detector_id": "test-detector",
        "detector_version": "1.0.0",
        "status": "available",
        "created_at": "2026-10-01T08:00:00+00:00",
        "updated_at": "2026-10-01T08:00:00+00:00",
        "training_run_id": "run-source",
        "dataset": {"source": "catalogue", "dataset_id": "ds-train", "version": 1, "partition_id": "part-train"},
        "snapshot": {"snapshot_id": "snapshot-one", "snapshot_sha256": "c" * 64},
        "feature_identity": {"feature_order": ["cpu", "memory"], "feature_order_sha256": feature_hash},
        "training_parameters": {},
        "evaluation": {},
        "promotion": {"status": "not_applicable", "promoted_at": None, "safe_reference": None},
        "artifact": {"format": "dummy"},
        "model_metadata": {"modality": "metrics", "display_name": model_id},
        "inference": ({
            "status": "ready",
            "artifact_id": f"artifact-{model_id}",
            "artifact_manifest_sha256": ARTIFACT_HASH,
            "contract": "cloudsentinel.inference/v1",
            "artifact_format": "dummy-v1",
        } if ready else {"status": "legacy_external"}),
    }


def dataset_context():
    return {
        "dataset_id": "ds-eval",
        "display_name": "CPU incident",
        "version": 2,
        "partition_id": "part-eval",
        "partition_checksum": "d" * 64,
        "status": "available",
        "modality": "metrics",
        "feature_order": ["cpu", "memory"],
        "feature_order_sha256": FEATURE_HASH,
        "observation_count": 3,
        "start_time": "2026-10-01T10:00:00Z",
        "end_time": "2026-10-01T10:02:00Z",
        "workload_context": {"workload_intensity": "High", "dominant_workload_characteristic": "CPU-intensive", "anomaly_scenario": "CPU stress"},
        "ground_truth_available": True,
        "known_incident_windows": [{"incident_id": "incident-one", "start_time": "2026-10-01T10:01:00Z", "end_time": "2026-10-01T10:02:00Z", "scenario": "CPU stress"}],
    }


def comparison_record(models=None):
    models = models or [model_record("model-one"), model_record("model-two")]
    selected = []
    for model in models:
        selected.append({
            "model_id": model["model_id"], "display_name": model["model_id"],
            "detector_id": model["detector_id"], "detector_version": model["detector_version"],
            "model_created_at": model["created_at"], "training_run_id": model["training_run_id"],
            "training_dataset": model["dataset"], "feature_identity": model["feature_identity"],
            "artifact_identity": {
                "artifact_id": model["inference"]["artifact_id"],
                "artifact_manifest_sha256": model["inference"]["artifact_manifest_sha256"],
                "contract": model["inference"]["contract"],
                "artifact_format": model["inference"]["artifact_format"],
            }, "plugin": {},
        })
    dataset = dataset_context()
    return {
        "schema_version": 1, "comparison_id": "comparison-one", "name": "CPU comparison",
        "analysis_type": "anomaly_detection", "configuration_version": "cloudsentinel.comparison/v1",
        "status": "queued", "created_at": "2026-10-01T10:00:00+00:00", "started_at": None,
        "updated_at": "2026-10-01T10:00:00+00:00", "completed_at": None,
        "client_request_id": "client-one", "evaluation_dataset": dataset,
        "modality": "metrics", "feature_identity": {"feature_order": ["cpu", "memory"], "sha256": FEATURE_HASH},
        "ground_truth_available": True, "known_incident_window": dataset["known_incident_windows"][0],
        "known_incident_windows": dataset["known_incident_windows"],
        "selected_models": selected, "results": [], "result_reference": None,
        "safe_failure_summary": None, "deterministic_summary": None,
    }


class FakeBundleClient:
    def __init__(self, root, *, labels=None, timestamps=None):
        self.root = Path(root)
        self.matrix = self.root / "matrix.csv"
        self.matrix.write_text("0,0\n1,1\n2,2\n", encoding="utf-8")
        self.labels = labels
        self.timestamps = timestamps

    def download(self, _dataset):
        return {
            "temporary_root": self.root,
            "matrix_path": self.matrix,
            "labels": self.labels,
            "timestamps": self.timestamps,
            "manifest": {
                "feature_order": ["cpu", "memory"], "feature_order_sha256": FEATURE_HASH,
                "ground_truth_available": self.labels is not None,
                "counts": {"observations": 3, "features": 2},
                "known_incident_windows": dataset_context()["known_incident_windows"],
            },
        }


class FakeRuntimeClient:
    def __init__(self, failures=None):
        self.failures = set(failures or [])
        self.model_ids = []

    def evaluate(self, model_id, *_args):
        self.model_ids.append(model_id)
        if model_id in self.failures:
            raise RuntimeError("private failure at /app/secret")
        return {
            "binary_predictions": [0, 1, 1], "anomaly_scores": [0.1, 0.8, 0.9],
            "warmup_observations": 0, "runtime_ms": 12.5, "input_observation_count": 3,
            "artifact_manifest_sha256": ARTIFACT_HASH,
        }


class ComparisonMetricTests(unittest.TestCase):
    def test_labelled_metrics_and_lead_time_semantics(self):
        metrics = labelled_metrics([0, 1, 0, 1], [0, 1, 1, 0])
        self.assertEqual(metrics["confusion_matrix"], {"tp": 1, "fp": 1, "tn": 1, "fn": 1})
        self.assertEqual(metrics["precision"], 0.5)
        before = detection_timing([1, 0], ["2026-10-01T10:00:00Z", "2026-10-01T10:01:00Z"], {"start_time": "2026-10-01T10:01:00Z"})
        self.assertEqual(before["lead_time_seconds"], 60)
        at_start = detection_timing(
            [1],
            ["2026-10-01T10:01:00Z"],
            {"start_time": "2026-10-01T10:01:00Z"},
        )
        self.assertEqual(at_start["lead_time_seconds"], 0)
        self.assertEqual(at_start["lead_time_status"], "before_or_at_incident_start")
        after = detection_timing([0, 1], ["2026-10-01T10:00:00Z", "2026-10-01T10:02:00Z"], {"start_time": "2026-10-01T10:01:00Z"})
        self.assertEqual(after["lead_time_status"], "after_incident_start")
        self.assertIsNone(after["lead_time_seconds"])
        unavailable = detection_timing([1], None, None)
        self.assertEqual(unavailable["lead_time_status"], "unavailable")

    def test_label_free_execution_omits_accuracy_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ComparisonStore(Path(temp) / "comparisons")
            record = comparison_record(); record["ground_truth_available"] = False
            store.create(record)
            bundle_root = Path(temp) / "bundle"; bundle_root.mkdir()
            result = execute_comparison(
                "comparison-one", storage_root=str(store.root), catalogue_url="http://catalogue",
                generic_runtime_url="http://generic", bundle_client=FakeBundleClient(bundle_root),
                runtime_client=FakeRuntimeClient(),
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["known_incident_windows"], dataset_context()["known_incident_windows"])
            self.assertFalse(result["results"][0]["ground_truth_available"])
            self.assertEqual(result["results"][0]["metrics"], {})
            self.assertEqual(
                result["results"][0]["lead_time_status"],
                "ground_truth_unavailable",
            )
            self.assertIsNone(result["results"][0]["lead_time_seconds"])

    def test_partial_and_failed_lifecycle_are_model_isolated(self):
        for failures, expected in [({"model-two"}, "partial"), ({"model-one", "model-two"}, "failed")]:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temp:
                store = ComparisonStore(Path(temp) / "comparisons"); store.create(comparison_record())
                bundle_root = Path(temp) / "bundle"; bundle_root.mkdir()
                result = execute_comparison(
                    "comparison-one", storage_root=str(store.root), catalogue_url="http://catalogue",
                    generic_runtime_url="http://generic",
                    bundle_client=FakeBundleClient(bundle_root, labels=[0, 1, 1], timestamps=["2026-10-01T10:00:00Z", "2026-10-01T10:01:00Z", "2026-10-01T10:02:00Z"]),
                    runtime_client=FakeRuntimeClient(failures),
                )
                self.assertEqual(result["status"], expected)
                self.assertNotIn("/app/secret", json.dumps(public_comparison(result)))

    def test_executor_contains_no_detector_specific_branch(self):
        source = Path("learning_adaptation/comparison_execution.py").read_text(encoding="utf-8")
        for detector_id in ("cgnn", "isolation-forest", "local-outlier-factor"):
            self.assertNotIn(detector_id, source)
        self.assertIn("GenericEvaluationClient", source)

    def test_generic_evaluation_response_is_bounded(self):
        class Response:
            status_code = 200
            headers = {}

            def __init__(self, payload):
                self.payload = payload

            def json(self):
                return self.payload

        responses = iter(
            [
                Response({"status": "ready"}),
                Response({"binary_predictions": [0] * 100, "anomaly_scores": [0.0] * 100}),
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            matrix = Path(temp) / "matrix.csv"
            matrix.write_text("1\n", encoding="utf-8")
            client = GenericEvaluationClient(
                "http://generic",
                maximum_response_bytes=100,
                request_post=lambda *_args, **_kwargs: next(responses),
            )
            with self.assertRaisesRegex(ComparisonEvaluationError, "response exceeds"):
                client.evaluate("model-one", matrix, ["cpu"], FEATURE_HASH, None)


class FakeTask:
    def __init__(self): self.calls = []
    def apply_async(self, args, task_id): self.calls.append((args, task_id)); return Mock(id=task_id)


class FakeResponse:
    def __init__(self, payload, status=200): self.payload = payload; self.status_code = status
    def json(self): return self.payload


class ComparisonApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.models = ModelCatalogueStore(root / "models")
        self.models.write(model_record("model-one")); self.models.write(model_record("model-two"))
        self.models.write(model_record("model-legacy", ready=False))
        self.models.write(model_record("model-mismatch", feature_hash="f" * 64))
        self.task = FakeTask()
        self.app = Flask(__name__)
        self.app.config.update(
            TESTING=True, COMPARISON_STORAGE_ROOT=str(root / "comparisons"),
            MODEL_CATALOGUE_STORAGE_ROOT=str(root / "models"),
            API_DATA_CATALOGUE_URL="http://catalogue.internal",
        )
        self.dataset = dataset_context()
        version = {
            "status": "available", "version": 2,
            "partition": {"mode": "time_range", "partition_id": "part-eval", "checksum": "d" * 64, "test": {"start_time": self.dataset["start_time"], "end_time": self.dataset["end_time"], "observation_count": 3}, "feature_order_reference": {"sha256": FEATURE_HASH}},
            "technical_schema": {"modality": "metrics", "feature_order": ["cpu", "memory"]},
            "workload_context": self.dataset["workload_context"], "ground_truth": {"labels_available": True},
            "incident_context": self.dataset["known_incident_windows"], "source": {},
        }
        detail = {"dataset_id": "ds-eval", "display_name": "CPU incident", "latest_version": 2, "version": version, "workload_context": self.dataset["workload_context"]}
        listing = {"items": [{"dataset_id": "ds-eval", "version": 2}], "total": 1}
        def get(_url, params=None, timeout=None):
            if _url.endswith("/datasets"): return FakeResponse(listing)
            if _url.endswith("/versions/2"): return FakeResponse(version)
            return FakeResponse(detail)
        self.app.config["COMPARISON_CATALOGUE_GET"] = get
        self.app.register_blueprint(create_comparisons_blueprint(self.task))
        self.client = self.app.test_client()

    def test_ready_compatibility_and_incompatible_reasons(self):
        response = self.client.get("/api/comparisons/compatible-models?dataset_id=ds-eval&version=2&partition_id=part-eval")
        self.assertEqual(response.status_code, 200)
        items = {item["model_id"]: item for item in response.get_json()["items"]}
        self.assertTrue(items["model-one"]["compatible"])
        self.assertIn("not inference-ready", items["model-legacy"]["compatibility_reasons"][0])
        self.assertIn("Feature identity", items["model-mismatch"]["compatibility_reasons"][-1])
        logs_model = model_record("model-logs")
        logs_model["model_metadata"]["modality"] = "logs"
        compatible, reasons = _model_compatibility(logs_model, dataset_context())
        self.assertFalse(compatible)
        self.assertIn("Requires Logs data", reasons)

    def test_minimum_models_snapshot_and_duplicate_submit(self):
        payload = {"name": "CPU comparison", "evaluation_dataset": {"dataset_id": "ds-eval", "version": 2, "partition_id": "part-eval"}, "model_ids": ["model-one"], "client_request_id": "request-one"}
        self.assertEqual(self.client.post("/api/comparisons", json=payload).status_code, 400)
        payload["model_ids"] = ["model-one", "model-two"]
        first = self.client.post("/api/comparisons", json=payload)
        self.assertEqual(first.status_code, 202)
        second = self.client.post("/api/comparisons", json=payload)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.get_json()["comparison_id"], second.get_json()["comparison_id"])
        self.assertEqual(len(self.task.calls), 1)
        stored = ComparisonStore(Path(self.temp.name) / "comparisons").read(first.get_json()["comparison_id"])
        self.assertEqual(stored["selected_models"][0]["artifact_identity"]["artifact_manifest_sha256"], ARTIFACT_HASH)
        self.assertEqual(stored["known_incident_windows"], self.dataset["known_incident_windows"])

    def test_filters_pagination_and_safe_serialisation(self):
        payload = {"name": "CPU comparison", "evaluation_dataset": {"dataset_id": "ds-eval", "version": 2, "partition_id": "part-eval"}, "model_ids": ["model-one", "model-two"], "client_request_id": "request-filter"}
        created = self.client.post("/api/comparisons", json=payload).get_json()
        listing = self.client.get("/api/comparisons?search=CPU&modality=metrics&workload=High&status=queued&page=1&page_size=10")
        self.assertEqual(listing.status_code, 200); self.assertEqual(listing.get_json()["total"], 1)
        detail = self.client.get(f"/api/comparisons/{created['comparison_id']}")
        text = detail.get_data(as_text=True)
        for forbidden in ("/app/", "http://catalogue.internal", "task_id", "entry_point"):
            self.assertNotIn(forbidden, text)

    def test_public_projection_hides_internal_fields(self):
        record = comparison_record(); record["task_id"] = "celery-secret"; record["private_path"] = "/app/private"
        text = json.dumps(public_comparison(record))
        self.assertNotIn("celery-secret", text); self.assertNotIn("/app/private", text)


class ComparisonTimelineAssetTests(unittest.TestCase):
    def test_chart_uses_chartjs_parsing_for_labelled_numeric_series(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "user_interface/monitoring_project/config_app/static/js/comparison_detail.js"
        ).read_text(encoding="utf-8")
        self.assertIn('new Chart(canvas', source)
        self.assertNotIn('parsing: false', source)
        self.assertIn('data-timeline-error', source)
        self.assertIn('incidentBands', source)
        self.assertIn('comparison-incident-windows', source)


if __name__ == "__main__":
    unittest.main()
