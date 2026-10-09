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
from learning_adaptation.comparison_robustness import aggregate_robustness
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
        "evaluation": {"precision": 0.8, "recall": 0.75, "f1": 0.774},
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


class ComparisonRobustnessTests(unittest.TestCase):
    @staticmethod
    def _terminal_record(
        comparison_id,
        *,
        models=None,
        checksum,
        completed_at,
        status="completed",
        labelled=True,
        f1_values=None,
        failed_models=None,
        workload="High",
        scenario="CPU stress",
    ):
        record = comparison_record(models=models)
        record.update(
            comparison_id=comparison_id,
            name=f"Comparison {comparison_id}",
            status=status,
            completed_at=completed_at,
            updated_at=completed_at,
            ground_truth_available=labelled,
        )
        record["evaluation_dataset"].update(
            dataset_id=f"ds-{checksum[-4:]}",
            display_name=f"Dataset {checksum[-4:]}",
            version=1,
            partition_id=f"part-{checksum[-4:]}",
            partition_checksum=checksum,
            ground_truth_available=labelled,
            workload_context={
                "workload_intensity": workload,
                "dominant_workload_characteristic": "CPU-intensive",
                "anomaly_scenario": scenario,
            },
        )
        f1_values = f1_values or {}
        failed_models = set(failed_models or [])
        results = []
        for selected in record["selected_models"]:
            model_id = selected["model_id"]
            failed = model_id in failed_models
            f1 = f1_values.get(model_id, 0.5)
            results.append(
                {
                    "model_id": model_id,
                    "display_name": selected["display_name"],
                    "detector_id": selected["detector_id"],
                    "detector_version": selected["detector_version"],
                    "status": "failed" if failed else "completed",
                    "ground_truth_available": labelled,
                    "metrics": (
                        {"precision": f1 + 0.05, "recall": f1 - 0.05, "f1_score": f1}
                        if labelled and not failed
                        else {}
                    ),
                    "runtime_ms": 10.0 + f1,
                    "lead_time_status": "before_or_at_incident_start" if labelled else "ground_truth_unavailable",
                    "lead_time_seconds": 60.0 if labelled else None,
                }
            )
        record["results"] = results
        return record

    def test_exact_artifact_dedup_metrics_shared_matrix_and_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ComparisonStore(Path(temp) / "comparisons")
            models = [model_record("model-one"), model_record("model-two")]
            current = self._terminal_record(
                "comparison-current", models=models, checksum="a" * 64,
                completed_at="2026-10-03T10:00:00Z",
                f1_values={"model-one": 0.6, "model-two": 0.7},
            )
            older_duplicate = self._terminal_record(
                "comparison-old", models=models, checksum="a" * 64,
                completed_at="2026-10-01T10:00:00Z",
                f1_values={"model-one": 0.1, "model-two": 0.2},
            )
            second = self._terminal_record(
                "comparison-second", models=[models[0]], checksum="b" * 64,
                completed_at="2026-10-02T10:00:00Z", status="partial",
                f1_values={"model-one": 0.8}, workload="Variable",
            )
            unlabelled = self._terminal_record(
                "comparison-unlabelled", models=[models[0]], checksum="c" * 64,
                completed_at="2026-10-02T11:00:00Z", labelled=False,
            )
            failed_run = self._terminal_record(
                "comparison-failed", models=[models[0]], checksum="d" * 64,
                completed_at="2026-10-02T12:00:00Z", status="failed",
                f1_values={"model-one": 0.99},
            )
            failed_child = self._terminal_record(
                "comparison-child-failed", models=[models[0]], checksum="e" * 64,
                completed_at="2026-10-02T13:00:00Z", status="partial",
                failed_models={"model-one"},
            )
            changed_artifact_model = model_record("model-one")
            changed_artifact_model["inference"]["artifact_id"] = "artifact-other"
            changed_artifact = self._terminal_record(
                "comparison-other-artifact", models=[changed_artifact_model],
                checksum="f" * 64, completed_at="2026-10-02T14:00:00Z",
                f1_values={"model-one": 1.0},
            )
            for record in (
                current, older_duplicate, second, unlabelled, failed_run,
                failed_child, changed_artifact,
            ):
                store.create(record)

            payload = aggregate_robustness(store, current)
            by_model = {model["model_id"]: model for model in payload["models"]}
            one = by_model["model-one"]
            self.assertEqual(one["context_count"], 3)
            self.assertEqual(one["labelled_context_count"], 2)
            self.assertEqual(one["statistics"]["f1_score"]["median"], 0.7)
            self.assertEqual(one["statistics"]["f1_score"]["minimum"], 0.6)
            self.assertEqual(one["statistics"]["f1_score"]["maximum"], 0.8)
            self.assertAlmostEqual(one["statistics"]["f1_score"]["range"], 0.2)
            self.assertAlmostEqual(one["statistics"]["f1_score"]["standard_deviation"], 0.1)
            self.assertEqual(one["statistics"]["precision"]["median"], 0.75)
            self.assertAlmostEqual(one["statistics"]["recall"]["median"], 0.65)
            self.assertEqual(one["statistics"]["runtime_ms"]["median"], 10.6)
            self.assertEqual(one["statistics"]["lead_time_seconds"]["median"], 60.0)
            self.assertNotIn("comparison-old", one["related_comparison_ids"])
            self.assertNotIn("comparison-other-artifact", one["related_comparison_ids"])
            self.assertEqual(by_model["model-two"]["context_count"], 1)
            self.assertEqual(len(payload["shared_context_matrix"]), 1)
            self.assertEqual(len(payload["shared_context_matrix"][0]["models"]), 2)
            self.assertTrue(any("coverage differs" in warning for warning in payload["coverage_warnings"]))
            self.assertTrue(any("latest completed" in warning for warning in payload["coverage_warnings"]))

            labelled = aggregate_robustness(store, current, filters={"labelled_only": True})
            self.assertEqual(
                {model["model_id"]: model["context_count"] for model in labelled["models"]},
                {"model-one": 2, "model-two": 1},
            )
            shared = aggregate_robustness(store, current, filters={"shared_only": True})
            self.assertEqual(
                {model["model_id"]: model["context_count"] for model in shared["models"]},
                {"model-one": 1, "model-two": 1},
            )
            variable = aggregate_robustness(
                store,
                current,
                filters={"workload": "Variable", "scenario": "CPU stress"},
            )
            self.assertEqual(
                {model["model_id"]: model["context_count"] for model in variable["models"]},
                {"model-one": 1, "model-two": 0},
            )

    def test_unlabelled_contexts_do_not_produce_accuracy_or_lead_time(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ComparisonStore(Path(temp) / "comparisons")
            model = model_record("model-one")
            current = self._terminal_record(
                "comparison-current", models=[model], checksum="a" * 64,
                completed_at="2026-10-03T10:00:00Z", labelled=False,
            )
            store.create(current)
            payload = aggregate_robustness(store, current)
            aggregate = payload["models"][0]
            self.assertEqual(aggregate["context_count"], 1)
            self.assertEqual(aggregate["labelled_context_count"], 0)
            self.assertIsNone(aggregate["statistics"]["f1_score"]["median"])
            self.assertIsNone(aggregate["statistics"]["precision"]["median"])
            self.assertIsNone(aggregate["statistics"]["recall"]["median"])
            self.assertEqual(aggregate["statistics"]["lead_time_seconds"]["count"], 0)
            self.assertFalse(aggregate["sufficient_contexts"])

    def test_robustness_has_no_detector_specific_branch(self):
        source = Path("learning_adaptation/comparison_robustness.py").read_text(
            encoding="utf-8"
        )
        for detector_id in ("cgnn", "isolation-forest", "local-outlier-factor"):
            self.assertNotIn(detector_id, source)

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
        self.assertEqual(
            items["model-one"]["stored_evaluation"]["context"]["partition_id"],
            "part-train",
        )
        self.assertEqual(
            items["model-one"]["stored_evaluation"]["metric_sets"][0]["metrics"]["f1_score"],
            0.774,
        )
        self.assertIn("not inference-ready", items["model-legacy"]["compatibility_reasons"][0])
        self.assertIn("Feature identity", items["model-mismatch"]["compatibility_reasons"][-1])
        self.assertEqual(
            [item["model_id"] for item in response.get_json()["items"][:2]],
            ["model-one", "model-two"],
        )
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

        selected = self.models.read("model-one")["inference"]
        related = self.client.get(
            "/api/comparisons",
            query_string={
                "model_id": "model-one",
                "artifact_id": selected["artifact_id"],
                "artifact_manifest_sha256": selected["artifact_manifest_sha256"],
            },
        )
        self.assertEqual(related.status_code, 200)
        self.assertEqual(related.get_json()["total"], 1)
        incomplete = self.client.get("/api/comparisons?model_id=model-one")
        self.assertEqual(incomplete.status_code, 400)

    def test_robustness_endpoint_is_safe_bounded_and_filterable(self):
        store = ComparisonStore(Path(self.temp.name) / "comparisons")
        models = [model_record("model-one"), model_record("model-two")]
        current = ComparisonRobustnessTests._terminal_record(
            "comparison-robust", models=models, checksum="a" * 64,
            completed_at="2026-10-03T10:00:00Z",
            f1_values={"model-one": 0.6, "model-two": 0.7},
        )
        current["private_path"] = "/app/private"
        current["internal_url"] = "http://secret.internal"
        current["selected_models"][0]["artifact_identity"]["path"] = "/app/artifact"
        current["results"][0]["binary_predictions"] = [0, 1]
        store.create(current)
        corrupt = store.root / "comparison-corrupt"
        corrupt.mkdir(parents=True)
        (corrupt / "comparison.json").write_text("{broken", encoding="utf-8")
        response = self.client.get(
            "/api/comparisons/comparison-robust/robustness?labelled_only=true&shared_only=true"
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["aggregation_version"], "cloudsentinel.comparison-robustness/v1")
        self.assertEqual(len(payload["shared_context_matrix"]), 1)
        self.assertEqual(payload["skipped_corrupt_records"], 1)
        text = response.get_data(as_text=True)
        for forbidden in (
            "/app/private", "/app/artifact", "http://secret.internal", "binary_predictions",
            "anomaly_scores", "entry_point", "task_id",
        ):
            self.assertNotIn(forbidden, text)
        invalid = self.client.get(
            "/api/comparisons/comparison-robust/robustness?labelled_only=perhaps"
        )
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(
            self.client.get("/api/comparisons/comparison-missing/robustness").status_code,
            404,
        )

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

    def test_robustness_template_is_grounded_and_contains_no_rca(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "user_interface/monitoring_project/config_app/templates/config_app/comparison/_robustness.html"
        ).read_text(encoding="utf-8")
        for expected in (
            "same saved model artefact",
            "Shared evaluation contexts",
            "Historical context results",
            "No completed comparisons found for this exact saved model artefact",
            "Metrics are unavailable because the available contexts have no ground truth",
            "View related comparisons",
        ):
            self.assertIn(expected, source)
        for forbidden in ("Root Cause Analysis", "View RCA", "robustness score"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
