import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from flask import Flask

from learning_adaptation.model_catalogue import ModelCatalogueStore
from learning_adaptation.training_lifecycle import (
    mark_cgnn_promoted,
    normalize_run_record,
    reconcile_run,
    record_successful_model,
    update_child,
)
from learning_adaptation.training_run_storage import TrainingRunStore
from learning_adaptation.training_runs_api import create_training_runs_blueprint


class ModelsBackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_root = self.root / "runs"
        self.model_root = self.root / "models"
        self.snapshot_root = self.root / "snapshots"
        self.dataset_root = self.root / "datasets"
        self.run_store = TrainingRunStore(self.run_root)
        self.model_store = ModelCatalogueStore(self.model_root)
        self.statuses = {}
        self.app = Flask(__name__)
        self.app.config.update(
            TESTING=True,
            EXISTING_DATASETS_ROOT=str(self.dataset_root),
            DATASET_SNAPSHOT_STORAGE_ROOT=str(self.snapshot_root),
            TRAINING_RUN_STORAGE_ROOT=str(self.run_root),
            MODEL_CATALOGUE_STORAGE_ROOT=str(self.model_root),
        )
        self.app.register_blueprint(
            create_training_runs_blueprint(
                Mock(), status_reader=lambda task_id: self.statuses[task_id]
            )
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.temporary.cleanup()

    def run_record(
        self,
        run_id,
        created_at,
        *,
        detector_id="cgnn",
        dataset_id="dataset-one",
        source="catalogue",
        status="queued",
        task_id=None,
    ):
        return {
            "schema_version": 2,
            "run_id": run_id,
            "status": status,
            "created_at": created_at,
            "dataset": {
                "source": source,
                "dataset_id": dataset_id,
                "version": 1,
                "partition_id": "partition-one",
            },
            "snapshot_id": f"snapshot-{run_id}",
            "snapshot_sha256": "a" * 64,
            "children": [
                {
                    "detector_id": detector_id,
                    "model_id": f"model-{run_id}",
                    "task_id": task_id,
                    "status": status,
                    "parameters": {},
                    "snapshot_id": f"snapshot-{run_id}",
                    "validation_error": None,
                    "dispatch_error": None,
                }
            ],
        }

    def model_record(self, model_id="model-one", detector_id="cgnn"):
        return {
            "schema_version": 1,
            "model_id": model_id,
            "detector_id": detector_id,
            "detector_version": "1.0.0",
            "status": "available",
            "created_at": "2026-09-29T10:00:00+00:00",
            "updated_at": "2026-09-29T10:00:00+00:00",
            "training_run_id": "run-one",
            "training_child_id": "run-one-cgnn",
            "dataset": {
                "source": "catalogue",
                "dataset_id": "dataset-one",
                "version": 3,
                "partition_id": "partition-one",
                "partition_checksum": "b" * 64,
            },
            "snapshot": {"snapshot_id": "snapshot-one", "snapshot_sha256": "c" * 64},
            "feature_identity": {
                "feature_order": ["cpu", "memory"],
                "feature_order_sha256": "d" * 64,
            },
            "training_parameters": {"epochs": 1},
            "evaluation": {"f1": 0.8},
            "promotion": {
                "status": "not_promoted",
                "promoted_at": None,
                "safe_reference": None,
            },
        }

    def test_empty_history_and_model_catalogue(self):
        self.assertEqual(self.client.get("/training_runs").get_json()["items"], [])
        self.assertEqual(self.client.get("/models").get_json()["items"], [])

    def test_history_pagination_sort_filters_corruption_and_safe_summary(self):
        first = self.run_record("run-one", "2026-09-28T10:00:00+00:00")
        first["children"][0]["status"] = "validation_failed"
        first["children"][0]["validation_error"] = (
            "failed at /private/models/secret.pt via redis://secret.internal:6379/2"
        )
        second = self.run_record(
            "run-two",
            "2026-09-29T10:00:00+00:00",
            detector_id="isolation-forest",
            dataset_id="dataset-two",
            source="existing",
        )
        self.run_store.write(first)
        self.run_store.write(second)
        corrupt = self.run_root / "run-corrupt"
        corrupt.mkdir(parents=True)
        (corrupt / "run.json").write_text("{", encoding="utf-8")

        body = self.client.get("/training_runs?page_size=1").get_json()
        self.assertEqual(body["items"][0]["run_id"], "run-two")
        self.assertEqual(body["total"], 2)
        self.assertEqual(body["pages"], 2)
        self.assertEqual(body["skipped_corrupt_records"], 1)
        self.assertNotIn("task_id", json.dumps(body))

        filtered = self.client.get(
            "/training_runs?detector_id=cgnn&dataset_id=dataset-one&source=catalogue&status=failed&sort=oldest"
        ).get_json()
        self.assertEqual([item["run_id"] for item in filtered["items"]], ["run-one"])
        encoded = json.dumps(filtered)
        self.assertNotIn("/private/models", encoded)
        self.assertNotIn("redis://", encoded)
        self.assertIn("redacted", encoded)

    def test_old_run_is_normalized_and_invalid_pagination_is_rejected(self):
        self.run_store.write(self.run_record("run-old", "2026-09-27T10:00:00+00:00"))
        body = self.client.get("/training_runs").get_json()["items"][0]
        self.assertIn("updated_at", body)
        self.assertIn("started_at", body["children"][0])
        self.assertEqual(self.client.get("/training_runs?page_size=101").status_code, 400)

    def test_reconciliation_persists_running_completed_and_failed(self):
        running = self.run_record(
            "run-progress", "2026-09-29T10:00:00+00:00", task_id="task-one"
        )
        self.run_store.write(running)
        result = reconcile_run(
            self.run_store,
            "run-progress",
            lambda _task: types.SimpleNamespace(
                state="TRAINING", info={"values": [1, 10, 3, 8]}
            ),
        )
        self.assertEqual(result["status"], "running")
        self.assertIsNotNone(result["children"][0]["started_at"])
        self.assertEqual(result["children"][0]["progress_phase"], "TRAINING")
        self.assertEqual(
            result["children"][0]["detail"], {"values": [1, 10, 3, 8]}
        )

        result = reconcile_run(
            self.run_store,
            "run-progress",
            lambda _task: types.SimpleNamespace(
                state="SUCCESS",
                info={
                    "status": "done",
                    "model_id": "model-run-progress",
                    "artifact_dir": "/private/worker/model",
                    "evaluation": {"f1": 0.8, "binary_predictions": [0, 1]},
                },
            ),
        )
        self.assertEqual(result["status"], "completed")
        self.assertIsNotNone(result["completed_at"])
        result_metadata = result["children"][0]["result_metadata"]
        self.assertNotIn("artifact_dir", result_metadata)
        self.assertNotIn("binary_predictions", result_metadata["evaluation"])
        self.assertEqual(
            result["children"][0]["stored_evaluation"]["metric_sets"][0]["metrics"]["f1_score"],
            0.8,
        )
        self.assertEqual(
            result["children"][0]["stored_evaluation"]["context"]["partition_id"],
            "partition-one",
        )

        # Redis expiry must not downgrade a stored terminal result.
        result = reconcile_run(
            self.run_store,
            "run-progress",
            lambda _task: types.SimpleNamespace(state="PENDING", info=None),
        )
        self.assertEqual(result["status"], "completed")

        failed = self.run_record(
            "run-failed", "2026-09-29T11:00:00+00:00", task_id="task-two"
        )
        self.run_store.write(failed)
        result = reconcile_run(
            self.run_store,
            "run-failed",
            lambda _task: types.SimpleNamespace(
                state="FAILURE", info=RuntimeError("failed at /tmp/model.pt")
            ),
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("redacted", result["children"][0]["failure_summary"])

    def test_parent_status_for_validation_failure_and_sibling(self):
        run = self.run_record("run-partial", "2026-09-29T10:00:00+00:00")
        run["children"] = [
            {**run["children"][0], "detector_id": "cgnn", "status": "validation_failed"},
            {
                **run["children"][0],
                "detector_id": "isolation-forest",
                "model_id": "model-if",
                "status": "running",
            },
        ]
        normalized = normalize_run_record(run)
        self.assertEqual(normalized["status"], "partial_success")
        normalized["children"][1]["status"] = "completed"
        self.assertEqual(normalize_run_record(normalized)["status"], "partial_success")

    def test_model_list_detail_filters_and_unknown_model(self):
        first = self.model_record()
        first["artifact_path"] = "/private/worker/model.pt"
        first["evaluation"]["details"] = "loaded from /private/worker/model.pt"
        self.model_store.write(first)
        other = self.model_record("model-two", "isolation-forest")
        other["created_at"] = other["updated_at"] = "2026-09-30T10:00:00+00:00"
        other["training_run_id"] = "run-two"
        other["dataset"]["dataset_id"] = "dataset-two"
        other["promotion"] = {
            "status": "promoted",
            "promoted_at": "2026-09-30T10:00:00+00:00",
            "safe_reference": "model-two",
        }
        self.model_store.write(other)
        (self.model_root / "broken.json").write_text("not-json", encoding="utf-8")

        body = self.client.get("/models?page_size=1").get_json()
        self.assertEqual(body["items"][0]["model_id"], "model-two")
        self.assertEqual(
            body["items"][0]["stored_evaluation"]["metric_sets"][0]["metrics"]["f1_score"],
            0.8,
        )
        self.assertEqual(
            body["items"][0]["stored_evaluation"]["context"]["partition_id"],
            "partition-one",
        )
        self.assertEqual(body["skipped_corrupt_records"], 1)
        filtered = self.client.get(
            "/models?detector_id=cgnn&dataset_id=dataset-one&training_run_id=run-one"
        ).get_json()
        self.assertEqual([item["model_id"] for item in filtered["items"]], ["model-one"])
        detail = self.client.get("/models/model-one")
        self.assertEqual(detail.status_code, 200)
        encoded = json.dumps(detail.get_json())
        for forbidden in (
            "artifact_dir",
            "artifact_path",
            "service_url",
            "task_id",
            "/private/worker",
            "/tmp/",
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertIn("redacted", encoded)
        self.assertTrue(detail.get_json()["stored_evaluation"]["available"])
        self.assertEqual(self.client.get("/models/missing").status_code, 404)

    def test_stored_evaluation_preserves_named_methods_without_ranking(self):
        record = self.model_record("model-cgnn-methods")
        record["evaluation"] = {
            "epsilon_result": {"precision": 0.7, "recall": 0.6, "f1": 0.64},
            "pot_result": {"precision": 0.8, "recall": 0.5, "f1": 0.61},
        }
        self.model_store.write(record)
        projection = self.client.get("/models/model-cgnn-methods").get_json()[
            "stored_evaluation"
        ]
        self.assertEqual(
            [item["method"] for item in projection["metric_sets"]],
            ["epsilon result", "pot result"],
        )
        self.assertNotIn("winner", json.dumps(projection).lower())

        unavailable = self.model_record("model-without-evaluation")
        unavailable["evaluation"] = {}
        self.model_store.write(unavailable)
        missing = self.client.get("/models/model-without-evaluation").get_json()[
            "stored_evaluation"
        ]
        self.assertFalse(missing["available"])
        self.assertEqual(missing["metric_sets"], [])
        self.assertEqual(missing["context"]["partition_id"], "partition-one")

    def test_successful_cgnn_and_if_lifecycle_create_model_records(self):
        with patch.dict(
            os.environ,
            {
                "TRAINED_MODELS_TEMP_ROOT": str(self.root / "trained_models_temp"),
                "TRAINING_RUN_STORAGE_ROOT": str(self.run_root),
                "MODEL_CATALOGUE_STORAGE_ROOT": str(self.model_root),
            },
        ):
            for detector_id, promotion in (
                ("cgnn", {"status": "not_promoted"}),
                (
                    "isolation-forest",
                    {
                        "status": "available",
                        "detector_id": "isolation-forest",
                        "model_id": "model-isolation-forest",
                    },
                ),
            ):
                run_id = f"run-{detector_id}"
                model_id = f"model-{detector_id}"
                self.run_store.write(
                    self.run_record(
                        run_id,
                        "2026-09-29T10:00:00+00:00",
                        detector_id=detector_id,
                    )
                )
                stored = self.run_store.read(run_id)
                stored["children"][0]["model_id"] = model_id
                self.run_store.write(stored)
                context = {
                    "training_run_id": run_id,
                    "training_child_id": f"{run_id}-{detector_id}",
                    "model_id": model_id,
                    "detector_id": detector_id,
                    "detector_version": "1.0.0",
                    "dataset": stored["dataset"],
                    "snapshot": {
                        "snapshot_id": stored["snapshot_id"],
                        "snapshot_sha256": stored["snapshot_sha256"],
                    },
                    "feature_identity": {
                        "feature_order": ["cpu"],
                        "feature_order_sha256": "d" * 64,
                    },
                    "training_parameters": {},
                }
                record_successful_model(
                    context,
                    evaluation={"f1": 1.0, "binary_predictions": [0, 1]},
                    promotion=promotion,
                )
                record = self.model_store.read(model_id)
                self.assertEqual(record["detector_id"], detector_id)
                self.assertNotIn("binary_predictions", record["evaluation"])
                expected_promotion = (
                    "promoted" if detector_id == "isolation-forest" else "not_promoted"
                )
                self.assertEqual(record["promotion"]["status"], expected_promotion)
                child = self.run_store.read(run_id)["children"][0]
                self.assertEqual(child["status"], "completed")
                self.assertEqual(child["model_status"], "available")

    def test_model_record_retry_reuses_stable_record(self):
        self.run_store.write(self.run_record("run-retry", "2026-09-29T10:00:00+00:00"))
        context = {
            "training_run_id": "run-retry",
            "training_child_id": "run-retry-cgnn",
            "model_id": "model-run-retry",
            "detector_id": "cgnn",
            "detector_version": "1.0.0",
            "dataset": self.run_store.read("run-retry")["dataset"],
            "snapshot": {"snapshot_id": "snapshot-retry", "snapshot_sha256": "a" * 64},
            "feature_identity": {
                "feature_order": ["cpu"],
                "feature_order_sha256": "b" * 64,
            },
            "training_parameters": {},
        }
        with patch.dict(
            os.environ,
            {
                "TRAINING_RUN_STORAGE_ROOT": str(self.run_root),
                "MODEL_CATALOGUE_STORAGE_ROOT": str(self.model_root),
            },
        ):
            first = record_successful_model(
                context, evaluation={"f1": 0.8}, promotion={"status": "not_promoted"}
            )
            second = record_successful_model(
                context, evaluation={"f1": 0.9}, promotion={"status": "not_promoted"}
            )
        self.assertEqual(first, second)
        self.assertEqual(self.model_store.read("model-run-retry")["evaluation"]["f1"], 0.8)

        with patch.dict(
            os.environ,
            {
                "TRAINING_RUN_STORAGE_ROOT": str(self.run_root),
                "MODEL_CATALOGUE_STORAGE_ROOT": str(self.model_root),
            },
        ):
            mark_cgnn_promoted("model-run-retry")
        promoted = self.model_store.read("model-run-retry")
        self.assertEqual(promoted["promotion"]["status"], "promoted")
        child = self.run_store.read("run-retry")["children"][0]
        self.assertEqual(child["result_metadata"]["promotion"]["status"], "promoted")

    def test_child_updates_preserve_sibling_state(self):
        run = self.run_record("run-siblings", "2026-09-29T10:00:00+00:00")
        run["children"].append(
            {
                **run["children"][0],
                "detector_id": "isolation-forest",
                "model_id": "model-if",
            }
        )
        self.run_store.write(run)

        update_child(
            self.run_store,
            "run-siblings",
            "cgnn",
            lambda child, _now: child.update(status="completed"),
        )
        updated = update_child(
            self.run_store,
            "run-siblings",
            "isolation-forest",
            lambda child, _now: child.update(status="failed"),
        )
        states = {child["detector_id"]: child["status"] for child in updated["children"]}
        self.assertEqual(states, {"cgnn": "completed", "isolation-forest": "failed"})
        self.assertEqual(updated["status"], "partial_success")
        self.assertEqual(updated["child_summary"]["completed"], 1)
        self.assertEqual(updated["child_summary"]["failed"], 1)

    def test_failed_child_does_not_create_model_record(self):
        self.run_store.write(
            self.run_record(
                "run-no-model",
                "2026-09-29T10:00:00+00:00",
                task_id="task-failed",
            )
        )
        reconcile_run(
            self.run_store,
            "run-no-model",
            lambda _task: types.SimpleNamespace(state="FAILURE", info="failed"),
        )
        self.assertEqual(self.model_store.records()[0], [])


if __name__ == "__main__":
    unittest.main()
