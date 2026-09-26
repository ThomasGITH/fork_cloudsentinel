import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock
import zipfile

from flask import Flask
import numpy as np

from data_ingestion.catalogue.api import (
    configure_catalogue_defaults,
    create_catalogue_blueprint,
)
from data_ingestion.catalogue.models import version_summary
from data_ingestion.catalogue.repository import FileCatalogueRepository
from learning_adaptation.catalogue_training import (
    CatalogueBundleError,
    verify_bundle,
)
from learning_adaptation.training_runs_api import (
    configure_training_run_defaults,
    create_training_runs_blueprint,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CatalogueMaterializationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.catalogue_root = self.root / "catalogue"
        self.legacy_root = self.root / "legacy"
        self.legacy_root.mkdir()
        self.app = Flask(__name__)
        configure_catalogue_defaults(self.app)
        self.app.config.update(
            TESTING=True,
            CATALOGUE_STORAGE_ROOT=str(self.catalogue_root),
            LEGACY_DATASETS_ROOT=str(self.legacy_root),
        )
        self.app.register_blueprint(create_catalogue_blueprint())
        self.client = self.app.test_client()
        self.repository = FileCatalogueRepository(self.catalogue_root)

    def tearDown(self):
        self.temporary.cleanup()

    def _read_bundle(self, response):
        self.assertEqual(response.status_code, 200, response.get_json(silent=True))
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            return {
                name: archive.read(name)
                for name in ("manifest.json", "train.csv", "test.csv", "labels.csv")
            }

    def _legacy(self):
        directory = self.legacy_root / "legacy-one"
        directory.mkdir()
        np.savetxt(directory / "train_legacy-one", [[0, 1], [1, 2], [2, 3]], delimiter=",")
        np.savetxt(directory / "test_legacy-one", [[3, 4], [4, 5]], delimiter=",")
        np.savetxt(directory / "test_label_legacy-one", [0, 1], delimiter=",")
        (directory / "details.json").write_text(
            json.dumps(
                {
                    "dataset": "Legacy one",
                    "containers": ["service"],
                    "metrics": ["cpu", "memory"],
                    "step_size": 60,
                    "data_entries": 2,
                }
            ),
            encoding="utf-8",
        )
        detail = self.client.get("/datasets/legacy-one").get_json()
        return detail["version"]["partition"]

    def _draft_payload(self, dataset_id):
        return {
            "dataset_id": dataset_id,
            "display_name": dataset_id,
            "purpose": ["training", "evaluation"],
            "source": {
                "type": "prometheus",
                "prometheus_source_id": "cluster-default",
                "application": "shop",
                "namespace": "default",
                "requested_targets": {"services": [], "pods": [], "label_selectors": []},
                "start_time": "2026-09-25T10:00:00Z",
                "end_time": "2026-09-25T10:06:00Z",
                "sampling_interval_seconds": 60,
                "queries": [],
            },
            "workload_context": {
                "workload_intensity": "Normal",
                "dominant_workload_characteristic": "Mixed",
                "anomaly_scenario": "Unknown",
            },
            "partition": {"mode": "none"},
        }

    def _available(self, dataset_id):
        response = self.client.post("/datasets", json=self._draft_payload(dataset_id))
        self.assertEqual(response.status_code, 201, response.get_json())
        version = self.repository.read_version(dataset_id, 1)
        active = self.repository.artifact_directory(dataset_id, 1)
        canonical = active / "canonical"
        canonical.mkdir(parents=True)
        timestamps = [f"2026-09-25T10:0{index}:00Z" for index in range(7)]
        observations = canonical / "observations.csv"
        with observations.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(["timestamp", "cpu", "memory"])
            for index, timestamp in enumerate(timestamps):
                writer.writerow([timestamp, index, index + 10])
        timestamp_file = canonical / "timestamps.csv"
        with timestamp_file.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(["timestamp"])
            writer.writerows([[value] for value in timestamps])
        provenance = active / "provenance.json"
        provenance.write_text('{"schema_version":1}', encoding="utf-8")
        version["status"] = "available"
        version["technical_schema"] = {
            "modality": "metrics",
            "timestamp_column": "timestamp",
            "timestamp_timezone": "UTC",
            "observation_count": 7,
            "feature_count": 2,
            "feature_identity": "named",
            "feature_order": ["cpu", "memory"],
            "missing_values": {"total": 0, "by_feature": {"cpu": 0, "memory": 0}},
            "normalization": "none",
            "imputation": "none",
        }
        version["artifacts"] = {
            "canonical/observations.csv": {
                "path": str(observations.relative_to(self.catalogue_root.resolve())),
                "sha256": sha256(observations),
                "bytes": observations.stat().st_size,
            },
            "canonical/timestamps.csv": {
                "path": str(timestamp_file.relative_to(self.catalogue_root.resolve())),
                "sha256": sha256(timestamp_file),
                "bytes": timestamp_file.stat().st_size,
            },
            "provenance.json": {
                "path": str(provenance.relative_to(self.catalogue_root.resolve())),
                "sha256": sha256(provenance),
                "bytes": provenance.stat().st_size,
            },
        }
        version["checksums"] = {
            name: value["sha256"] for name, value in version["artifacts"].items()
        }
        self.repository.update_version(dataset_id, 1, version)
        record = self.repository.read_record(dataset_id)
        record["status"] = "available"
        record["technical_schema"] = version["technical_schema"]
        record["versions"] = [version_summary(version)]
        self.repository.update_record(dataset_id, record)

    @staticmethod
    def _partition():
        return {
            "mode": "time_range",
            "train": {
                "start_time": "2026-09-25T10:00:00Z",
                "end_time": "2026-09-25T10:02:00Z",
            },
            "test": {
                "start_time": "2026-09-25T10:04:00Z",
                "end_time": "2026-09-25T10:06:00Z",
            },
            "gap_seconds": 120,
            "labels_source": "ground_truth",
            "labeled_evaluation": True,
        }

    def test_legacy_predefined_bundle_has_stable_partition_identity(self):
        first = self._legacy()
        self.assertRegex(first["partition_id"], r"^legacy-[0-9a-f]{32}$")
        second = self.client.get("/datasets/legacy-one").get_json()["version"]["partition"]
        self.assertEqual(first["partition_id"], second["partition_id"])
        response = self.client.get(
            f"/datasets/legacy-one/versions/1/partitions/{first['partition_id']}/training-bundle"
        )
        files = self._read_bundle(response)
        manifest = json.loads(files["manifest.json"])
        self.assertEqual(manifest["partition_checksum"], first["checksum"])
        self.assertEqual(files["labels.csv"].decode().splitlines(), ["0", "1"])

    def test_existing_legacy_projection_is_backfilled_with_stable_partition_identity(self):
        partition = self._legacy()
        version = self.repository.read_version("legacy-one", 1)
        del version["partition"]["partition_id"]
        self.repository.update_version("legacy-one", 1, version)
        restored = self.client.get("/datasets/legacy-one").get_json()["version"]["partition"]
        self.assertEqual(restored["partition_id"], partition["partition_id"])

    def test_time_range_bundle_supports_row_and_incident_derived_labels(self):
        for dataset_id, source in (("manual", "row_level"), ("derived", "incident_windows")):
            with self.subTest(source=source):
                self._available(dataset_id)
                if source == "incident_windows":
                    incident = {
                        "incident_id": "incident-one",
                        "start_time": "2026-09-25T10:04:30Z",
                        "end_time": "2026-09-25T10:05:30Z",
                        "scenario": "CPU stress",
                    }
                    self.assertEqual(
                        self.client.post(
                            f"/datasets/{dataset_id}/versions/1/incidents", json=incident
                        ).status_code,
                        201,
                    )
                    labels_payload = {"source": source}
                else:
                    labels_payload = {"source": source, "labels": [0, 0, 0, 0, 1, 1, 0]}
                self.assertEqual(
                    self.client.post(
                        f"/datasets/{dataset_id}/versions/1/labels", json=labels_payload
                    ).status_code,
                    201,
                )
                partition_response = self.client.post(
                    f"/datasets/{dataset_id}/versions/1/partitions", json=self._partition()
                )
                self.assertEqual(partition_response.status_code, 201, partition_response.get_json())
                partition = partition_response.get_json()
                bundle = self.client.get(
                    f"/datasets/{dataset_id}/versions/1/partitions/"
                    f"{partition['partition_id']}/training-bundle"
                )
                files = self._read_bundle(bundle)
                manifest = json.loads(files["manifest.json"])
                self.assertEqual(manifest["feature_order"], ["cpu", "memory"])
                self.assertEqual(manifest["label_source"]["type"], source)
                self.assertEqual(len(files["test.csv"].decode().splitlines()), 3)
                self.assertEqual(len(files["labels.csv"].decode().splitlines()), 3)

    def test_unknown_dataset_version_and_partition_are_targeted_errors(self):
        self.assertEqual(
            self.client.get(
                "/datasets/missing/versions/1/partitions/partition-x/training-bundle"
            ).status_code,
            404,
        )
        partition = self._legacy()
        self.assertEqual(
            self.client.get(
                f"/datasets/legacy-one/versions/2/partitions/{partition['partition_id']}/training-bundle"
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(
                "/datasets/legacy-one/versions/1/partitions/partition-missing/training-bundle"
            ).status_code,
            409,
        )


class CatalogueTrainingRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cgnn_task = Mock()
        self.cgnn_task.apply_async.return_value = types.SimpleNamespace(id="cgnn-task")
        self.if_task = Mock()
        self.if_task.apply_async.return_value = types.SimpleNamespace(id="if-task")
        self.app = Flask(__name__)
        configure_training_run_defaults(self.app)
        self.app.config.update(
            TESTING=True,
            EXISTING_DATASETS_ROOT=str(self.root / "existing"),
            DATASET_SNAPSHOT_STORAGE_ROOT=str(self.root / "snapshots"),
            TRAINING_RUN_STORAGE_ROOT=str(self.root / "runs"),
        )
        self.app.register_blueprint(
            create_training_runs_blueprint(
                self.cgnn_task,
                self.if_task,
                status_reader=lambda _task_id: None,
            )
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.temporary.cleanup()

    def source(self, *, rows=105, nonfinite=False):
        temporary = Path(tempfile.mkdtemp(dir=self.root))
        train = np.column_stack((np.arange(rows), np.arange(rows) + 1)).astype(float)
        test = np.column_stack((np.arange(rows), np.arange(rows) + 2)).astype(float)
        if nonfinite:
            train[0, 0] = np.nan
        labels = np.zeros(rows, dtype=int)
        files = {}
        for name, value in (("train", train), ("test", test), ("labels", labels)):
            path = temporary / f"{name}.csv"
            np.savetxt(path, value, delimiter=",")
            files[name] = path
        file_manifest = {
            f"{name}.csv": {"sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in files.items()
        }
        feature_order = ["promql.cpu|pod=a", "promql.memory|pod=a"]
        feature_hash = hashlib.sha256(
            json.dumps(feature_order, separators=(",", ":")).encode()
        ).hexdigest()
        manifest = {
            "schema_version": 1,
            "dataset_id": "catalogue-one",
            "dataset_version": 3,
            "partition_id": "partition-one",
            "partition_checksum": "a" * 64,
            "partition_mode": "time_range",
            "feature_order": feature_order,
            "feature_order_sha256": feature_hash,
            "label_source": {"type": "row_level", "sha256": "b" * 64},
            "counts": {"train": rows, "test": rows, "labels": rows, "features": 2},
            "source_artifact_checksums": {"canonical/observations.csv": "c" * 64},
            "provenance_reference": {"artifact": "provenance.json", "sha256": "d" * 64},
            "dataset_details": {
                "feature_order": feature_order,
                "step_size": 60,
                "data_entries": rows,
            },
            "files": file_manifest,
        }
        return {
            "directory": temporary,
            "temporary_root": temporary,
            "files": files,
            "details": manifest["dataset_details"],
            "manifest": manifest,
        }

    @staticmethod
    def payload(detectors):
        return {
            "dataset": {
                "source": "catalogue",
                "dataset_id": "catalogue-one",
                "version": 3,
                "partition_id": "partition-one",
            },
            "detectors": detectors,
        }

    def test_valid_catalogue_run_dispatches_both_and_preserves_provenance(self):
        source = self.source()
        original = source["files"]["train"].read_bytes()
        self.app.config["CATALOGUE_BUNDLE_FETCHER"] = lambda _dataset: source
        response = self.client.post(
            "/training_runs",
            json=self.payload(
                [
                    {"detector_id": "cgnn", "parameters": {}},
                    {"detector_id": "isolation-forest", "parameters": {}},
                ]
            ),
        )
        self.assertEqual(response.status_code, 202, response.get_json())
        body = response.get_json()
        self.assertEqual(body["dataset"]["partition_checksum"], "a" * 64)
        self.assertEqual(body["dataset"]["feature_order"], source["manifest"]["feature_order"])
        snapshot = self.root / "snapshots" / body["snapshot_id"]
        self.assertEqual((snapshot / "train.csv").read_bytes(), original)
        metadata = json.loads((snapshot / "snapshot.json").read_text())
        self.assertEqual(metadata["catalogue_provenance"]["partition_id"], "partition-one")
        self.assertFalse(source["temporary_root"].exists())
        self.cgnn_task.apply_async.assert_called_once()
        self.if_task.apply_async.assert_called_once()

    def test_if_validation_failure_does_not_block_cgnn(self):
        self.app.config["CATALOGUE_BUNDLE_FETCHER"] = lambda _dataset: self.source(nonfinite=True)
        response = self.client.post(
            "/training_runs",
            json=self.payload(
                [
                    {"detector_id": "cgnn", "parameters": {}},
                    {"detector_id": "isolation-forest", "parameters": {}},
                ]
            ),
        )
        self.assertEqual(response.status_code, 202, response.get_json())
        children = {item["detector_id"]: item for item in response.get_json()["children"]}
        self.assertEqual(children["isolation-forest"]["status"], "validation_failed")
        self.assertIn("finite", children["isolation-forest"]["validation_error"])
        self.assertEqual(children["cgnn"]["task_id"], "cgnn-task")
        self.assertEqual(response.get_json()["status"], "partial_success")
        self.cgnn_task.apply_async.assert_called_once()
        self.if_task.apply_async.assert_not_called()

    def test_cgnn_validation_failure_does_not_block_if(self):
        self.app.config["CATALOGUE_BUNDLE_FETCHER"] = lambda _dataset: self.source(rows=4)
        response = self.client.post(
            "/training_runs",
            json=self.payload(
                [
                    {"detector_id": "cgnn", "parameters": {}},
                    {"detector_id": "isolation-forest", "parameters": {}},
                ]
            ),
        )
        self.assertEqual(response.status_code, 202, response.get_json())
        children = {item["detector_id"]: item for item in response.get_json()["children"]}
        self.assertEqual(children["cgnn"]["status"], "validation_failed")
        self.assertIn("lookback", children["cgnn"]["validation_error"])
        self.assertEqual(children["isolation-forest"]["task_id"], "if-task")
        self.cgnn_task.apply_async.assert_not_called()
        self.if_task.apply_async.assert_called_once()

    def test_central_bundle_failure_dispatches_nothing(self):
        def fail(_dataset):
            raise CatalogueBundleError("partition checksum mismatch")

        self.app.config["CATALOGUE_BUNDLE_FETCHER"] = fail
        response = self.client.post(
            "/training_runs",
            json=self.payload([{"detector_id": "isolation-forest", "parameters": {}}]),
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("checksum", response.get_json()["error"])
        self.cgnn_task.apply_async.assert_not_called()
        self.if_task.apply_async.assert_not_called()
        self.assertFalse((self.root / "runs").exists())

    def test_corrupt_bundle_checksum_is_rejected(self):
        source = self.source(rows=4)
        archive = self.root / "bad.zip"
        manifest = dict(source["manifest"])
        manifest["files"] = dict(manifest["files"])
        manifest["files"]["train.csv"] = dict(manifest["files"]["train.csv"])
        manifest["files"]["train.csv"]["sha256"] = "0" * 64
        with zipfile.ZipFile(archive, "w") as target:
            target.writestr("manifest.json", json.dumps(manifest))
            for name in ("train", "test", "labels"):
                target.write(source["files"][name], arcname=f"{name}.csv")
        with self.assertRaisesRegex(CatalogueBundleError, "checksum"):
            verify_bundle(archive, self.payload([])["dataset"])

    def test_arbitrary_feature_importance_is_detector_specific_failure(self):
        self.app.config["CATALOGUE_BUNDLE_FETCHER"] = lambda _dataset: self.source()
        response = self.client.post(
            "/training_runs",
            json=self.payload(
                [
                    {"detector_id": "cgnn", "parameters": {"feature_importance": True}},
                    {"detector_id": "isolation-forest", "parameters": {}},
                ]
            ),
        )
        self.assertEqual(response.status_code, 202, response.get_json())
        cgnn = next(
            item for item in response.get_json()["children"] if item["detector_id"] == "cgnn"
        )
        self.assertEqual(cgnn["status"], "validation_failed")
        self.assertIn("unavailable", cgnn["validation_error"])
        self.if_task.apply_async.assert_called_once()


if __name__ == "__main__":
    unittest.main()
