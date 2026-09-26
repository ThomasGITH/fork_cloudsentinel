import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from flask import Flask

from data_ingestion.catalogue.api import (
    configure_catalogue_defaults,
    create_catalogue_blueprint,
)
from data_ingestion.catalogue.models import version_summary
from data_ingestion.catalogue.repository import FileCatalogueRepository


class DataCatalogueLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "legacy").mkdir()
        self.app = Flask(__name__)
        configure_catalogue_defaults(self.app)
        self.app.config.update(
            TESTING=True,
            CATALOGUE_STORAGE_ROOT=str(self.root / "catalogue"),
            LEGACY_DATASETS_ROOT=str(self.root / "legacy"),
        )
        self.app.register_blueprint(create_catalogue_blueprint())
        self.client = self.app.test_client()
        self.repository = FileCatalogueRepository(self.root / "catalogue")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def payload(self, dataset_id="lifecycle"):
        return {
            "dataset_id": dataset_id,
            "display_name": "Lifecycle dataset",
            "description": "versioned metrics",
            "purpose": ["training", "evaluation"],
            "source": {
                "type": "prometheus",
                "prometheus_source_id": "cluster-default",
                "application": "shop",
                "namespace": "production",
                "requested_targets": {
                    "services": ["checkout"],
                    "pods": ["checkout-1"],
                    "label_selectors": [],
                },
                "start_time": "2026-09-25T10:00:00Z",
                "end_time": "2026-09-25T10:06:00Z",
                "sampling_interval_seconds": 60,
                "queries": [
                    {
                        "query_id": "cpu",
                        "display_name": "CPU",
                        "promql_template": "cpu_metric",
                        "feature_name": "cpu",
                    }
                ],
            },
            "workload_context": {
                "workload_intensity": "Normal",
                "dominant_workload_characteristic": "Mixed",
                "anomaly_scenario": "Normal operation",
            },
            "partition": {"mode": "none"},
        }

    def create_available(self, dataset_id="lifecycle"):
        response = self.client.post("/datasets", json=self.payload(dataset_id))
        self.assertEqual(response.status_code, 201, response.get_json())
        version = self.repository.read_version(dataset_id, 1)
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
        version["provenance"]["configuration_frozen"] = True
        active = self.repository.artifact_directory(dataset_id, 1) / "canonical"
        active.mkdir(parents=True)
        timestamps = [f"2026-09-25T10:0{minute}:00Z" for minute in range(7)]
        with (active / "timestamps.csv").open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(["timestamp"])
            writer.writerows([[item] for item in timestamps])
        with (active / "observations.csv").open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(["timestamp", "cpu", "memory"])
            for index, timestamp in enumerate(timestamps):
                writer.writerow([timestamp, index, index + 10])
        self.repository.update_version(dataset_id, 1, version)
        record = self.repository.read_record(dataset_id)
        record["status"] = "available"
        record["technical_schema"] = version["technical_schema"]
        record["versions"] = [version_summary(version)]
        self.repository.update_record(dataset_id, record)
        return version

    def valid_partition(self, labeled=False):
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
            "labeled_evaluation": labeled,
        }

    def valid_incident(self, **updates):
        value = {
            "incident_id": "incident-1",
            "start_time": "2026-09-25T10:04:30Z",
            "end_time": "2026-09-25T10:05:30Z",
            "scenario": "CPU stress",
            "affected_services": ["checkout"],
            "affected_metrics": ["cpu"],
            "annotation": "controlled stress",
            "source": "operator",
        }
        value.update(updates)
        return value

    def test_new_version_preserves_previous_version_and_artifacts(self):
        self.create_available()
        version_path = self.root / "catalogue" / "datasets" / "lifecycle" / "versions" / "000001.json"
        before_version = version_path.read_bytes()
        timestamp_path = self.repository.artifact_directory("lifecycle", 1) / "canonical" / "timestamps.csv"
        before_timestamp_hash = hashlib.sha256(timestamp_path.read_bytes()).hexdigest()
        response = self.client.post(
            "/datasets/lifecycle/versions",
            json={
                "copy_from_version": 1,
                "source": {
                    "end_time": "2026-09-25T11:00:00Z",
                    "sampling_interval_seconds": 120,
                },
                "workload_context": {
                    "workload_intensity": "High",
                    "dominant_workload_characteristic": "CPU-intensive",
                    "anomaly_scenario": "CPU stress",
                },
            },
        )
        self.assertEqual(response.status_code, 201, response.get_json())
        body = response.get_json()
        self.assertEqual(body["latest_version"], 2)
        self.assertEqual(body["version"]["status"], "draft")
        self.assertEqual(body["version"]["copied_from_version"], 1)
        self.assertEqual(body["version"]["source"]["sampling_interval_seconds"], 120)
        self.assertEqual(version_path.read_bytes(), before_version)
        self.assertEqual(hashlib.sha256(timestamp_path.read_bytes()).hexdigest(), before_timestamp_hash)
        old = self.client.get("/datasets/lifecycle/versions/1").get_json()
        self.assertEqual(old["status"], "available")
        self.assertEqual(old["source"]["sampling_interval_seconds"], 60)

    def test_patch_updates_only_logical_metadata_and_tracks_revision(self):
        self.create_available()
        version_before = copy.deepcopy(self.repository.read_version("lifecycle", 1))
        response = self.client.patch(
            "/datasets/lifecycle",
            json={
                "display_name": "Renamed dataset",
                "tags": ["production", "cpu", "cpu"],
                "ground_truth_annotations": [
                    {"annotation": "Reviewed by operator", "source": "manual"}
                ],
            },
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        body = response.get_json()
        self.assertEqual(body["display_name"], "Renamed dataset")
        self.assertEqual(body["tags"], ["cpu", "production"])
        self.assertEqual(body["metadata_revision"], 2)
        self.assertEqual(body["metadata_history"][-1]["changed_fields"], [
            "display_name", "ground_truth_annotations", "tags"
        ])
        self.assertEqual(self.repository.read_version("lifecycle", 1), version_before)
        rejected = self.client.patch(
            "/datasets/lifecycle", json={"checksums": {"forged": "value"}}
        )
        self.assertEqual(rejected.status_code, 400)
        self.assertEqual(self.repository.read_version("lifecycle", 1), version_before)

    def test_incident_validation_and_overlap_rules(self):
        self.create_available()
        created = self.client.post(
            "/datasets/lifecycle/versions/1/incidents", json=self.valid_incident()
        )
        self.assertEqual(created.status_code, 201, created.get_json())
        self.assertEqual(created.get_json()["incident_id"], "incident-1")
        stored = self.repository.read_version("lifecycle", 1)
        self.assertEqual(len(stored["incident_context"]), 1)
        self.assertEqual(stored["incident_context"][0]["source"], "operator")

        invalid_cases = [
            self.valid_incident(incident_id="bad-order", start_time="2026-09-25T10:05:00Z", end_time="2026-09-25T10:04:00Z"),
            self.valid_incident(incident_id="outside", start_time="2026-09-25T09:59:00Z"),
            self.valid_incident(incident_id="bad-scenario", scenario="Disk stress"),
            self.valid_incident(incident_id="unsafe/id"),
            self.valid_incident(incident_id="overlap", start_time="2026-09-25T10:05:00Z", end_time="2026-09-25T10:05:45Z"),
        ]
        for payload in invalid_cases:
            with self.subTest(payload=payload["incident_id"]):
                response = self.client.post(
                    "/datasets/lifecycle/versions/1/incidents", json=payload
                )
                self.assertEqual(response.status_code, 400, response.get_json())
        overlapping = self.client.post(
            "/datasets/lifecycle/versions/1/incidents",
            json=self.valid_incident(
                incident_id="allowed-overlap",
                start_time="2026-09-25T10:05:00Z",
                end_time="2026-09-25T10:05:45Z",
                allow_overlap=True,
            ),
        )
        self.assertEqual(overlapping.status_code, 201, overlapping.get_json())

    def test_incident_derived_labels_are_separate_hashed_and_not_overwritten(self):
        self.create_available()
        immutable_before = self.repository.read_version("lifecycle", 1)
        immutable_fields = {
            key: copy.deepcopy(immutable_before[key])
            for key in ("source", "provenance", "technical_schema", "artifacts", "checksums")
        }
        self.assertEqual(
            self.client.post(
                "/datasets/lifecycle/versions/1/incidents", json=self.valid_incident()
            ).status_code,
            201,
        )
        response = self.client.post(
            "/datasets/lifecycle/versions/1/labels",
            json={"source": "incident_windows"},
        )
        self.assertEqual(response.status_code, 201, response.get_json())
        label_source = response.get_json()
        self.assertEqual(label_source["semantics"], {"0": "normal", "1": "anomaly"})
        self.assertEqual(
            label_source["derivation"]["method"],
            "timestamp_within_incident_window_inclusive",
        )
        label_path = self.root / "catalogue" / label_source["artifact"]
        before = label_path.read_bytes()
        self.assertEqual(hashlib.sha256(before).hexdigest(), label_source["sha256"])
        self.assertEqual(before.decode().splitlines(), ["label", "0", "0", "0", "0", "0", "1", "0"])
        repeated = self.client.post(
            "/datasets/lifecycle/versions/1/labels", json={"source": "row_level", "labels": [0] * 7}
        )
        self.assertEqual(repeated.status_code, 409)
        self.assertEqual(label_path.read_bytes(), before)
        stored = self.repository.read_version("lifecycle", 1)
        self.assertEqual(len(stored["incident_context"]), 1)
        for key, value in immutable_fields.items():
            self.assertEqual(stored[key], value)

    def test_manual_existing_labels_are_supported(self):
        self.create_available("manual-labels")
        response = self.client.post(
            "/datasets/manual-labels/versions/1/labels",
            json={"source": "row_level", "labels": [0, 0, 0, 0, 1, 1, 0]},
        )
        self.assertEqual(response.status_code, 201, response.get_json())
        self.assertEqual(response.get_json()["type"], "row_level")
        self.assertEqual(response.get_json()["row_count"], 7)

    def test_time_range_partition_validation_checksum_and_usage_states(self):
        self.create_available()
        unlabeled = self.client.get("/datasets/lifecycle/versions/1").get_json()
        self.assertTrue(unlabeled["usages"]["unsupervised_training"]["supported"])
        self.assertFalse(unlabeled["usages"]["labeled_evaluation"]["supported"])

        first = self.client.post(
            "/datasets/lifecycle/versions/1/partitions", json=self.valid_partition()
        )
        self.assertEqual(first.status_code, 201, first.get_json())
        partition = first.get_json()
        self.assertEqual(partition["train"]["observation_count"], 3)
        self.assertEqual(partition["test"]["observation_count"], 3)
        self.assertEqual(partition["source_dataset_version"], 1)
        self.assertEqual(partition["feature_order_reference"]["feature_count"], 2)
        second = self.client.post(
            "/datasets/lifecycle/versions/1/partitions", json=self.valid_partition()
        )
        self.assertEqual(second.status_code, 201, second.get_json())
        self.assertEqual(first.get_json()["checksum"], second.get_json()["checksum"])

        invalid = [
            {
                **self.valid_partition(),
                "train": {"start_time": "2026-09-25T10:00:00Z", "end_time": "2026-09-25T10:04:00Z"},
                "test": {"start_time": "2026-09-25T10:03:00Z", "end_time": "2026-09-25T10:06:00Z"},
            },
            {
                **self.valid_partition(),
                "test": {"start_time": "2026-09-25T10:04:00Z", "end_time": "2026-09-25T10:07:00Z"},
            },
            {**self.valid_partition(), "gap_seconds": 180},
        ]
        for index, payload in enumerate(invalid):
            with self.subTest(index=index):
                self.assertEqual(
                    self.client.post(
                        "/datasets/lifecycle/versions/1/partitions", json=payload
                    ).status_code,
                    400,
                )

        self.assertEqual(
            self.client.post(
                "/datasets/lifecycle/versions/1/incidents", json=self.valid_incident()
            ).status_code,
            201,
        )
        self.assertEqual(
            self.client.post(
                "/datasets/lifecycle/versions/1/labels", json={"source": "incident_windows"}
            ).status_code,
            201,
        )
        labeled_partition = self.client.post(
            "/datasets/lifecycle/versions/1/partitions",
            json=self.valid_partition(labeled=True),
        )
        self.assertEqual(labeled_partition.status_code, 201, labeled_partition.get_json())
        derived_usage = self.client.get("/datasets/lifecycle/versions/1").get_json()["usages"]
        self.assertTrue(derived_usage["labeled_evaluation"]["supported"])

        self.create_available("existing-labels")
        self.assertEqual(
            self.client.post(
                "/datasets/existing-labels/versions/1/labels",
                json={"source": "row_level", "labels": [0, 0, 0, 0, 1, 1, 0]},
            ).status_code,
            201,
        )
        self.assertEqual(
            self.client.post(
                "/datasets/existing-labels/versions/1/partitions",
                json=self.valid_partition(labeled=True),
            ).status_code,
            201,
        )
        existing_usage = self.client.get("/datasets/existing-labels/versions/1").get_json()["usages"]
        self.assertTrue(existing_usage["labeled_evaluation"]["supported"])

    def test_none_and_predefined_partition_contracts_remain_explicit(self):
        self.create_available()
        none = self.client.post(
            "/datasets/lifecycle/versions/1/partitions", json={"mode": "none"}
        )
        self.assertEqual(none.status_code, 201, none.get_json())
        self.assertEqual(none.get_json()["mode"], "none")
        predefined = self.client.post(
            "/datasets/lifecycle/versions/1/partitions",
            json={
                "mode": "predefined",
                "train": {"artifact": "canonical/train.csv", "row_count": 4},
                "test": {"artifact": "canonical/test.csv", "row_count": 3},
            },
        )
        self.assertEqual(predefined.status_code, 201, predefined.get_json())
        self.assertEqual(predefined.get_json()["mode"], "predefined")


if __name__ == "__main__":
    unittest.main()
