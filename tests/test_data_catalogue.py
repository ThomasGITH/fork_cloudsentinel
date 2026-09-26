import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from flask import Flask

from data_ingestion.catalogue.api import create_catalogue_blueprint
from data_ingestion.catalogue.repository import (
    FileCatalogueRepository,
    atomic_write_json,
)
from data_ingestion.catalogue.validation import CatalogueValidationError


class _FakeCeleryConfiguration:
    def update(self, *args, **kwargs):
        return None


class _FakeCelery:
    def __init__(self, *args, **kwargs):
        self.conf = _FakeCeleryConfiguration()
        self.control = types.SimpleNamespace()

    def task(self, *args, **kwargs):
        return lambda function: function


class DataCatalogueApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.catalogue_root = self.root / "catalogue"
        self.legacy_root = self.root / "legacy"
        self.legacy_root.mkdir()
        self.app = Flask(__name__)
        self.app.config.update(
            TESTING=True,
            CATALOGUE_STORAGE_ROOT=str(self.catalogue_root),
            LEGACY_DATASETS_ROOT=str(self.legacy_root),
        )
        self.app.register_blueprint(create_catalogue_blueprint())
        self.client = self.app.test_client()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def payload(self, **overrides):
        value = {
            "display_name": "Checkout metrics",
            "description": "Controlled workload",
            "purpose": ["training", "evaluation"],
            "source": {
                "type": "prometheus",
                "prometheus_source_id": "cluster-default",
                "application": "shop",
                "namespace": "production",
                "requested_targets": {
                    "services": ["checkout"],
                    "pods": [],
                    "label_selectors": ["app=checkout"],
                },
                "start_time": "2026-09-25T10:00:00Z",
                "end_time": "2026-09-25T12:00:00Z",
                "sampling_interval_seconds": 60,
                "queries": [
                    {
                        "query_id": "cpu",
                        "display_name": "CPU usage",
                        "promql": "sum(rate(cpu_total[1m]))",
                        "feature_name": "cpu_usage",
                    }
                ],
            },
            "workload_context": {
                "workload_intensity": "High",
                "dominant_workload_characteristic": "CPU-intensive",
                "anomaly_scenario": "CPU stress",
            },
            "ground_truth": {
                "labels_available": False,
                "known_incident_windows": [],
            },
            "partition": {"mode": "none"},
        }
        value.update(overrides)
        return value

    def post(self, **overrides):
        return self.client.post("/datasets", json=self.payload(**overrides))

    def write_legacy_dataset(self, dataset_id, *, details=True, declared_entries=999):
        directory = self.legacy_root / dataset_id
        directory.mkdir()
        contents = {
            f"train_{dataset_id}": "1,2\n3,4\n5,6\n",
            f"test_{dataset_id}": "7,8\n9,10\n",
            f"test_label_{dataset_id}": "0\n1\n",
        }
        for filename, content in contents.items():
            (directory / filename).write_text(content, encoding="utf-8")
        if details:
            (directory / "details.json").write_text(
                json.dumps(
                    {
                        "dataset": f"Legacy {dataset_id}",
                        "containers": ["service-a"],
                        "metrics": ["cpu", "memory"],
                        "data_entries": declared_entries,
                        "step_size": 60,
                        "duration": 120,
                        "anomaly_sequence": False,
                    }
                ),
                encoding="utf-8",
            )
        return directory, contents

    def test_post_creates_draft_record_and_none_partition(self):
        response = self.post(dataset_id="dataset-1")
        self.assertEqual(response.status_code, 201)
        body = response.get_json()
        self.assertEqual(body["dataset_id"], "dataset-1")
        self.assertEqual(body["status"], "draft")
        self.assertEqual(body["latest_version"], 1)
        self.assertEqual(body["metadata_revision"], 1)
        self.assertEqual(body["version"]["status"], "draft")
        self.assertEqual(body["version"]["partition"]["mode"], "none")
        self.assertEqual(len(body["version"]["partition"]["checksum"]), 64)
        self.assertEqual(body["version"]["source"]["queries"][0]["status"], "not_executed")
        self.assertFalse(body["version"]["provenance"]["configuration_frozen"])
        self.assertTrue((self.catalogue_root / "datasets" / "dataset-1" / "dataset.json").is_file())
        self.assertTrue(
            (self.catalogue_root / "datasets" / "dataset-1" / "versions" / "000001.json").is_file()
        )

    def test_all_workload_context_fields_accept_only_declared_values(self):
        valid_values = {
            "workload_intensity": ["Low", "Normal", "High", "Variable"],
            "dominant_workload_characteristic": [
                "CPU-intensive",
                "Memory-intensive",
                "High traffic",
                "Latency-intensive",
                "Mixed",
            ],
            "anomaly_scenario": [
                "Normal operation",
                "CPU stress",
                "Memory stress",
                "Network delay",
                "Unknown",
            ],
        }
        defaults = self.payload()["workload_context"]
        counter = 0
        for field, values in valid_values.items():
            for value in values:
                counter += 1
                context = {**defaults, field: value}
                response = self.post(dataset_id=f"valid-{counter}", workload_context=context)
                self.assertEqual(response.status_code, 201, response.get_json())
            invalid = self.post(
                dataset_id=f"invalid-{field}",
                workload_context={**defaults, field: "unsupported"},
            )
            self.assertEqual(invalid.status_code, 400)
            self.assertIn(field, invalid.get_json()["error"])

    def test_list_supports_search_filters_and_pagination(self):
        self.assertEqual(self.post(dataset_id="alpha").status_code, 201)
        second = self.payload(
            dataset_id="beta",
            display_name="Memory experiment",
            purpose="evaluation",
            workload_context={
                "workload_intensity": "Low",
                "dominant_workload_characteristic": "Memory-intensive",
                "anomaly_scenario": "Normal operation",
            },
            ground_truth={"labels_available": True},
        )
        self.assertEqual(self.client.post("/datasets", json=second).status_code, 201)

        search = self.client.get("/datasets?search=memory").get_json()
        self.assertEqual(search["total"], 1)
        self.assertEqual(search["items"][0]["dataset_id"], "beta")
        filtered = self.client.get(
            "/datasets?workload_intensity=Low&"
            "dominant_workload_characteristic=Memory-intensive&"
            "anomaly_scenario=Normal%20operation&labels_available=true&"
            "purpose=evaluation&status=draft&modality=metrics"
        ).get_json()
        self.assertEqual(filtered["total"], 1)
        paged = self.client.get("/datasets?page=2&page_size=1").get_json()
        self.assertEqual(paged["total"], 2)
        self.assertEqual(paged["pages"], 2)
        self.assertEqual(len(paged["items"]), 1)
        self.assertEqual(self.client.get("/datasets?page=0").status_code, 400)
        self.assertEqual(self.client.get("/datasets?page_size=101").status_code, 400)
        self.assertEqual(self.client.get("/datasets?labels_available=maybe").status_code, 400)

    def test_detail_returns_version_partition_schema_and_ground_truth(self):
        ground_truth = {
            "labels_available": True,
            "label_semantics": {"0": "normal", "1": "anomaly"},
            "known_incident_windows": [
                {
                    "start_time": "2026-09-25T11:00:00Z",
                    "end_time": "2026-09-25T11:10:00Z",
                    "scenario": "CPU stress",
                    "affected_services": ["checkout"],
                    "affected_metrics": ["cpu"],
                }
            ],
        }
        created = self.post(dataset_id="detail-1", ground_truth=ground_truth)
        self.assertEqual(created.status_code, 201)
        response = self.client.get("/datasets/detail-1")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["ground_truth"]["known_incident_count"], 1)
        self.assertEqual(body["version"]["ground_truth"]["label_semantics"]["1"], "anomaly")
        self.assertEqual(body["version"]["technical_schema"]["feature_count"], 0)
        self.assertEqual(body["version"]["partition"]["mode"], "none")
        self.assertIn("versions", body)

    def test_dataset_id_and_storage_paths_are_safe(self):
        for unsafe in ("../escape", "/absolute", ".", "..", "a/b", "a\\b"):
            with self.subTest(unsafe=unsafe):
                response = self.post(dataset_id=unsafe)
                self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/datasets/%2E%2E").status_code, 400)
        repository = FileCatalogueRepository(self.catalogue_root)
        with self.assertRaises(CatalogueValidationError):
            repository.read_record("../escape")
        self.assertFalse((self.root / "escape").exists())

    def test_atomic_write_preserves_existing_json_when_replace_fails(self):
        path = self.root / "atomic" / "record.json"
        atomic_write_json(path, {"value": "old"})
        with patch(
            "data_ingestion.catalogue.repository.os.replace",
            side_effect=OSError("replace failed"),
        ):
            with self.assertRaises(OSError):
                atomic_write_json(path, {"value": "new"})
        self.assertEqual(json.loads(path.read_text()), {"value": "old"})
        self.assertEqual(list(path.parent.glob(".record.json.*.tmp")), [])

    def test_legacy_projection_uses_actual_facts_hashes_and_predefined_partition(self):
        directory, contents = self.write_legacy_dataset("legacy-1", declared_entries=999)
        before = {name: (directory / name).read_bytes() for name in contents}
        response = self.client.get("/datasets/legacy-1")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        version = body["version"]
        schema = version["technical_schema"]
        self.assertEqual(body["status"], "available")
        self.assertEqual(schema["train_observation_count"], 3)
        self.assertEqual(schema["test_observation_count"], 2)
        self.assertEqual(schema["observation_count"], 5)
        self.assertEqual(schema["feature_count"], 2)
        self.assertEqual(schema["feature_order"], ["service-a_cpu", "service-a_memory"])
        partition = version["partition"]
        self.assertEqual(partition["mode"], "predefined")
        self.assertEqual(partition["train"]["row_count"], 3)
        self.assertEqual(partition["test"]["row_count"], 2)
        self.assertEqual(partition["labels"]["row_count"], 2)
        self.assertEqual(partition["labels"]["semantics"], {"0": "normal", "1": "anomaly"})
        for name, filename in (
            ("train", "train_legacy-1"),
            ("test", "test_legacy-1"),
            ("labels", "test_label_legacy-1"),
        ):
            self.assertEqual(
                version["checksums"][name],
                hashlib.sha256((directory / filename).read_bytes()).hexdigest(),
            )
        warnings = version["validation"]["warnings"]
        self.assertTrue(any("data_entries=999" in warning for warning in warnings))
        self.assertEqual(
            {name: (directory / name).read_bytes() for name in contents}, before
        )

    def test_legacy_dataset_without_details_remains_visible_as_draft(self):
        self.write_legacy_dataset("legacy-no-details", details=False)
        listing = self.client.get("/datasets?search=legacy-no-details").get_json()
        self.assertEqual(listing["total"], 1)
        self.assertEqual(listing["items"][0]["status"], "draft")
        detail = self.client.get("/datasets/legacy-no-details").get_json()
        self.assertEqual(detail["version"]["source"]["type"], "legacy")
        self.assertEqual(detail["version"]["partition"]["mode"], "predefined")
        self.assertTrue(
            any("details.json is missing" in warning for warning in detail["version"]["validation"]["warnings"])
        )
        self.assertEqual(detail["workload_context"]["workload_intensity"], None)
        self.assertEqual(detail["version"]["provenance"]["prometheus_provenance"], "unknown")

    def test_time_range_partition_is_accepted_and_checksummed(self):
        partition = {
            "mode": "time_range",
            "train": {
                "start_time": "2026-09-25T10:00:00Z",
                "end_time": "2026-09-25T10:45:00Z",
            },
            "test": {
                "start_time": "2026-09-25T11:00:00Z",
                "end_time": "2026-09-25T12:00:00Z",
            },
            "gap_seconds": 900,
            "labels_source": "ground_truth",
        }
        response = self.post(dataset_id="time-partition", partition=partition)
        self.assertEqual(response.status_code, 201, response.get_json())
        stored = response.get_json()["version"]["partition"]
        self.assertEqual(stored["mode"], "time_range")
        self.assertEqual(stored["gap_seconds"], 900)
        self.assertEqual(len(stored["checksum"]), 64)

    def test_invalid_time_range_partitions_are_rejected(self):
        cases = (
            {
                "mode": "time_range",
                "train": {"start_time": "2026-09-25T10:30:00Z", "end_time": "2026-09-25T10:00:00Z"},
                "test": {"start_time": "2026-09-25T11:00:00Z", "end_time": "2026-09-25T12:00:00Z"},
            },
            {
                "mode": "time_range",
                "train": {"start_time": "2026-09-25T10:00:00Z", "end_time": "2026-09-25T11:30:00Z"},
                "test": {"start_time": "2026-09-25T11:00:00Z", "end_time": "2026-09-25T12:00:00Z"},
            },
            {
                "mode": "time_range",
                "train": {"start_time": "2026-09-25T10:00:00Z", "end_time": "2026-09-25T11:00:00Z"},
                "test": {"start_time": "2026-09-25T11:05:00Z", "end_time": "2026-09-25T12:00:00Z"},
                "gap_seconds": 600,
            },
            {
                "mode": "time_range",
                "train": {"start_time": "2026-09-25T09:00:00Z", "end_time": "2026-09-25T10:00:00Z"},
                "test": {"start_time": "2026-09-25T11:00:00Z", "end_time": "2026-09-25T12:00:00Z"},
            },
            {
                "mode": "time_range",
                "train": {"start_time": "2026-09-25T10:00:00Z", "end_time": "2026-09-25T11:00:00Z"},
                "test": {"start_time": "2026-09-25T11:00:00Z", "end_time": "2026-09-25T12:00:00Z"},
                "gap_seconds": -1,
            },
        )
        for index, partition in enumerate(cases):
            with self.subTest(index=index):
                response = self.post(dataset_id=f"invalid-time-{index}", partition=partition)
                self.assertEqual(response.status_code, 400)


class DataIngestionAppRegistrationTests(unittest.TestCase):
    def test_data_catalogue_routes_are_registered_on_the_service_app(self):
        repository_root = Path(__file__).resolve().parents[1]
        app_path = repository_root / "data_ingestion" / "app.py"
        celery_module = types.ModuleType("celery")
        celery_module.Celery = _FakeCelery
        kubernetes_module = types.ModuleType("kubernetes")
        kubernetes_module.client = types.SimpleNamespace()
        kubernetes_module.config = types.SimpleNamespace(ConfigException=Exception)
        redis_module = types.ModuleType("redis")
        redis_module.StrictRedis = Mock(return_value=Mock())
        collector_module = types.ModuleType("data_collector")
        collector_module.collect_crca_data = Mock()
        collector_module.fetch_metrics = Mock()
        config_module = types.ModuleType("config")
        config_module.set_initial_metric_config = Mock()
        config_module.get_config = Mock(return_value={})
        config_module.set_config = Mock()

        module_name = "dc2_data_ingestion_app"
        specification = importlib.util.spec_from_file_location(module_name, app_path)
        module = importlib.util.module_from_spec(specification)
        stubs = {
            "celery": celery_module,
            "kubernetes": kubernetes_module,
            "redis": redis_module,
            "data_collector": collector_module,
            "config": config_module,
        }
        with patch.dict(sys.modules, stubs):
            assert specification.loader is not None
            specification.loader.exec_module(module)

        routes = {rule.rule for rule in module.app.url_map.iter_rules()}
        self.assertIn("/datasets", routes)
        self.assertIn("/datasets/<dataset_id>", routes)
        self.assertIn("/datasets/<dataset_id>/versions/<int:version>/fetch", routes)
        self.assertIn("/datasets/<dataset_id>/fetch-status", routes)
        self.assertIn("/datasets/<dataset_id>/preview", routes)

    def test_worker_entrypoint_registers_catalogue_fetch_task_with_real_celery(self):
        repository_root = Path(__file__).resolve().parents[1]
        code = """
import sys
import types
kubernetes = types.ModuleType('kubernetes')
kubernetes.client = types.SimpleNamespace()
kubernetes.config = types.SimpleNamespace(ConfigException=Exception)
sys.modules['kubernetes'] = kubernetes
import app
registered = app.celery.tasks['fetch_catalogue_dataset_task']
assert registered.name == 'fetch_catalogue_dataset_task'
assert registered.run.__module__ == 'app'
assert registered.run.__name__ == 'fetch_catalogue_dataset_task'
assert 'app.monitoring_task' in app.celery.tasks
assert app.app.extensions['catalogue_fetch_dispatch'] is app._dispatch_catalogue_fetch
print('registry_check=PASS')
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(repository_root), environment.get("PYTHONPATH")])
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repository_root / "data_ingestion",
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("registry_check=PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
