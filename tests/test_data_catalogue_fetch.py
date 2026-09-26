import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import requests
from flask import Flask

from data_ingestion.catalogue.api import (
    configure_catalogue_defaults,
    create_catalogue_blueprint,
)
from data_ingestion.catalogue.fetching import (
    CatalogueFetchError,
    FetchLimits,
    assemble_time_series,
    build_query_executions,
    execute_catalogue_fetch,
)
from data_ingestion.catalogue.repository import FileCatalogueRepository


def prometheus_payload(series, warnings=None):
    payload = {
        "status": "success",
        "data": {"resultType": "matrix", "result": series},
    }
    if warnings:
        payload["warnings"] = warnings
    return payload


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self.body = json.dumps(payload).encode("utf-8")

    def iter_content(self, chunk_size=65536):
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start : start + chunk_size]


class CatalogueFetchTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.app = Flask(__name__)
        configure_catalogue_defaults(self.app)
        self.app.config.update(
            TESTING=True,
            CATALOGUE_STORAGE_ROOT=str(self.root / "catalogue"),
            LEGACY_DATASETS_ROOT=str(self.root / "legacy"),
            CATALOGUE_PROMETHEUS_SOURCES={"cluster-default": "http://prometheus.invalid"},
            CATALOGUE_MAX_RETRIES=0,
            CATALOGUE_MAX_PREVIEW_ROWS=2,
        )
        (self.root / "legacy").mkdir()
        self.dispatch = Mock()
        self.app.extensions["catalogue_fetch_dispatch"] = self.dispatch
        self.app.register_blueprint(create_catalogue_blueprint())
        self.client = self.app.test_client()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def dataset_payload(self, *, queries=None, source_updates=None):
        source = {
            "type": "prometheus",
            "prometheus_source_id": "cluster-default",
            "application": "shop",
            "namespace": "production",
            "requested_targets": {
                "services": ["checkout"],
                "pods": ["checkout-2", "checkout-1"],
                "label_selectors": [],
            },
            "start_time": "2026-09-25T10:00:00Z",
            "end_time": "2026-09-25T10:02:00Z",
            "sampling_interval_seconds": 60,
            "queries": queries
            or [
                {
                    "query_id": "cpu",
                    "display_name": "CPU",
                    "promql_template": 'rate(cpu{namespace="${namespace}"}[1m])',
                    "feature_name": "cpu",
                    "required": True,
                    "execution_mode": "single",
                    "identity_labels": ["pod"],
                }
            ],
        }
        source.update(source_updates or {})
        return {
            "dataset_id": "fetch-dataset",
            "display_name": "Fetch dataset",
            "description": "test",
            "purpose": ["training"],
            "source": source,
            "workload_context": {
                "workload_intensity": "Normal",
                "dominant_workload_characteristic": "Mixed",
                "anomaly_scenario": "Normal operation",
            },
            "partition": {"mode": "none"},
        }

    def create_dataset(self, **kwargs):
        response = self.client.post("/datasets", json=self.dataset_payload(**kwargs))
        self.assertEqual(response.status_code, 201, response.get_json())
        return response.get_json()

    def start_fetch(self):
        return self.client.post("/datasets/fetch-dataset/versions/1/fetch")

    def fake_get(self, payload):
        return Mock(return_value=FakeResponse(payload))

    def execute(self, response_payload):
        repository = FileCatalogueRepository(self.app.config["CATALOGUE_STORAGE_ROOT"])
        version = repository.read_version("fetch-dataset", 1)
        fetch = version["fetch"]
        return execute_catalogue_fetch(
            "fetch-dataset",
            1,
            fetch["attempt_id"],
            dict(self.app.config),
            request_get=self.fake_get(response_payload),
        )

    def test_fetch_endpoint_dispatches_exactly_one_one_shot_task(self):
        self.create_dataset()
        response = self.start_fetch()
        self.assertEqual(response.status_code, 202, response.get_json())
        body = response.get_json()
        self.assertEqual(body["status"], "fetching")
        self.assertEqual(self.dispatch.call_count, 1)
        self.assertEqual(
            self.dispatch.call_args.args,
            ("fetch-dataset", 1, body["attempt_id"], body["fetch_task_id"]),
        )
        status = self.client.get("/datasets/fetch-dataset/fetch-status?version=1")
        self.assertEqual(status.get_json()["phase"], "fetching")
        self.assertEqual(status.get_json()["fetch_task_id"], body["fetch_task_id"])

    def test_source_resolver_and_caller_url_are_rejected_before_dispatch(self):
        self.create_dataset(source_updates={"prometheus_source_id": "unknown"})
        response = self.start_fetch()
        self.assertEqual(response.status_code, 400)
        self.assertIn("unknown Prometheus source", response.get_json()["error"])
        self.dispatch.assert_not_called()

        unsafe = self.dataset_payload(source_updates={"url": "http://caller.example"})
        unsafe["dataset_id"] = "unsafe-source"
        rejected = self.client.post("/datasets", json=unsafe)
        self.assertEqual(rejected.status_code, 400)
        self.assertIn("URLs and credentials", rejected.get_json()["error"])

    def test_templates_single_and_per_target_are_safe_and_deterministic(self):
        queries = [
            {
                "query_id": "single",
                "display_name": "single",
                "promql_template": 'metric{namespace="${namespace}",pod="${pod}"}',
                "target_context": {"pod": 'pod"\\name'},
                "execution_mode": "single",
            },
            {
                "query_id": "per-pod",
                "display_name": "per pod",
                "promql_template": 'metric{pod="${pod}"}',
                "execution_mode": "per_target",
                "target_type": "pod",
            },
        ]
        version = self.create_dataset(queries=queries)["version"]
        limits = FetchLimits.from_config(dict(self.app.config))
        executions = build_query_executions(version, limits)
        self.assertIn('pod="pod\\"\\\\name"', executions[0]["resolved_query"])
        self.assertEqual(
            [item["resolved_target"]["pod"] for item in executions[1:]],
            ["checkout-1", "checkout-2"],
        )

    def test_all_series_missing_values_and_order_are_preserved(self):
        self.create_dataset()
        self.assertEqual(self.start_fetch().status_code, 202)
        payload = prometheus_payload(
            [
                {"metric": {"pod": "b"}, "values": [[1790330400, "2"], [1790330520, "4"]]},
                {"metric": {"pod": "a"}, "values": [[1790330400, "1"], [1790330460, "3"]]},
            ]
        )
        result = self.execute(payload)
        self.assertEqual(result["status"], "available")
        repository = FileCatalogueRepository(self.app.config["CATALOGUE_STORAGE_ROOT"])
        version = repository.read_version("fetch-dataset", 1)
        self.assertEqual(version["technical_schema"]["feature_order"], ["cpu|pod=a", "cpu|pod=b"])
        self.assertEqual(version["technical_schema"]["missing_values"]["total"], 2)
        matrix = repository.artifact_directory("fetch-dataset", 1) / "canonical" / "feature_matrix.csv"
        with matrix.open(encoding="utf-8") as source:
            rows = list(csv.reader(source))
        self.assertEqual(rows[2], ["3.0", ""])
        self.assertEqual(rows[3], ["", "4.0"])

    def test_feature_collision_and_required_empty_query_fail(self):
        self.create_dataset(
            queries=[
                {
                    "query_id": "collision",
                    "display_name": "collision",
                    "promql_template": "metric",
                    "feature_name": "same",
                    "required": True,
                }
            ]
        )
        self.assertEqual(self.start_fetch().status_code, 202)
        collision = prometheus_payload(
            [
                {"metric": {}, "values": [[1790330400, "1"]]},
                {"metric": {}, "values": [[1790330400, "2"]]},
            ]
        )
        with self.assertRaisesRegex(CatalogueFetchError, "collision"):
            self.execute(collision)
        repository = FileCatalogueRepository(self.app.config["CATALOGUE_STORAGE_ROOT"])
        self.assertEqual(repository.read_version("fetch-dataset", 1)["status"], "failed")
        self.assertFalse(repository.artifact_directory("fetch-dataset", 1).exists())

        second = self.dataset_payload()
        second["dataset_id"] = "required-empty"
        self.assertEqual(self.client.post("/datasets", json=second).status_code, 201)
        started = self.client.post("/datasets/required-empty/versions/1/fetch").get_json()
        with self.assertRaisesRegex(CatalogueFetchError, "returned no series"):
            execute_catalogue_fetch(
                "required-empty",
                1,
                started["attempt_id"],
                dict(self.app.config),
                request_get=self.fake_get(prometheus_payload([])),
            )

    def test_optional_empty_query_is_warning_when_another_query_has_data(self):
        self.create_dataset(
            queries=[
                {
                    "query_id": "required",
                    "display_name": "required",
                    "promql_template": "required_metric",
                    "feature_name": "required",
                },
                {
                    "query_id": "optional",
                    "display_name": "optional",
                    "promql_template": "optional_metric",
                    "feature_name": "optional",
                    "required": False,
                },
            ]
        )
        self.assertEqual(self.start_fetch().status_code, 202)
        responses = [
            FakeResponse(prometheus_payload([{"metric": {}, "values": [[1790330400, "1"]]}])),
            FakeResponse(prometheus_payload([])),
        ]
        repository = FileCatalogueRepository(self.app.config["CATALOGUE_STORAGE_ROOT"])
        fetch = repository.read_version("fetch-dataset", 1)["fetch"]
        execute_catalogue_fetch(
            "fetch-dataset",
            1,
            fetch["attempt_id"],
            dict(self.app.config),
            request_get=Mock(side_effect=responses),
        )
        status = self.client.get("/datasets/fetch-dataset/fetch-status?version=1").get_json()
        self.assertEqual(status["status"], "available")
        self.assertTrue(any("optional" in warning for warning in status["warnings"]))

    def test_success_writes_complete_artifacts_checksums_and_limited_preview(self):
        self.create_dataset()
        self.assertEqual(self.start_fetch().status_code, 202)
        payload = prometheus_payload(
            [{"metric": {"pod": "checkout"}, "values": [[1790330400, "1"], [1790330460, "2"]]}]
        )
        self.execute(payload)
        repository = FileCatalogueRepository(self.app.config["CATALOGUE_STORAGE_ROOT"])
        active = repository.artifact_directory("fetch-dataset", 1)
        expected = {
            "fetch_request.json",
            "provenance.json",
            "schema.json",
            "validation.json",
            "checksums.json",
            "attempt.json",
            "raw/q0000-t0000.json",
            "canonical/observations.csv",
            "canonical/feature_matrix.csv",
            "canonical/timestamps.csv",
        }
        self.assertEqual(
            {str(path.relative_to(active)) for path in active.rglob("*") if path.is_file()},
            expected,
        )
        checksums = json.loads((active / "checksums.json").read_text())
        self.assertEqual(set(checksums), expected - {"checksums.json"})
        preview = self.client.get("/datasets/fetch-dataset/preview?version=1&limit=1")
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.get_json()["returned_rows"], 1)
        self.assertTrue(preview.get_json()["rows"][0]["timestamp"].endswith("Z"))
        self.assertEqual(
            self.client.get("/datasets/fetch-dataset/preview?version=1&limit=3").status_code,
            400,
        )

    def test_staging_promotion_failure_never_creates_an_available_version(self):
        self.create_dataset()
        started = self.start_fetch().get_json()
        payload = prometheus_payload(
            [{"metric": {"pod": "checkout"}, "values": [[1790330400, "1"]]}]
        )
        with patch.object(
            FileCatalogueRepository,
            "promote_staging",
            side_effect=OSError("promotion failed"),
        ):
            with self.assertRaises(OSError):
                execute_catalogue_fetch(
                    "fetch-dataset",
                    1,
                    started["attempt_id"],
                    dict(self.app.config),
                    request_get=self.fake_get(payload),
                )
        repository = FileCatalogueRepository(self.app.config["CATALOGUE_STORAGE_ROOT"])
        self.assertEqual(repository.read_version("fetch-dataset", 1)["status"], "failed")
        self.assertFalse(repository.artifact_directory("fetch-dataset", 1).exists())
        self.assertTrue(
            repository.staging_directory("fetch-dataset", 1, started["attempt_id"]).is_dir()
        )

    def test_timeout_and_api_error_create_safe_failed_status_without_active_data(self):
        for index, side_effect in enumerate(
            [requests.Timeout("secret-url"), FakeResponse({"status": "error", "errorType": "bad_data"})]
        ):
            payload = self.dataset_payload()
            payload["dataset_id"] = f"failure-{index}"
            self.assertEqual(self.client.post("/datasets", json=payload).status_code, 201)
            started = self.client.post(f"/datasets/failure-{index}/versions/1/fetch").get_json()
            getter = Mock(side_effect=side_effect) if isinstance(side_effect, Exception) else Mock(return_value=side_effect)
            with self.assertRaises(CatalogueFetchError):
                execute_catalogue_fetch(
                    f"failure-{index}",
                    1,
                    started["attempt_id"],
                    dict(self.app.config),
                    request_get=getter,
                )
            status = self.client.get(
                f"/datasets/failure-{index}/fetch-status?version=1"
            ).get_json()
            self.assertEqual(status["status"], "failed")
            self.assertNotIn("secret-url", status["error"]["message"])
            repository = FileCatalogueRepository(self.app.config["CATALOGUE_STORAGE_ROOT"])
            self.assertFalse(repository.artifact_directory(f"failure-{index}", 1).exists())
            if index == 0:
                retried = self.client.post(f"/datasets/failure-{index}/versions/1/fetch")
                self.assertEqual(retried.status_code, 202, retried.get_json())

    def test_bounded_http_retry_can_recover_without_a_second_celery_task(self):
        self.app.config["CATALOGUE_MAX_RETRIES"] = 1
        self.create_dataset()
        started = self.start_fetch().get_json()
        payload = prometheus_payload(
            [{"metric": {"pod": "checkout"}, "values": [[1790330400, "1"]]}]
        )
        getter = Mock(side_effect=[requests.Timeout("lost response"), FakeResponse(payload)])
        result = execute_catalogue_fetch(
            "fetch-dataset",
            1,
            started["attempt_id"],
            dict(self.app.config),
            request_get=getter,
        )
        self.assertEqual(result["status"], "available")
        self.assertEqual(getter.call_count, 2)
        self.assertEqual(self.dispatch.call_count, 1)

    def test_fetch_limits_and_unknown_template_variables_fail_before_dispatch(self):
        self.app.config["CATALOGUE_MAX_QUERY_COUNT"] = 1
        too_many = self.dataset_payload(
            queries=[
                {"query_id": "one", "display_name": "one", "promql_template": "one"},
                {"query_id": "two", "display_name": "two", "promql_template": "two"},
            ]
        )
        self.assertEqual(self.client.post("/datasets", json=too_many).status_code, 201)
        response = self.start_fetch()
        self.assertEqual(response.status_code, 400)
        self.dispatch.assert_not_called()

        other = self.dataset_payload(
            queries=[
                {
                    "query_id": "bad-template",
                    "display_name": "bad",
                    "promql_template": 'metric{node="${node}"}',
                }
            ]
        )
        other["dataset_id"] = "bad-template"
        self.assertEqual(self.client.post("/datasets", json=other).status_code, 201)
        rejected = self.client.post("/datasets/bad-template/versions/1/fetch")
        self.assertEqual(rejected.status_code, 400)
        self.assertIn("unsupported variables", rejected.get_json()["error"])


class LocalPrometheusHandler(BaseHTTPRequestHandler):
    requests = []

    def do_GET(self):
        parsed = urlparse(self.path)
        self.__class__.requests.append({"path": parsed.path, "query": parse_qs(parsed.query)})
        body = json.dumps(
            prometheus_payload(
                [
                    {
                        "metric": {"pod": "local-pod"},
                        "values": [[1790330400, "1.5"], [1790330460, "9.5"]],
                    }
                ]
            )
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return None


class CatalogueFetchLocalHttpIntegrationTests(unittest.TestCase):
    def test_draft_to_real_http_fetch_artifacts_available_and_preview(self):
        temporary_directory = tempfile.TemporaryDirectory()
        server = ThreadingHTTPServer(("127.0.0.1", 0), LocalPrometheusHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            root = Path(temporary_directory.name)
            (root / "legacy").mkdir()
            app = Flask(__name__)
            configure_catalogue_defaults(app)
            app.config.update(
                TESTING=True,
                CATALOGUE_STORAGE_ROOT=str(root / "catalogue"),
                LEGACY_DATASETS_ROOT=str(root / "legacy"),
                CATALOGUE_PROMETHEUS_SOURCES={
                    "cluster-default": f"http://127.0.0.1:{server.server_port}"
                },
                CATALOGUE_MAX_RETRIES=0,
            )

            def synchronous_dispatch(dataset_id, version, attempt_id, task_id):
                execute_catalogue_fetch(
                    dataset_id, version, attempt_id, dict(app.config)
                )

            app.extensions["catalogue_fetch_dispatch"] = synchronous_dispatch
            app.register_blueprint(create_catalogue_blueprint())
            client = app.test_client()
            payload = CatalogueFetchTests.dataset_payload(Mock())
            response = client.post("/datasets", json=payload)
            self.assertEqual(response.status_code, 201, response.get_json())
            started = client.post("/datasets/fetch-dataset/versions/1/fetch")
            self.assertEqual(started.status_code, 202, started.get_json())
            status = client.get("/datasets/fetch-dataset/fetch-status?version=1").get_json()
            self.assertEqual(status["status"], "available")
            self.assertEqual(LocalPrometheusHandler.requests[-1]["path"], "/api/v1/query_range")
            preview = client.get("/datasets/fetch-dataset/preview?version=1&limit=2")
            self.assertEqual(preview.status_code, 200, preview.get_json())
            self.assertEqual(preview.get_json()["returned_rows"], 2)
            repository = FileCatalogueRepository(root / "catalogue")
            self.assertTrue(
                (repository.artifact_directory("fetch-dataset", 1) / "raw" / "q0000-t0000.json").is_file()
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            temporary_directory.cleanup()


if __name__ == "__main__":
    unittest.main()
