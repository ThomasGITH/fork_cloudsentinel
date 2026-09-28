from unittest.mock import Mock, patch

from django.test import Client, SimpleTestCase
from django.urls import resolve, reverse

from config_app.catalogue_client import CatalogueClientError, DataCatalogueClient


def version_payload(status="available"):
    return {
        "schema_version": 1,
        "dataset_id": "ds_demo",
        "version": 1,
        "status": status,
        "created_at": "2026-09-25T10:00:00Z",
        "workload_context": {
            "workload_intensity": "Normal",
            "dominant_workload_characteristic": "Mixed",
            "anomaly_scenario": "Normal operation",
        },
        "source": {
            "type": "prometheus",
            "prometheus_source_id": "cluster-default",
            "application": "shop",
            "namespace": "production",
            "requested_targets": {"services": ["checkout"], "pods": [], "label_selectors": []},
            "resolved_targets": {"services": ["checkout"], "pods": []},
            "start_time": "2026-09-25T10:00:00Z",
            "end_time": "2026-09-25T12:00:00Z",
            "sampling_interval_seconds": 60,
            "fetch_timestamp": "2026-09-25T12:01:00Z",
            "queries": [{"query_id": "cpu", "display_name": "CPU", "promql_template": "rate(cpu[1m])", "required": True}],
        },
        "technical_schema": {
            "feature_count": 1,
            "observation_count": 121,
            "feature_order": ["cpu"],
            "missing_values": {"total": 0, "by_feature": {"cpu": 0}},
        },
        "ground_truth": {"labels_available": False, "active_label_source": None},
        "incident_context": [],
        "partition": {"mode": "none", "checksum": "partition-checksum"},
        "partition_history": [],
        "validation": {"status": "passed", "warnings": [], "errors": []},
        "provenance": {"resolved_queries": [], "query_warnings": [], "artifact": "/private/catalogue/raw.json"},
        "artifacts": {"canonical": "/private/catalogue/data.csv"},
        "checksums": {"canonical": "abc123"},
        "usages": {
            "unsupervised_training": {"supported": True, "reasons": []},
            "labeled_evaluation": {"supported": False, "reasons": ["No labels"]},
        },
    }


def detail_payload(status="available"):
    version = version_payload(status)
    return {
        "dataset_id": "ds_demo",
        "display_name": "Checkout metrics",
        "description": "Metrics for checkout",
        "purpose": ["training"],
        "status": status,
        "created_at": "2026-09-25T10:00:00Z",
        "metadata_revision": 1,
        "latest_version": 1,
        "workload_context": version["workload_context"],
        "ground_truth": {"labels_available": False, "known_incident_count": 0},
        "technical_schema": version["technical_schema"],
        "versions": [{"version": 1, "status": status}],
        "version": version,
        "usages": version["usages"],
    }


class DataCatalogueUiTests(SimpleTestCase):
    def setUp(self):
        self.client_api = Mock()
        self.client_api.get_dataset.return_value = detail_payload()
        self.client_api.get_version.return_value = version_payload()
        patcher = patch("config_app.views.data_catalogue.get_catalogue_client", return_value=self.client_api)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_navigation_and_routes(self):
        self.client_api.list_datasets.return_value = {"items": [], "page": 1, "page_size": 20, "total": 0, "pages": 0}
        response = self.client.get(reverse("data_catalogue_overview"))
        self.assertContains(response, "Data Catalogue")
        self.assertContains(response, 'class="dropdown-item active"', html=False)
        self.assertEqual(self.client.get(reverse("data_catalogue_new")).status_code, 200)
        self.assertEqual(self.client.get(reverse("data_catalogue_detail", args=["ds_demo"])).status_code, 200)
        self.assertEqual(self.client.get(reverse("data_catalogue_new_version", args=["ds_demo"])).status_code, 200)
        home = self.client.get(reverse("home"))
        self.assertContains(home, reverse("data_catalogue_overview"))

    def test_overview_forwards_filters_and_pagination(self):
        self.client_api.list_datasets.return_value = {"items": [], "page": 2, "page_size": 10, "total": 0, "pages": 0}
        response = self.client.get(reverse("data_catalogue_overview"), {"search": "cpu", "status": "available", "page": 2, "page_size": 10})
        self.assertEqual(response.status_code, 200)
        self.client_api.list_datasets.assert_called_once_with({"search": "cpu", "status": "available", "page": "2", "page_size": "10"})

    def test_safe_backend_error_translation(self):
        self.client_api.list_datasets.side_effect = CatalogueClientError("Service at http://secret.internal failed", retryable=True)
        response = self.client.get(reverse("data_catalogue_overview"))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "secret.internal")
        self.assertContains(response, "Retry")

    def valid_form(self):
        return {
            "display_name": "Checkout metrics",
            "description": "A dataset",
            "purpose": "training",
            "application": "shop",
            "namespace": "production",
            "services": "checkout",
            "start_time_utc": "2026-09-25T10:00:00Z",
            "end_time_utc": "2026-09-25T12:00:00Z",
            "sampling_interval_seconds": "60",
            "queries_json": '[{"query_id":"cpu","display_name":"CPU","promql_template":"rate(cpu[1m])","required":true,"execution_mode":"single"}]',
            "workload_intensity": "Normal",
            "dominant_workload_characteristic": "Mixed",
            "anomaly_scenario": "Normal operation",
            "partition_mode": "none",
        }

    def test_create_dataset_payload(self):
        self.client_api.create_dataset.return_value = detail_payload("draft")
        data = self.valid_form(); data["action"] = "save"
        response = self.client.post(reverse("data_catalogue_new"), data)
        self.assertEqual(response.status_code, 302)
        payload = self.client_api.create_dataset.call_args.args[0]
        self.assertEqual(payload["source"]["prometheus_source_id"], "cluster-default")
        self.assertEqual(payload["source"]["queries"][0]["query_id"], "cpu")
        self.assertFalse(payload["ground_truth"]["labels_available"])
        self.assertNotIn("url", payload["source"])

    def test_create_and_fetch_flow(self):
        self.client_api.create_dataset.return_value = detail_payload("draft")
        data = self.valid_form(); data["action"] = "save_and_fetch"
        response = self.client.post(reverse("data_catalogue_new"), data)
        self.assertRedirects(response, "/data-catalogue/ds_demo/?version=1", fetch_redirect_response=False)
        self.client_api.start_fetch.assert_called_once_with("ds_demo", 1)

    def test_fetch_status_and_preview_proxies(self):
        self.client_api.fetch_status.return_value = {"status": "fetching", "phase": "queries", "progress": {"completed": 1, "total": 2}, "artifact": "/secret"}
        status = self.client.get(reverse("data_catalogue_fetch_status", args=["ds_demo"]), {"version": 1})
        self.assertEqual(status.status_code, 200)
        self.assertNotIn("artifact", status.json())
        self.client_api.preview.return_value = {
            "feature_names": ["cpu", "memory"],
            "rows": [
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "values": {"cpu": 1.0, "memory": None},
                    "missing": {"cpu": False, "memory": True},
                }
            ],
        }
        preview = self.client.get(reverse("data_catalogue_preview", args=["ds_demo"]), {"version": 1, "limit": 10})
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.json(), self.client_api.preview.return_value)
        self.client_api.preview.assert_called_once_with("ds_demo", 1, 10)

    def test_incident_label_partition_and_version_payloads(self):
        incident_url = reverse("data_catalogue_incident", args=["ds_demo", 1])
        self.client.post(incident_url, {"incident_start_time_utc": "2026-09-25T10:30:00Z", "incident_end_time_utc": "2026-09-25T10:40:00Z", "incident_scenario": "CPU stress", "incident_services": "checkout", "incident_metrics": "cpu"})
        incident = self.client_api.add_incident.call_args.args[2]
        self.assertEqual(incident["scenario"], "CPU stress")
        self.client.post(reverse("data_catalogue_derive_labels", args=["ds_demo", 1]))
        self.client_api.derive_labels.assert_called_once_with("ds_demo", 1)
        self.client.post(reverse("data_catalogue_partition", args=["ds_demo", 1]), {"new_partition_mode": "time_range", "new_train_start_time_utc": "2026-09-25T10:00:00Z", "new_train_end_time_utc": "2026-09-25T10:50:00Z", "new_test_start_time_utc": "2026-09-25T11:00:00Z", "new_test_end_time_utc": "2026-09-25T12:00:00Z", "new_gap_seconds": "600", "new_labeled_evaluation": "on"})
        partition = self.client_api.create_partition.call_args.args[2]
        self.assertEqual(partition["mode"], "time_range")
        self.assertTrue(partition["labeled_evaluation"])
        self.client_api.create_version.return_value = {**detail_payload("draft"), "latest_version": 2}
        form = self.valid_form(); form.pop("display_name"); form.pop("description"); form.pop("purpose")
        response = self.client.post(reverse("data_catalogue_new_version", args=["ds_demo"]), form)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.client_api.create_version.call_args.args[1]["copy_from_version"], 1)

    def test_internal_urls_and_artifact_paths_are_not_rendered(self):
        response = self.client.get(reverse("data_catalogue_detail", args=["ds_demo"]), {"version": 1})
        self.assertNotContains(response, "/private/catalogue")
        self.assertNotContains(response, "data-ingestion-service")
        self.assertContains(response, "abc123")

    def test_mutating_routes_require_csrf(self):
        csrf_client = Client(enforce_csrf_checks=True)
        for url in (
            reverse("data_catalogue_new"),
            reverse("data_catalogue_fetch", args=["ds_demo", 1]),
            reverse("data_catalogue_incident", args=["ds_demo", 1]),
            reverse("data_catalogue_derive_labels", args=["ds_demo", 1]),
            reverse("data_catalogue_partition", args=["ds_demo", 1]),
            reverse("data_catalogue_new_version", args=["ds_demo"]),
        ):
            with self.subTest(url=url):
                self.assertEqual(csrf_client.post(url, {}).status_code, 403)

    def test_existing_ui_route_still_resolves(self):
        self.assertEqual(resolve(reverse("monitoring_home")).url_name, "monitoring_home")


class DataCatalogueClientTests(SimpleTestCase):
    def test_client_maps_catalogue_contracts_to_server_side_url(self):
        response = Mock(status_code=200)
        response.json.return_value = {"ok": True}
        transport = Mock()
        transport.request.return_value = response
        client = DataCatalogueClient("http://catalogue.internal", transport=transport)

        client.list_datasets({"page": 2})
        client.get_dataset("ds_demo")
        client.get_version("ds_demo", 3)
        client.create_dataset({"display_name": "Demo"})
        client.update_metadata("ds_demo", {"description": "Updated"})
        client.create_version("ds_demo", {"copy_from_version": 3})
        client.start_fetch("ds_demo", 3)
        client.fetch_status("ds_demo", 3)
        client.preview("ds_demo", 3, 10)
        client.add_incident("ds_demo", 3, {"scenario": "CPU stress"})
        client.derive_labels("ds_demo", 3)
        client.create_partition("ds_demo", 3, {"mode": "none"})

        calls = [(call.args[0], call.args[1]) for call in transport.request.call_args_list]
        self.assertIn(("GET", "http://catalogue.internal/datasets"), calls)
        self.assertIn(("GET", "http://catalogue.internal/datasets/ds_demo/versions/3"), calls)
        self.assertIn(("PATCH", "http://catalogue.internal/datasets/ds_demo"), calls)
        self.assertIn(("POST", "http://catalogue.internal/datasets/ds_demo/versions/3/partitions"), calls)

    def test_client_redacts_internal_url_from_backend_error(self):
        response = Mock(status_code=503)
        response.json.return_value = {"error": "failed via http://prometheus.secret:9090/query"}
        transport = Mock()
        transport.request.return_value = response
        client = DataCatalogueClient("http://catalogue.internal", transport=transport)
        with self.assertRaises(CatalogueClientError) as caught:
            client.list_datasets({})
        self.assertNotIn("prometheus.secret", str(caught.exception))
        self.assertTrue(caught.exception.retryable)
