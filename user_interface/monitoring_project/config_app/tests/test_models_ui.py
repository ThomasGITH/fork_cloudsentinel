from unittest.mock import Mock, patch

from django.test import Client, SimpleTestCase
from django.urls import resolve, reverse

from config_app.learning_client import LearningAdaptationClient, LearningClientError


def detector(detector_id="isolation-forest"):
    return {
        "schema_version": 1,
        "id": detector_id,
        "name": "Isolation Forest" if detector_id == "isolation-forest" else "CGNN",
        "version": "1.0.0",
        "description": "Metrics anomaly detector",
        "supported_modalities": ["metrics"],
        "capabilities": {"training": {"enabled": True, "protocol": "cloudsentinel.training/v1", "input_profile": "metrics-partition-v1"}},
        "input_requirements": {"training": {"format": "numeric matrix"}},
        "training_parameters": {
            "n_estimators": {"type": "integer", "default": 100, "minimum": 1, "description": "Tree count"},
            "advanced_flag": {"type": "boolean", "default": False, "advanced": True, "description": "Advanced option"},
        },
        "entry_point": "secret.runtime:Adapter",
    }


def ready_version():
    return {
        "dataset_id": "ds_demo", "version": 3, "status": "available",
        "workload_context": {"workload_intensity": "Normal", "dominant_workload_characteristic": "Mixed", "anomaly_scenario": "CPU stress"},
        "technical_schema": {"feature_count": 2, "observation_count": 30, "feature_order": ["cpu", "memory"]},
        "ground_truth": {"labels_available": True, "active_label_source": "incident-derived"},
        "partition": {
            "mode": "time_range", "partition_id": "partition_demo", "checksum": "partition-sha",
            "train": {"observation_count": 20}, "test": {"observation_count": 10},
            "labels_source": "incident-derived", "labeled_evaluation": True,
            "feature_order_reference": {"sha256": "feature-sha"},
        },
        "usages": {"labeled_evaluation": {"supported": True, "reasons": []}},
    }


def dataset_detail():
    return {"dataset_id": "ds_demo", "display_name": "Checkout metrics", "purpose": ["training"], "latest_version": 3}


def dataset_list():
    return {"items": [{"dataset_id": "ds_demo", "display_name": "Checkout metrics", "version": 3, "status": "available", "purpose": ["training"], "feature_count": 2, "observation_count": 30, "partition_mode": "time_range", "labels_available": True, "usages": {"labeled_evaluation": {"supported": True, "reasons": []}}}], "page": 1, "page_size": 100, "total": 1, "pages": 1}


def run_payload(status="partial_success"):
    return {
        "run_id": "run_demo", "run_name": "Baseline", "status": status,
        "created_at": "2026-09-29T10:00:00Z", "started_at": "2026-09-29T10:01:00Z", "updated_at": "2026-09-29T10:02:00Z", "completed_at": None,
        "dataset": {"source": "catalogue", "dataset_id": "ds_demo", "version": 3, "partition_id": "partition_demo", "partition_checksum": "partition-sha", "feature_order_sha256": "feature-sha", "label_source": "incident-derived"},
        "snapshot_id": "snapshot_demo", "snapshot_sha256": "snapshot-sha",
        "children": [
            {"detector_id": "isolation-forest", "model_id": "model_demo", "status": "completed", "model_status": "available", "parameters": {"n_estimators": 100}, "task_id": "celery-secret", "result_metadata": {"promotion": {"status": "promoted"}}},
            {"detector_id": "cgnn", "model_id": "model_cgnn", "status": "validation_failed", "validation_error": "Insufficient observations", "parameters": {}},
        ],
    }


def model_payload():
    return {
        "model_id": "model_demo", "detector_id": "isolation-forest", "detector_version": "1.0.0", "status": "available", "created_at": "2026-09-29T10:00:00Z", "updated_at": "2026-09-29T10:02:00Z", "training_run_id": "run_demo",
        "dataset": {"source": "catalogue", "dataset_id": "ds_demo", "version": 3, "partition_id": "partition_demo", "partition_checksum": "partition-sha"},
        "snapshot": {"snapshot_id": "snapshot_demo", "snapshot_sha256": "snapshot-sha"},
        "feature_identity": {"feature_order": ["cpu", "memory"], "feature_order_sha256": "feature-sha"},
        "training_parameters": {"n_estimators": 100}, "evaluation": {"f1": 0.9}, "promotion": {"status": "promoted", "promoted_at": "2026-09-29T10:02:00Z"},
        "artifact": {"safe_reference": "/app/private/model.joblib"}, "internal_url": "http://learning.internal/model",
    }


class ModelsUiTests(SimpleTestCase):
    def setUp(self):
        self.learning = Mock()
        self.learning.list_detectors.return_value = {"detectors": [detector(), detector("cgnn")], "discovery_errors": []}
        self.learning.get_detector.side_effect = lambda value: detector(value)
        self.learning.list_training_runs.return_value = {"items": [run_payload()], "page": 1, "page_size": 20, "total": 1, "pages": 1}
        self.learning.get_training_run.return_value = run_payload()
        self.learning.list_models.return_value = {"items": [model_payload()], "page": 1, "page_size": 20, "total": 1, "pages": 1}
        self.learning.get_model.return_value = model_payload()
        self.learning.create_training_run.return_value = {"run_id": "run_created", "status": "queued"}
        self.catalogue = Mock()
        self.catalogue.list_datasets.return_value = dataset_list()
        self.catalogue.get_dataset.return_value = dataset_detail()
        self.catalogue.get_version.return_value = ready_version()
        patches = [
            patch("config_app.views.models_ui.get_learning_client", return_value=self.learning),
            patch("config_app.views.models_ui.get_catalogue_client", return_value=self.catalogue),
        ]
        for item in patches:
            self.addCleanup(item.stop); item.start()

    def test_navigation_routes_and_active_tab(self):
        response = self.client.get(reverse("models_overview"))
        self.assertContains(response, "Detector Library")
        self.assertContains(response, 'href="/models/"', html=False)
        self.assertContains(response, 'aria-current="page"', html=False)
        self.assertEqual(resolve(reverse("models_train")).url_name, "models_train")
        self.assertContains(self.client.get(reverse("home")), reverse("models_overview"))
        self.assertEqual(resolve(reverse("train_algorithms")).url_name, "train_algorithms")

    def test_detector_list_detail_and_discovery_errors(self):
        self.learning.list_detectors.return_value["discovery_errors"] = [{"plugin": "broken", "error": "invalid manifest at /app/secret", "entry_point": "bad:Code"}]
        response = self.client.get(reverse("models_overview"))
        self.assertContains(response, "Isolation Forest")
        self.assertContains(response, "broken")
        self.assertNotContains(response, "bad:Code")
        self.assertNotContains(response, "/app/secret")
        detail = self.client.get(reverse("models_detector_detail", args=["isolation-forest"]))
        self.assertContains(detail, "n_estimators")
        self.assertContains(detail, "Advanced parameters")
        self.assertNotContains(detail, "secret.runtime")

    def test_saved_models_and_history_filters_pagination(self):
        saved = self.client.get(reverse("models_overview"), {"tab": "saved", "detector_id": "isolation-forest", "page": 2, "sort": "oldest"})
        self.assertContains(saved, "model_demo")
        self.assertContains(saved, "Checkout metrics v3")
        self.learning.list_models.assert_called_once_with({"detector_id": "isolation-forest", "page": "2", "sort": "oldest"})
        history = self.client.get(reverse("models_overview"), {"tab": "history", "status": "partial_success", "page_size": 10})
        self.assertContains(history, "Partial success", count=None)
        self.assertContains(history, "validation_failed")
        self.assertContains(history, "Checkout metrics v3")
        self.learning.list_training_runs.assert_called_once_with({"status": "partial_success", "page_size": "10"})

    def test_model_and_run_details_hide_runtime_internals(self):
        model = self.client.get(reverse("models_saved_detail", args=["model_demo"]))
        self.assertContains(model, "partition-sha")
        self.assertNotContains(model, "/app/private")
        self.assertNotContains(model, "learning.internal")
        run = self.client.get(reverse("models_training_run_detail", args=["run_demo"]))
        self.assertContains(run, "validation_failed")
        self.assertNotContains(run, "celery-secret")

    def test_safe_upstream_error_translation(self):
        self.learning.list_models.side_effect = LearningClientError("http://secret.internal failed at /app/models", retryable=True)
        response = self.client.get(reverse("models_overview"), {"tab": "saved"})
        self.assertContains(response, "Retry")
        self.assertNotContains(response, "secret.internal")
        self.assertNotContains(response, "/app/models")

    def test_wizard_gates_unlabelled_partition(self):
        invalid = ready_version(); invalid["ground_truth"] = {"labels_available": False}; invalid["partition"]["labels_source"] = None; invalid["partition"]["labeled_evaluation"] = False; invalid["usages"]["labeled_evaluation"] = {"supported": False, "reasons": ["No labels"]}
        self.catalogue.get_version.return_value = invalid
        first = self.client.get(reverse("models_train"))
        response = self.client.post(reverse("models_train"), {"step": 1, "wizard_state": first.context["state_token"], "dataset_id": "ds_demo", "version": 3})
        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "not ready", status_code=400)
        self.assertContains(response, "No labels", status_code=400)
        self.learning.create_training_run.assert_not_called()

    def test_wizard_explains_and_rejects_incompatible_detector(self):
        incompatible = detector("future-detector")
        incompatible["capabilities"]["training"]["input_profile"] = "metrics-partition-v2"
        self.learning.list_detectors.return_value = {"detectors": [detector(), incompatible], "discovery_errors": []}
        first = self.client.get(reverse("models_train"))
        step2 = self.client.post(reverse("models_train"), {"step": 1, "wizard_state": first.context["state_token"], "dataset_id": "ds_demo", "version": 3})
        self.assertContains(step2, "does not accept the selected metrics partition profile")
        rejected = self.client.post(reverse("models_train"), {"step": 2, "wizard_state": step2.context["state_token"], "detectors": ["future-detector"]})
        self.assertEqual(rejected.status_code, 400)
        self.learning.create_training_run.assert_not_called()

    def _to_step(self, target):
        response = self.client.get(reverse("models_train"), {"detector": "isolation-forest", "dataset": "ds_demo", "version": 3})
        if target == 1: return response
        response = self.client.post(reverse("models_train"), {"step": 1, "wizard_state": response.context["state_token"], "dataset_id": "ds_demo", "version": 3})
        if target == 2: return response
        response = self.client.post(reverse("models_train"), {"step": 2, "wizard_state": response.context["state_token"], "detectors": ["isolation-forest"]})
        if target == 3: return response
        response = self.client.post(reverse("models_train"), {"step": 3, "wizard_state": response.context["state_token"], "param__isolation-forest__n_estimators": "55", "param__isolation-forest__advanced_flag": "true", "param__isolation-forest__unknown": "leak"})
        return response

    def test_wizard_preselection_dynamic_parameters_and_payload(self):
        step2 = self._to_step(2)
        self.assertContains(step2, 'value="isolation-forest" checked', html=False)
        step3 = self.client.post(reverse("models_train"), {"step": 2, "wizard_state": step2.context["state_token"], "detectors": ["isolation-forest"]})
        self.assertContains(step3, "n_estimators")
        review = self.client.post(reverse("models_train"), {"step": 3, "wizard_state": step3.context["state_token"], "param__isolation-forest__n_estimators": "55", "param__isolation-forest__advanced_flag": "true", "param__isolation-forest__unknown": "leak"})
        self.assertContains(review, "partition-sha")
        result = self.client.post(reverse("models_train"), {"step": 4, "wizard_state": review.context["state_token"], "run_name": "Baseline"})
        self.assertRedirects(result, "/models/training-runs/run_created/", fetch_redirect_response=False)
        payload = self.learning.create_training_run.call_args.args[0]
        self.assertEqual(payload["dataset"], {"source": "catalogue", "dataset_id": "ds_demo", "version": 3, "partition_id": "partition_demo"})
        self.assertEqual(payload["detectors"][0]["parameters"], {"n_estimators": 55, "advanced_flag": True})
        self.assertNotIn("unknown", payload["detectors"][0]["parameters"])
        self.assertEqual(payload["run_name"], "Baseline")

    def test_duplicate_submit_is_prevented(self):
        review = self._to_step(4)
        post = {"step": 4, "wizard_state": review.context["state_token"]}
        self.client.post(reverse("models_train"), post)
        second = self.client.post(reverse("models_train"), post)
        self.assertEqual(second.status_code, 400)
        self.assertEqual(self.learning.create_training_run.call_count, 1)

    def test_run_polling_proxy_is_safe(self):
        response = self.client.get(reverse("models_training_run_status", args=["run_demo"]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "partial_success")
        self.assertNotIn("task_id", response.content.decode())
        self.assertContains(self.client.get(reverse("models_training_run_detail", args=["run_demo"])), "models_run.js")

    def test_structured_cgnn_progress_is_rendered_as_epoch_feedback(self):
        running = run_payload("running")
        running["children"] = [{
            "detector_id": "cgnn",
            "model_id": "model_cgnn",
            "status": "running",
            "progress_phase": "TRAINING",
            "detail": {"values": [1, 10, 3, 8]},
            "parameters": {},
        }]
        self.learning.get_training_run.return_value = running

        page = self.client.get(reverse("models_training_run_detail", args=["run_demo"]))
        self.assertContains(page, "Training — Epoch 2 of 10 · batch 4 of 8")
        self.assertNotContains(page, "[object Object]")

        status = self.client.get(reverse("models_training_run_status", args=["run_demo"]))
        self.assertEqual(
            status.json()["children"][0]["display_detail"],
            "Training — Epoch 2 of 10 · batch 4 of 8",
        )

    def test_mutating_route_requires_csrf(self):
        csrf_client = Client(enforce_csrf_checks=True)
        self.assertEqual(csrf_client.post(reverse("models_train"), {}).status_code, 403)


class LearningClientTests(SimpleTestCase):
    def test_contract_mapping_and_detector_detail_fallback(self):
        response = Mock(status_code=200, headers={})
        response.json.return_value = {"detectors": [detector()]}
        transport = Mock(); transport.request.return_value = response
        client = LearningAdaptationClient("http://learning.internal", transport=transport)
        self.assertEqual(client.get_detector("isolation-forest")["id"], "isolation-forest")
        client.list_training_runs({"page": 2}); client.get_training_run("run_demo")
        client.create_training_run({"dataset": {}}); client.list_models({"status": "available"}); client.get_model("model_demo")
        paths = [call.args[1] for call in transport.request.call_args_list]
        self.assertEqual(paths, ["http://learning.internal/detectors", "http://learning.internal/training_runs", "http://learning.internal/training_runs/run_demo", "http://learning.internal/training_runs", "http://learning.internal/models", "http://learning.internal/models/model_demo"])

    def test_client_translates_transport_and_upstream_errors(self):
        transport = Mock(); transport.request.side_effect = __import__("requests").Timeout()
        with self.assertRaises(LearningClientError) as caught:
            LearningAdaptationClient("http://internal", transport=transport).list_models({})
        self.assertTrue(caught.exception.retryable)
        response = Mock(status_code=500, headers={}); response.json.return_value = {"error": "failure at http://secret.internal and /app/private"}; transport.request.side_effect = None; transport.request.return_value = response
        with self.assertRaises(LearningClientError) as caught:
            LearningAdaptationClient("http://internal", transport=transport).list_models({})
        self.assertNotIn("secret.internal", str(caught.exception)); self.assertNotIn("/app/private", str(caught.exception))
