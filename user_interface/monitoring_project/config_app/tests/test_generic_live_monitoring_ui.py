from unittest.mock import Mock, patch

from django.test import SimpleTestCase
from django.test import Client
from django.urls import reverse

from config_app.workflow_context import decode_context, encode_context, new_context


def model(model_id="model-ready", ready=True):
    return {
        "model_id": model_id,
        "detector_id": "isolation-forest",
        "detector_version": "1.0.0",
        "status": "available",
        "inference": {
            "status": "ready",
            "contract": "cloudsentinel.inference/v1",
        },
        "live_monitoring": (
            {"status": "ready", "sampling_interval_seconds": 60}
            if ready
            else {"status": "not_ready", "reason": "No pinned recipe"}
        ),
    }


class GenericLiveMonitoringUiTests(SimpleTestCase):
    def setUp(self):
        self.learning = Mock()
        self.learning.list_models.return_value = {
            "items": [model(), model("model-old", False)],
            "page": 1,
            "pages": 1,
        }
        self.patch = patch(
            "config_app.views.monitoring.get_learning_client",
            return_value=self.learning,
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.readiness_patch = patch("config_app.views.monitoring.requests.get")
        self.readiness_get = self.readiness_patch.start()
        self.readiness_get.return_value.status_code = 200
        self.readiness_get.return_value.json.return_value = {
            "status": "ready",
            "reason": None,
            "input_contract": "cloudsentinel.crca-input/v1",
            "default_method": "cloudsentinel.causal-rca/legacy-v1",
            "methods": [
                {
                    "id": "cloudsentinel.causal-rca/legacy-v1",
                    "display_name": "CausalRCA",
                    "status": "available",
                },
                {
                    "id": "cloudsentinel.baro-robust-scorer/v1",
                    "display_name": "BARO robust scorer",
                    "status": "unavailable",
                },
            ],
        }
        self.addCleanup(self.readiness_patch.stop)

    def test_existing_monitoring_page_uses_saved_models_and_prefill(self):
        response = self.client.get(reverse("monitoring"), {"model_id": "model-ready"})
        self.assertContains(response, "Saved model artefact")
        self.assertContains(response, "model-ready")
        self.assertContains(response, "Live-ready")
        self.assertContains(response, "Offline only")
        self.assertNotContains(response, "Containers")

    def test_guided_flow_keeps_selected_model_visible(self):
        flow = encode_context(
            new_context(
                dataset_id="ds-eval",
                dataset_version=2,
                comparison_id="comparison-one",
            )
        )
        response = self.client.get(
            reverse("monitoring"), {"model_id": "model-ready", "flow": flow}
        )

        self.assertContains(response, "Guided experiment")
        self.assertEqual(response.context["form"]["model_id"].value(), "model-ready")
        carried = decode_context(response.context["workflow_token"])
        self.assertEqual(carried["model_id"], "model-ready")

    def test_model_select_uses_readable_label_without_repeating_model_id(self):
        model_id = "run_1ddefa4845484057ba7079946735c8db-local-outlier-factor-2426c165"
        item = model(model_id)
        item.update({
            "detector_id": "local-outlier-factor",
            "detector_version": "1.0.1",
            "created_at": "2026-10-12T11:32:00Z",
        })
        self.learning.list_models.return_value = {"items": [item], "page": 1, "pages": 1}

        response = self.client.get(reverse("monitoring"), {"model_id": model_id})

        choices = dict(response.context["form"].fields["model_id"].choices)
        self.assertEqual(
            choices[model_id],
            "Local Outlier Factor v1.0.1 · 12 Oct 2026, 11:32 UTC",
        )
        self.assertNotIn(model_id, choices[model_id])
        self.assertEqual(response.context["form"]["model_id"].value(), model_id)

    @patch("config_app.views.monitoring.requests.post")
    def test_start_sends_only_explicit_model_and_safe_timing(self, post):
        post.return_value.status_code = 202
        post.return_value.json.return_value = {
            "status": "monitoring_started",
            "session": {"session_id": "monitor-one"},
        }
        response = self.client.post(
            reverse("monitoring"),
            {
                "display_name": "Production checkout",
                "model_id": "model-ready",
                "window_minutes": 10,
                "poll_interval_seconds": 30,
            },
        )
        self.assertEqual(response.status_code, 202)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model_id"], "model-ready")
        self.assertEqual(payload["display_name"], "Production checkout")
        self.assertEqual(payload["window_seconds"], 600)
        self.assertNotIn("detector_id", payload)
        self.assertNotIn("prometheus_url", payload)
        self.assertFalse(payload["automatic_rca"]["enabled"])
        self.assertEqual(
            payload["automatic_rca"]["method"],
            "cloudsentinel.causal-rca/legacy-v1",
        )
        self.assertFalse(payload["automatic_rca"]["reanalyze_within_episode"])
        self.assertEqual(payload["automatic_rca"]["reanalysis_min_interval_seconds"], 900)
        self.assertEqual(payload["automatic_rca"]["reanalysis_min_new_observations"], 20)
        self.assertEqual(payload["automatic_rca"]["max_attempts_per_episode"], 3)

    @patch("config_app.views.monitoring.requests.post")
    def test_started_session_is_added_to_guided_context(self, post):
        post.return_value.status_code = 202
        post.return_value.json.return_value = {
            "status": "monitoring_started",
            "session": {"session_id": "monitor-one"},
        }
        flow = encode_context(new_context(comparison_id="comparison-one"))

        response = self.client.post(
            f'{reverse("monitoring")}?flow={flow}',
            {
                "display_name": "Production checkout",
                "model_id": "model-ready",
                "window_minutes": 10,
                "poll_interval_seconds": 30,
            },
        )

        carried = decode_context(response.json()["workflow_token"])
        self.assertEqual(carried["model_id"], "model-ready")
        self.assertEqual(carried["monitoring_session_id"], "monitor-one")

    @patch("config_app.views.monitoring.requests.post")
    def test_start_sends_explicit_rca_configuration(self, post):
        post.return_value.status_code = 202
        post.return_value.json.return_value = {"status": "monitoring_started", "session": {"session_id": "monitor-one"}}
        response = self.client.post(
            reverse("monitoring"),
            {
                "model_id": "model-ready", "window_minutes": 10,
                "poll_interval_seconds": 30, "automatic_rca_enabled": "on",
                "rca_threshold_percentage": 75, "rca_analysis_window_minutes": 20,
                "rca_cooldown_minutes": 90, "rca_reanalysis_enabled": "on",
                "rca_reanalysis_min_interval_seconds": 1800,
                "rca_reanalysis_min_new_observations": 40,
                "rca_max_attempts_per_episode": 4,
            },
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(
            post.call_args.kwargs["json"]["automatic_rca"],
            {
                "enabled": True,
                "method": "cloudsentinel.causal-rca/legacy-v1",
                "threshold_percentage": 75.0,
                "analysis_window_seconds": 1200,
                "cooldown_seconds": 5400,
                "reanalyze_within_episode": True,
                "reanalysis_min_interval_seconds": 1800,
                "reanalysis_min_new_observations": 40,
                "max_attempts_per_episode": 4,
            },
        )

    def test_setup_explains_reanalysis_is_bounded_and_disabled_by_default(self):
        response = self.client.get(reverse("monitoring"))
        self.assertContains(response, "RCA method")
        self.assertContains(response, "CausalRCA")
        self.assertContains(response, "BARO RobustScorer")
        self.assertContains(response, "Dependency-oriented analysis")
        self.assertContains(response, "detector-derived anomalous segment")
        self.assertContains(response, "independent of the selected anomaly-detection model")
        self.assertContains(response, "Unavailable")
        self.assertNotIn(
            "cloudsentinel.baro-robust-scorer/v1",
            dict(response.context["form"].fields["rca_method"].choices),
        )
        self.assertContains(response, "Allow bounded RCA reanalysis within the same episode")
        self.assertContains(response, "Technical default: 900 seconds")

    def test_baro_is_selectable_only_when_backend_reports_available(self):
        self.readiness_get.return_value.json.return_value["methods"][1]["status"] = "available"
        response = self.client.get(reverse("monitoring"))
        self.assertContains(response, "BARO RobustScorer")
        self.assertIn(
            "cloudsentinel.baro-robust-scorer/v1",
            dict(response.context["form"].fields["rca_method"].choices),
        )

    @patch("config_app.views.monitoring.requests.post")
    def test_offline_only_model_cannot_start(self, post):
        response = self.client.post(
            reverse("monitoring"),
            {
                "model_id": "model-old",
                "window_minutes": 10,
                "poll_interval_seconds": 30,
            },
        )
        self.assertEqual(response.status_code, 200)
        post.assert_not_called()

    def test_dashboard_exposes_warmup_and_safe_status_language(self):
        response = self.client.get(reverse("monitoring_overview"))
        self.assertContains(response, "Warm-up observations")
        self.assertContains(response, "Buffered observations")
        self.assertContains(response, "Monitoring History")
        self.assertContains(response, "monitoring_rca_presentation.js")
        self.assertContains(response, "monitoring_history.js")
        self.assertNotContains(response, "API_CGNN_ANOMALY_DETECTION_URL")

        setup = self.client.get(reverse("monitoring"))
        self.assertContains(setup, "payload.session.session_id")

    @patch("config_app.views.task_results.requests.get")
    def test_paginated_history_proxy(self, get):
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.return_value = {
            "items": [{"session_id": "monitor-one", "status": "stopped"}],
            "page": 2, "page_size": 20, "total": 21, "pages": 2,
        }
        response = self.client.get(reverse("monitoring_sessions"), {"page": 2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["page"], 2)
        self.assertEqual(get.call_args.kwargs["params"], {"page": "2"})

    @patch("config_app.views.task_results.requests.get")
    def test_session_detail_proxy_exposes_safe_chart_payload(self, get):
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.return_value = {
            "session": {"session_id": "monitor-one", "status": "active"},
            "cycles": {"items": [{"cycle_id": "cycle-one", "anomaly_percentage": 5.0}], "page": 1, "pages": 1},
            "provenance": {"recipe_sha256": "r" * 64},
        }
        response = self.client.get(reverse("monitoring_session_detail", args=["monitor-one"]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["cycles"]["items"][0]["anomaly_percentage"], 5.0)

    @patch("config_app.views.task_results.requests.get")
    def test_rca_detail_proxy_returns_stored_safe_result(self, get):
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.return_value = {
            "run": {"run_id": "rca-one", "status": "completed"},
            "result": {"ranking": [{"feature_name": "cpu", "score": 0.8}], "graphs": []},
        }
        response = self.client.get(reverse("monitoring_rca_run_detail", args=["monitor-one", "rca-one"]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run"]["status"], "completed")

    @patch("config_app.views.task_results.requests.delete")
    def test_stop_route_requires_csrf(self, delete):
        client = Client(enforce_csrf_checks=True)
        response = client.post(reverse("stop_task", args=["monitor-one"]))
        self.assertEqual(response.status_code, 403)
        delete.assert_not_called()

    @patch("config_app.views.task_results.requests.get")
    def test_active_session_proxy_accepts_json_array(self, get):
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.return_value = [
            {"session_id": "monitor-one", "model_id": "model-ready", "status": "active"}
        ]

        response = self.client.get(reverse("fetch_active_tasks"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["session_id"], "monitor-one")
        self.assertEqual(get.call_args.kwargs["timeout"], (5, 15))

    @patch("config_app.views.task_results.requests.get")
    def test_active_session_proxy_translates_invalid_json_safely(self, get):
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.side_effect = ValueError("private upstream response")

        response = self.client.get(reverse("fetch_active_tasks"))

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["message"],
            "Monitoring status is temporarily unavailable.",
        )
        self.assertNotContains(response, "private upstream response", status_code=502)
