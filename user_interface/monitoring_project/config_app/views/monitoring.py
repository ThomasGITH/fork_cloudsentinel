"""Existing monitoring screens backed by generic Saved Model inference."""

from __future__ import annotations

import requests
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

from ..forms import MonitoringForm
from ..comparison_labels import readable_model_name
from ..learning_client import LearningClientError, get_learning_client


def _models():
    payload = get_learning_client().list_models(
        {"page": 1, "page_size": 100, "status": "available", "sort": "newest"}
    )
    items = [item for item in payload.get("items", []) if isinstance(item, dict)]
    for item in items:
        item["display_name"] = readable_model_name(item)
    return items


def _add_rca_readiness(items):
    base = settings.API_DATA_INGESTION_URL.rstrip('/')
    for item in items:
        if not _ready(item):
            item["rca_readiness"] = {"status": "incompatible", "reason": "A live-ready input recipe is required."}
            continue
        try:
            response = requests.get(
                f"{base}/monitoring-rca/readiness/{item['model_id']}", timeout=(5, 15)
            )
            payload = response.json()
            item["rca_readiness"] = payload if response.status_code == 200 and isinstance(payload, dict) else {"status": "unavailable", "reason": "RCA readiness could not be verified."}
        except (requests.RequestException, ValueError):
            item["rca_readiness"] = {"status": "unavailable", "reason": "RCA readiness could not be verified."}
    return items


def _ready(item):
    return (
        item.get("inference", {}).get("status") == "ready"
        and item.get("inference", {}).get("contract") == "cloudsentinel.inference/v1"
        and item.get("live_monitoring", {}).get("status") == "ready"
    )


def _choices(items):
    return [
        (
            item["model_id"],
            f"{item['display_name']} — {item['model_id']}",
        )
        for item in items
        if _ready(item)
    ]


def monitoring_home(request):
    return render(request, "config_app/monitoring/monitoring_home.html")


def monitoring(request):
    try:
        models = _add_rca_readiness(_models())
        upstream_error = None
    except LearningClientError:
        models = []
        upstream_error = "Saved Models are temporarily unavailable."
    form = MonitoringForm(
        request.POST or None,
        initial={"model_id": request.GET.get("model_id", "")},
    )
    form.fields["model_id"].choices = _choices(models)
    if request.method == "POST" and form.is_valid():
        selected = next(
            (item for item in models if item.get("model_id") == form.cleaned_data["model_id"]),
            None,
        )
        if selected is None or not _ready(selected):
            form.add_error("model_id", "This model is not ready for live monitoring.")
        else:
            try:
                response = requests.post(
                    f"{settings.API_DATA_INGESTION_URL.rstrip('/')}/start_monitoring",
                    json={
                        "display_name": form.cleaned_data["display_name"],
                        "model_id": selected["model_id"],
                        "window_seconds": form.cleaned_data["window_minutes"] * 60,
                        "poll_interval_seconds": form.cleaned_data["poll_interval_seconds"],
                        "automatic_rca": {
                            "enabled": form.cleaned_data["automatic_rca_enabled"],
                            "threshold_percentage": form.cleaned_data["rca_threshold_percentage"] if form.cleaned_data["rca_threshold_percentage"] is not None else 50,
                            "analysis_window_seconds": (form.cleaned_data["rca_analysis_window_minutes"] if form.cleaned_data["rca_analysis_window_minutes"] is not None else 10) * 60,
                            "cooldown_seconds": (form.cleaned_data["rca_cooldown_minutes"] if form.cleaned_data["rca_cooldown_minutes"] is not None else 60) * 60,
                            "reanalyze_within_episode": form.cleaned_data["rca_reanalysis_enabled"],
                            "reanalysis_min_interval_seconds": form.cleaned_data["rca_reanalysis_min_interval_seconds"] if form.cleaned_data["rca_reanalysis_min_interval_seconds"] is not None else 900,
                            "reanalysis_min_new_observations": form.cleaned_data["rca_reanalysis_min_new_observations"] if form.cleaned_data["rca_reanalysis_min_new_observations"] is not None else 20,
                            "max_attempts_per_episode": form.cleaned_data["rca_max_attempts_per_episode"] if form.cleaned_data["rca_max_attempts_per_episode"] is not None else 3,
                        },
                    },
                    timeout=(5, 30),
                )
                payload = response.json()
            except (requests.RequestException, ValueError):
                return JsonResponse(
                    {"status": "error", "message": "Live monitoring could not be started."},
                    status=502,
                )
            if response.status_code >= 400:
                return JsonResponse(
                    {"status": "error", "message": str(payload.get("error") or "The model could not be started.")[:500]},
                    status=response.status_code,
                )
            return JsonResponse(payload, status=202)
    return render(
        request,
        "config_app/monitoring/monitoring_setup.html",
        {"form": form, "models": models, "upstream_error": upstream_error, "ready_count": len(_choices(models))},
    )


def monitoring_overview(request):
    return render(request, "config_app/monitoring/monitoring_dashboard.html")


def load_kube_config():
    return JsonResponse({"status": "not_required"})


def task_manager(request):
    return render(request, "config_app/monitoring/monitoring_dashboard.html")
