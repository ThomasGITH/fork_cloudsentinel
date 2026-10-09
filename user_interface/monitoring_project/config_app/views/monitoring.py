"""Existing monitoring screens backed by generic Saved Model inference."""

from __future__ import annotations

import requests
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

from ..forms import MonitoringForm
from ..learning_client import LearningClientError, get_learning_client


def _models():
    payload = get_learning_client().list_models(
        {"page": 1, "page_size": 100, "status": "available", "sort": "newest"}
    )
    return [item for item in payload.get("items", []) if isinstance(item, dict)]


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
            f"{item.get('detector_id', 'Detector')} v{item.get('detector_version', '—')} — {item['model_id']}",
        )
        for item in items
        if _ready(item)
    ]


def monitoring_home(request):
    return render(request, "config_app/monitoring/monitoring_home.html")


def monitoring(request):
    try:
        models = _models()
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
                        "model_id": selected["model_id"],
                        "window_seconds": form.cleaned_data["window_minutes"] * 60,
                        "poll_interval_seconds": form.cleaned_data["poll_interval_seconds"],
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
