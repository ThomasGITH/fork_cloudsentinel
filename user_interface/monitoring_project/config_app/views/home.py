"""Guided entry point for the CloudSentinel experimentation workflow."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.http import require_GET

from ..catalogue_client import CatalogueClientError, DataCatalogueClient
from ..learning_client import LearningAdaptationClient, LearningClientError


_TIMEOUT = (1.5, 4.0)


def _total(payload: dict[str, Any]) -> int:
    """Return a safe total for paginated service responses."""
    value = payload.get("total")
    if isinstance(value, int) and value >= 0:
        return value
    items = payload.get("items")
    return len(items) if isinstance(items, list) else 0


def _prometheus_status() -> dict[str, Any]:
    try:
        payload = DataCatalogueClient(timeout=_TIMEOUT).test_monitored_environment()
        target_count = payload.get("target_count", 0)
        return {
            "state": "ready",
            "label": "Connected",
            "metric": f"{target_count} active series" if target_count else "Source responded",
            "detail": "The configured catalogue source accepted a live query.",
        }
    except CatalogueClientError:
        return {
            "state": "unavailable",
            "label": "Needs attention",
            "metric": "No response",
            "detail": "CloudSentinel could not verify the configured Prometheus source.",
        }


def _catalogue_status() -> dict[str, Any]:
    try:
        client = DataCatalogueClient(timeout=_TIMEOUT)
        count = _total(client.list_datasets({"page": 1, "page_size": 1}))
        return {
            "state": "ready" if count else "action",
            "label": "Ready" if count else "Next action",
            "metric": f"{count} dataset{'s' if count != 1 else ''}",
            "detail": (
                "Datasets are available for training and evaluation."
                if count
                else "Capture or register a metrics dataset before training a detector."
            ),
            "count": count,
        }
    except CatalogueClientError:
        return {
            "state": "unavailable",
            "label": "Unavailable",
            "metric": "Status unknown",
            "detail": "The Data Catalogue could not be reached.",
            "count": None,
        }


def _learning_status() -> dict[str, Any]:
    try:
        client = LearningAdaptationClient(connect_timeout=_TIMEOUT[0], read_timeout=_TIMEOUT[1])
        models = client.list_models(
            {"page": 1, "page_size": 100, "status": "available", "sort": "newest"}
        )
        comparisons = client.list_comparisons(
            {"page": 1, "page_size": 1, "status": "completed"}
        )
        model_items = models.get("items") if isinstance(models.get("items"), list) else []
        model_count = _total(models)
        comparison_count = _total(comparisons)
        live_ready_count = sum(
            1
            for model in model_items
            if isinstance(model, dict)
            and (model.get("inference") or {}).get("status") == "ready"
            and (model.get("live_monitoring") or {}).get("status") == "ready"
        )
        return {
            "models": {
                "state": "ready" if model_count else "action",
                "label": "Ready" if model_count else "Next action",
                "metric": f"{model_count} saved model{'s' if model_count != 1 else ''}",
                "detail": (
                    "Saved models are available for comparison."
                    if model_count
                    else "Train one or more detectors on a catalogue dataset."
                ),
                "count": model_count,
                "live_ready_count": live_ready_count,
            },
            "comparisons": {
                "state": "ready" if comparison_count else "action",
                "label": "Ready" if comparison_count else "Next action",
                "metric": f"{comparison_count} completed comparison{'s' if comparison_count != 1 else ''}",
                "detail": (
                    "Completed evidence is available across selected workload contexts."
                    if comparison_count
                    else "Compare compatible saved models on the same evaluation dataset."
                ),
                "count": comparison_count,
            },
        }
    except LearningClientError:
        unavailable = {
            "state": "unavailable",
            "label": "Unavailable",
            "metric": "Status unknown",
            "detail": "The learning service could not be reached.",
            "count": None,
        }
        return {"models": {**unavailable, "live_ready_count": None}, "comparisons": unavailable}


def _monitoring_status() -> dict[str, Any]:
    try:
        response = requests.get(
            f"{settings.API_DATA_INGESTION_URL.rstrip('/')}/monitoring-sessions",
            params={"page": 1, "page_size": 1},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ValueError("invalid monitoring response")
        count = _total(payload)
        return {
            "state": "ready" if count else "action",
            "label": "History available" if count else "Not started",
            "metric": f"{count} monitoring session{'s' if count != 1 else ''}",
            "detail": (
                "Open monitoring history to inspect detections and RCA results."
                if count
                else "Start live monitoring after an inference-ready model is available."
            ),
            "count": count,
        }
    except (requests.RequestException, ValueError, TypeError):
        return {
            "state": "unavailable",
            "label": "Unavailable",
            "metric": "Status unknown",
            "detail": "Monitoring History could not be reached.",
            "count": None,
        }


def _next_step(status: dict[str, Any]) -> dict[str, str]:
    if status["source"]["state"] != "ready":
        return {
            "label": "Retry connection check",
            "detail": "Verify the Prometheus deployment setting before collecting data.",
            "url": reverse("home"),
        }
    if status["catalogue"]["state"] == "unavailable":
        return {
            "label": "Check the Data Catalogue",
            "detail": "The catalogue status is unknown; retry it before preparing new training data.",
            "url": reverse("data_catalogue_overview"),
        }
    if status["catalogue"].get("count") == 0:
        return {
            "label": "Add the first dataset",
            "detail": "Define the monitored application, time range, workload context, and Prometheus queries.",
            "url": reverse("data_catalogue_new"),
        }
    if status["models"]["state"] == "unavailable":
        return {
            "label": "Check models and training",
            "detail": "The learning service status is unknown; verify it before starting a new run.",
            "url": reverse("models_overview"),
        }
    if status["models"].get("count") == 0:
        return {
            "label": "Train detectors",
            "detail": "Use a prepared catalogue partition to create comparable saved models.",
            "url": reverse("models_train"),
        }
    if status["comparisons"]["state"] == "unavailable":
        return {
            "label": "Check model comparisons",
            "detail": "Comparison status is unknown; verify the learning service before continuing.",
            "url": reverse("comparison_overview"),
        }
    if status["comparisons"].get("count") == 0:
        return {
            "label": "Compare saved models",
            "detail": "Evaluate at least two compatible models on the same labelled context.",
            "url": reverse("comparison_new"),
        }
    if status["models"].get("live_ready_count") == 0:
        return {
            "label": "Review model readiness",
            "detail": "Select or train a model with a live-input recipe before monitoring.",
            "url": f'{reverse("models_overview")}?tab=saved',
        }
    if status["monitoring"]["state"] == "unavailable":
        return {
            "label": "Check live monitoring",
            "detail": "Monitoring status is unknown; verify Data Ingestion before starting a session.",
            "url": reverse("monitoring_overview"),
        }
    if status["monitoring"].get("count") == 0:
        return {
            "label": "Start live monitoring",
            "detail": "Run an inference-ready model and optionally enable automatic RCA.",
            "url": reverse("monitoring"),
        }
    return {
        "label": "Review monitoring evidence",
        "detail": "Inspect detections, anomaly episodes, and stored RCA results.",
        "url": reverse("monitoring_overview"),
    }


def home(request):
    """Render the workflow immediately; live status is loaded separately."""
    return render(
        request,
        "config_app/home.html",
        {
            "source_type": "Prometheus",
        },
    )


@require_GET
def workflow_status(request):
    """Return a bounded, operator-safe summary of workflow readiness."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            "source": executor.submit(_prometheus_status),
            "catalogue": executor.submit(_catalogue_status),
            "learning": executor.submit(_learning_status),
            "monitoring": executor.submit(_monitoring_status),
        }
        status = {
            "source": futures["source"].result(),
            "catalogue": futures["catalogue"].result(),
            "monitoring": futures["monitoring"].result(),
        }
        status.update(futures["learning"].result())
    status["next_step"] = _next_step(status)
    return JsonResponse(status)
