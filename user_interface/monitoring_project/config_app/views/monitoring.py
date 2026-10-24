"""Existing monitoring screens backed by generic Saved Model inference."""

from __future__ import annotations

import requests
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

from ..forms import MonitoringForm
from ..comparison_labels import readable_model_name
from ..learning_client import LearningClientError, get_learning_client
from ..catalogue_client import CatalogueClientError, get_catalogue_client
from ..workflow_context import (
    context_from_request,
    encode_context,
    new_context,
    progress_context,
    update_context,
)


DEFAULT_RCA_METHOD = "cloudsentinel.causal-rca/legacy-v1"
BARO_RCA_METHOD = "cloudsentinel.baro-robust-scorer/v1"

RCA_METHOD_PRESENTATION = {
    DEFAULT_RCA_METHOD: {
        "display_name": "CausalRCA",
        "description": "Dependency-oriented analysis that ranks features and can show a dependency graph when one is available.",
        "input_type": "Time-aligned metrics with explicit, unique feature identities.",
        "output_type": "Ranked suspects and, when available, a dependency graph.",
    },
    BARO_RCA_METHOD: {
        "display_name": "BARO RobustScorer",
        "description": "Ranks robust changes between a baseline and the detector-derived anomalous segment.",
        "input_type": "Raw finite metrics with a valid detector-derived change point and sufficient baseline and anomalous observations.",
        "output_type": "Ranked suspects with change-point evidence; no causal graph.",
    },
}


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
            item["display_name"],
        )
        for item in items
        if _ready(item)
    ]


def _available_rca_methods(items):
    methods = {}
    for item in items:
        readiness = item.get("rca_readiness")
        if not isinstance(readiness, dict):
            continue
        advertised = readiness.get("methods")
        if isinstance(advertised, list):
            for method in advertised:
                if (
                    isinstance(method, dict)
                    and method.get("status") == "available"
                    and isinstance(method.get("id"), str)
                ):
                    methods[method["id"]] = str(method.get("display_name") or method["id"])
        elif readiness.get("status") == "ready":
            # Backward compatibility with a Data Ingestion API from before
            # multi-method capability advertisement.
            methods[DEFAULT_RCA_METHOD] = "CausalRCA"
    return sorted(methods.items(), key=lambda item: item[1].lower())


def _model_available_rca_methods(item):
    readiness = item.get("rca_readiness") if isinstance(item, dict) else None
    if not isinstance(readiness, dict):
        return set()
    advertised = readiness.get("methods")
    if isinstance(advertised, list):
        return {
            method["id"]
            for method in advertised
            if (
                isinstance(method, dict)
                and method.get("status") == "available"
                and isinstance(method.get("id"), str)
            )
        }
    return {DEFAULT_RCA_METHOD} if readiness.get("status") == "ready" else set()


def _rca_method_catalog(items):
    """Build a safe operator-facing projection of advertised method capability."""
    advertised = {method_id: [] for method_id in RCA_METHOD_PRESENTATION}
    for item in items:
        readiness = item.get("rca_readiness") if isinstance(item, dict) else None
        methods = readiness.get("methods") if isinstance(readiness, dict) else None
        if isinstance(methods, list):
            for method in methods:
                if isinstance(method, dict) and method.get("id") in advertised:
                    advertised[method["id"]].append(method)
        elif isinstance(readiness, dict) and readiness.get("status") == "ready":
            advertised[DEFAULT_RCA_METHOD].append({"status": "available"})

    result = []
    for method_id, fallback in RCA_METHOD_PRESENTATION.items():
        capabilities = advertised[method_id]
        available = any(item.get("status") == "available" for item in capabilities)
        representative = next(
            (item for item in capabilities if item.get("status") == "available"),
            capabilities[0] if capabilities else {},
        )
        reason = None
        if not available:
            reason = next(
                (str(item.get("reason"))[:500] for item in capabilities if item.get("reason")),
                "This method is not available for the current live-ready model inputs.",
            )
        result.append(
            {
                "id": method_id,
                # Stable UI names avoid small spelling/capitalisation differences
                # between older capability responses.
                "display_name": fallback["display_name"],
                "description": str(representative.get("description") or fallback["description"]),
                "input_type": str(representative.get("input_type") or fallback["input_type"]),
                "output_type": str(representative.get("output_type") or fallback["output_type"]),
                "status": "available" if available else "unavailable",
                "reason": reason,
            }
        )
    return result


def monitoring_home(request):
    return render(request, "config_app/monitoring/monitoring_home.html")


def monitoring(request):
    workflow = context_from_request(request)
    if workflow and request.GET.get("model_id"):
        workflow = update_context(workflow, model_id=request.GET["model_id"])
    try:
        models = _add_rca_readiness(_models())
        upstream_error = None
    except LearningClientError:
        models = []
        upstream_error = "Saved Models are temporarily unavailable."
    form = MonitoringForm(
        request.POST or None,
        initial={
            "model_id": request.GET.get("model_id", ""),
            "rca_method": DEFAULT_RCA_METHOD,
        },
    )
    form.fields["model_id"].choices = _choices(models)
    form.fields["rca_method"].choices = _available_rca_methods(models)
    if request.method == "POST" and form.is_valid():
        selected = next(
            (item for item in models if item.get("model_id") == form.cleaned_data["model_id"]),
            None,
        )
        if selected is None or not _ready(selected):
            form.add_error("model_id", "This model is not ready for live monitoring.")
        elif (
            form.cleaned_data["automatic_rca_enabled"]
            and (form.cleaned_data["rca_method"] or DEFAULT_RCA_METHOD)
            not in _model_available_rca_methods(selected)
        ):
            form.add_error("rca_method", "This RCA method is unavailable for the selected model input.")
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
                            "method": form.cleaned_data["rca_method"] or DEFAULT_RCA_METHOD,
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
            session_id = (payload.get("session") or {}).get("session_id") if isinstance(payload, dict) else None
            workflow = update_context(
                workflow or new_context(),
                model_id=selected["model_id"],
                monitoring_session_id=session_id,
            )
            payload["workflow_token"] = encode_context(workflow)
            return JsonResponse(payload, status=202)
    if request.method == "POST" and request.headers.get("X-Requested-With") == "XMLHttpRequest":
        first_error = next(
            (
                str(error)
                for errors in form.errors.values()
                for error in errors
            ),
            "Check the monitoring settings and try again.",
        )
        return JsonResponse(
            {"status": "error", "message": first_error[:500]},
            status=400,
        )
    try:
        observability = get_catalogue_client().get_monitored_environment()
    except CatalogueClientError:
        observability = {}
    return render(
        request,
        "config_app/monitoring/monitoring_setup.html",
        {
            "form": form,
            "models": models,
            "upstream_error": upstream_error,
            "ready_count": len(_choices(models)),
            "rca_methods": _rca_method_catalog(models),
            "workflow_progress": progress_context(workflow, "monitoring"),
            "workflow_token": encode_context(workflow) if workflow else "",
            "observability": observability,
        },
    )


def monitoring_overview(request):
    workflow = context_from_request(request)
    session_id = request.GET.get("session")
    if workflow and session_id:
        workflow = update_context(workflow, monitoring_session_id=session_id)
    return render(
        request,
        "config_app/monitoring/monitoring_dashboard.html",
        {
            "workflow_progress": progress_context(
                workflow, "evidence" if session_id else "monitoring"
            ),
            "workflow_token": encode_context(workflow) if workflow else "",
        },
    )


def load_kube_config():
    return JsonResponse({"status": "not_required"})


def task_manager(request):
    return render(request, "config_app/monitoring/monitoring_dashboard.html")
