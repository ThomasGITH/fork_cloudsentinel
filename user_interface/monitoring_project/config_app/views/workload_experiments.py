"""UI for bounded workload experiments and their dataset hand-off."""

from __future__ import annotations

from urllib.parse import urlencode

from django.contrib import messages
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from ..catalogue_client import CatalogueClientError, get_catalogue_client


def _dataset_url(experiment: dict) -> str | None:
    completed = experiment.get("completed_at")
    started = experiment.get("started_at")
    if not completed or not started:
        return None
    context = experiment.get("workload_context") or {}
    parameters = {
        "start_time": started,
        "end_time": completed,
        "incident_start_time": started,
        "incident_end_time": completed,
        "workload_intensity": context.get("workload_intensity", "High"),
        "dominant_workload_characteristic": context.get("dominant_workload_characteristic", "High traffic"),
        "anomaly_scenario": context.get("anomaly_scenario", "Other"),
        "anomaly_scenario_detail": context.get("anomaly_scenario_detail", experiment.get("profile_label", "Controlled workload")),
    }
    return f'{reverse("data_catalogue_new")}?{urlencode(parameters)}'


@require_http_methods(["GET", "POST"])
def workload_experiments(request):
    client = get_catalogue_client()
    error = None
    if request.method == "POST":
        try:
            if request.POST.get("action") == "stop":
                client.update_workload_experiment({"action": "stop"})
                messages.success(request, "The workload was restored to its previous settings.")
            else:
                client.update_workload_experiment(
                    {
                        "action": "start",
                        "profile": request.POST.get("profile", ""),
                        "duration_minutes": request.POST.get("duration_minutes", "5"),
                    }
                )
                messages.success(
                    request,
                    "The controlled workload has started. It will restore automatically.",
                )
            return redirect("workload_experiments")
        except CatalogueClientError as exc:
            error = str(exc)
    try:
        experiment_data = client.get_workload_experiments()
    except CatalogueClientError as exc:
        experiment_data = {"profiles": [], "history": [], "active": None}
        error = error or str(exc)
    for experiment in experiment_data.get("history", []):
        experiment["dataset_url"] = _dataset_url(experiment)
    return render(
        request,
        "config_app/workload_experiments.html",
        {"experiments": experiment_data, "error": error},
    )
