from django.conf import settings
from django.contrib import messages
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods

from config_app.catalogue_client import CatalogueClientError, get_catalogue_client

from .utils import update_config


@require_http_methods(["GET", "POST"])
def config(request):
    """Edit the actual catalogue source and show deployment wiring separately."""
    client = get_catalogue_client()
    error = None
    submitted = None
    if request.method == "POST":
        submitted = {
            "prometheus_url": request.POST.get("prometheus_url", "").strip(),
            "default_namespace": request.POST.get("default_namespace", "").strip(),
            "source_id": "cluster-default",
        }
        try:
            if request.POST.get("action") == "test":
                result = client.test_monitored_environment(submitted["prometheus_url"])
                messages.success(request, result.get("message", "Connection succeeded."))
            else:
                client.update_monitored_environment(submitted)
                messages.success(request, "Monitored environment settings saved.")
                return redirect("config")
        except CatalogueClientError as exc:
            error = str(exc)

    try:
        environment = submitted or client.get_monitored_environment()
    except CatalogueClientError as exc:
        environment = submitted or {
            "prometheus_url": settings.PROMETHEUS_URL,
            "default_namespace": settings.CLUSTER_NAMESPACE,
            "source_id": "cluster-default",
        }
        error = error or str(exc)

    diagnostics = [
        {
            "name": "Data Ingestion and Catalogue",
            "purpose": "Dataset capture, monitoring sessions, and RCA orchestration",
            "environment": "API_DATA_INGESTION_URL",
            "value": settings.API_DATA_INGESTION_URL,
        },
        {
            "name": "Learning and Model Catalogue",
            "purpose": "Detector discovery, training, saved models, and comparisons",
            "environment": "API_LEARNING_ADAPTATION_URL",
            "value": settings.API_LEARNING_ADAPTATION_URL,
        },
        {
            "name": "Data Processing (legacy)",
            "purpose": "Metrics preprocessing used only by legacy flows",
            "environment": "API_DATA_PROCESSING_URL",
            "value": settings.API_DATA_PROCESSING_URL,
        },
    ]
    return render(
        request,
        "config_app/update_config.html",
        {"environment": environment, "diagnostics": diagnostics, "error": error},
    )


def config_cgnn(request):
    return update_config(request, settings.API_CGNN_ANOMALY_DETECTION_URL, 'perform_anomaly_detection_cgnn')


def config_crca(request):
    return update_config(request, settings.API_CRCA_ANOMALY_DETECTION_URL, 'perform_anomaly_detection_crca')


def config_cgnn_train(request):
    return update_config(request, settings.API_LEARNING_ADAPTATION_URL, 'train_cgnn')
