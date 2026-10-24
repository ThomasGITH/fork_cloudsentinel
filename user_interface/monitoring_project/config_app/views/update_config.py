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
        if "auth_type" in request.POST:
            submitted.update({
                "auth_type": request.POST.get("auth_type", "none"),
                "username": request.POST.get("username", "").strip(),
                "password": request.POST.get("password", ""),
                "bearer_token": request.POST.get("bearer_token", ""),
                "ca_certificate": request.POST.get("ca_certificate", ""),
                "tls_verify": request.POST.get("tls_verify") == "on",
                "clear_credentials": request.POST.get("clear_credentials") == "on",
                "prometheus_browser_url": request.POST.get("prometheus_browser_url", "").strip(),
                "grafana_browser_url": request.POST.get("grafana_browser_url", "").strip(),
                "grafana_datasource_uid": request.POST.get("grafana_datasource_uid", "prometheus").strip(),
            })
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
            "auth_type": "none",
            "tls_verify": True,
            "prometheus_browser_url": "",
            "grafana_browser_url": "",
            "grafana_datasource_uid": "prometheus",
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
