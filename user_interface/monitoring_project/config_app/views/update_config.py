from django.conf import settings
from django.shortcuts import render
from django.views.decorators.http import require_GET

from .utils import update_config


@require_GET
def config(request):
    """Show the effective deployment-managed service connections.

    The retired form wrote a local config.json that Django no longer loaded and
    that could not update the Data Catalogue or other service deployments. A
    read-only view is intentionally honest about that deployment boundary.
    """
    connections = [
        {
            "name": "Prometheus",
            "purpose": "Direct source check from the user interface",
            "environment": "PROMETHEUS_URL",
            "value": settings.PROMETHEUS_URL,
        },
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
            "name": "Data Processing",
            "purpose": "Metrics preprocessing used by legacy flows",
            "environment": "API_DATA_PROCESSING_URL",
            "value": settings.API_DATA_PROCESSING_URL,
        },
        {
            "name": "Cluster namespace",
            "purpose": "Default namespace used by legacy discovery flows",
            "environment": "CLUSTER_NAMESPACE",
            "value": settings.CLUSTER_NAMESPACE,
        },
    ]
    return render(request, "config_app/update_config.html", {"connections": connections})


def config_cgnn(request):
    return update_config(request, settings.API_CGNN_ANOMALY_DETECTION_URL, 'perform_anomaly_detection_cgnn')


def config_crca(request):
    return update_config(request, settings.API_CRCA_ANOMALY_DETECTION_URL, 'perform_anomaly_detection_crca')


def config_cgnn_train(request):
    return update_config(request, settings.API_LEARNING_ADAPTATION_URL, 'train_cgnn')
