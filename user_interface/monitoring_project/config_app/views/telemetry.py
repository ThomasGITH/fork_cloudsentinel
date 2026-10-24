from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from ..catalogue_client import CatalogueClientError, get_catalogue_client
from ..observability_links import links_for


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@require_http_methods(["GET", "POST"])
def telemetry_explorer(request):
    client = get_catalogue_client()
    if request.method == "POST":
        try:
            payload = json.loads(request.body)
            result = client.query_telemetry(payload)
            environment = client.get_monitored_environment()
            result["external_links"] = links_for(
                environment,
                result["query"],
                result["start_time"],
                result["end_time"],
            )
            return JsonResponse(result)
        except (json.JSONDecodeError, KeyError, CatalogueClientError) as exc:
            status = exc.status_code if isinstance(exc, CatalogueClientError) else 400
            return JsonResponse({"error": str(exc)}, status=status)

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    try:
        environment = client.get_monitored_environment()
    except CatalogueClientError:
        environment = {}
    return render(
        request,
        "config_app/telemetry/explorer.html",
        {
            "environment": environment,
            "default_query": "up",
            "default_start": _iso(now - timedelta(hours=1)),
            "default_end": _iso(now),
        },
    )
