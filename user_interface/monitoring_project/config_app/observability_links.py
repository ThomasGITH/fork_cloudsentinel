"""Build credential-free deep links to external observability tools."""

from __future__ import annotations

import json
from urllib.parse import urlencode


def prometheus_graph_url(base_url: str, query: str, start: str, end: str) -> str:
    if not base_url:
        return ""
    return f"{base_url.rstrip('/')}/graph?{urlencode({'g0.expr': query, 'g0.tab': '0', 'g0.range_input': f'{start} to {end}'})}"


def grafana_explore_url(base_url: str, datasource_uid: str, query: str, start: str, end: str) -> str:
    if not base_url:
        return ""
    panes = {
        "cloudsentinel": {
            "datasource": datasource_uid or "prometheus",
            "queries": [{"refId": "A", "expr": query, "range": True}],
            "range": {"from": start, "to": end},
        }
    }
    return f"{base_url.rstrip('/')}/explore?{urlencode({'schemaVersion': 1, 'panes': json.dumps(panes, separators=(',', ':'))})}"


def links_for(environment: dict, query: str, start: str, end: str) -> dict[str, str]:
    return {
        "prometheus": prometheus_graph_url(
            environment.get("prometheus_browser_url", ""), query, start, end
        ),
        "grafana": grafana_explore_url(
            environment.get("grafana_browser_url", ""),
            environment.get("grafana_datasource_uid", "prometheus"),
            query,
            start,
            end,
        ),
    }
