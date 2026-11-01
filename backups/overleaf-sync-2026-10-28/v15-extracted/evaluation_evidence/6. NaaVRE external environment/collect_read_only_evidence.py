#!/usr/bin/env python3
"""Collect privacy-conscious, read-only evidence from NaaVRE's Grafana proxy.

The output deliberately contains aggregate series only. Authentication material and
raw Prometheus labels are neither printed nor written to disk.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import quote

import requests


DEFAULT_BASE_URL = "https://staging.demo.naavre.net/grafana"
DEFAULT_DATASOURCE_UID = "prometheus"
TOKEN_PATTERN = re.compile(r"glsa_[A-Za-z0-9_-]+")
EVENT_QUERIES = {
    "firing_alerts": 'sum(ALERTS{alertstate="firing"}) or vector(0)',
    "container_restarts_15m": (
        "sum(increase(kube_pod_container_status_restarts_total[15m])) or vector(0)"
    ),
    "oom_killed_containers": (
        'sum(kube_pod_container_status_last_terminated_reason{reason="OOMKilled"}) '
        "or vector(0)"
    ),
    "failed_or_unknown_pods": (
        'sum(kube_pod_status_phase{phase=~"Failed|Unknown"} == 1) or vector(0)'
    ),
}


def utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_token(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = TOKEN_PATTERN.findall(text)
    if len(matches) != 1:
        raise RuntimeError("expected exactly one Grafana service-account token")
    return matches[0]


class ReadOnlyGrafana:
    def __init__(self, base_url: str, datasource_uid: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.datasource_uid = datasource_uid
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.timings: list[float] = []

    def prometheus_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = (
            f"{self.base_url}/api/datasources/proxy/uid/"
            f"{quote(self.datasource_uid, safe='')}{path}"
        )
        started = time.monotonic()
        response = self.session.get(url, params=params, timeout=(10, 45))
        self.timings.append(time.monotonic() - started)
        if response.status_code != 200:
            raise RuntimeError(f"Grafana query failed with HTTP {response.status_code}")
        payload = response.json()
        if payload.get("status") != "success":
            raise RuntimeError("Prometheus datasource returned a non-success response")
        return payload.get("data")

    def query_range(
        self, expression: str, start: datetime, end: datetime, step: int
    ) -> list[dict[str, Any]]:
        data = self.prometheus_get(
            "/api/v1/query_range",
            {
                "query": expression,
                "start": utc(start),
                "end": utc(end),
                "step": str(step),
            },
        )
        if data.get("resultType") != "matrix":
            raise RuntimeError("expected a Prometheus matrix response")
        return data.get("result", [])


def numeric_points(series: list[dict[str, Any]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for item in series:
        for sample in item.get("values", []):
            try:
                points.append((float(sample[0]), float(sample[1])))
            except (IndexError, TypeError, ValueError):
                continue
    return sorted(points)


def aggregate_summary(series: list[dict[str, Any]]) -> dict[str, Any]:
    points = numeric_points(series)
    values = [value for _, value in points]
    non_zero = [(timestamp, value) for timestamp, value in points if value > 0]
    return {
        "series_count": len(series),
        "sample_count": len(points),
        "non_zero_sample_count": len(non_zero),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "first_non_zero_utc": (
            utc(datetime.fromtimestamp(non_zero[0][0], timezone.utc))
            if non_zero
            else None
        ),
        "last_non_zero_utc": (
            utc(datetime.fromtimestamp(non_zero[-1][0], timezone.utc))
            if non_zero
            else None
        ),
    }


def collect(client: ReadOnlyGrafana, end: datetime) -> dict[str, Any]:
    short_start = end - timedelta(hours=1)
    event_start = end - timedelta(days=7)
    retention_start = end - timedelta(days=45)

    availability_series = client.query_range(
        'up{node="k8s-staging-1"}', short_start, end, 60
    )
    availability_points = numeric_points(availability_series)
    latest_timestamp = max((item[0] for item in availability_points), default=None)

    event_signals = {
        name: aggregate_summary(client.query_range(query, event_start, end, 300))
        for name, query in EVENT_QUERIES.items()
    }
    retention_series = client.query_range(
        "count(up)", retention_start, end, 21_600
    )
    retention = aggregate_summary(retention_series)
    retention_points = numeric_points(retention_series)
    earliest_retained = min((item[0] for item in retention_points), default=None)

    protocol = {
        "connection": "Grafana Prometheus datasource proxy",
        "access": "read-only service account",
        "datasource_uid": client.datasource_uid,
        "availability_window": {"start": utc(short_start), "end": utc(end), "step": 60},
        "event_window": {"start": utc(event_start), "end": utc(end), "step": 300},
        "retention_probe": {
            "start": utc(retention_start),
            "end": utc(end),
            "step": 21_600,
        },
        "event_queries": EVENT_QUERIES,
        "privacy": "Only aggregate values; no raw labels or authentication material.",
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return {
        "schema_version": 1,
        "collected_at_utc": utc(datetime.now(timezone.utc)),
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "connection": {
            "successful": bool(availability_series),
            "availability_series_count": len(availability_series),
            "availability_sample_count": len(availability_points),
            "availability_minimum": (
                min((item[1] for item in availability_points), default=None)
            ),
            "latest_sample_utc": (
                utc(datetime.fromtimestamp(latest_timestamp, timezone.utc))
                if latest_timestamp is not None
                else None
            ),
            "latest_sample_age_seconds": (
                round(max(0.0, end.timestamp() - latest_timestamp), 3)
                if latest_timestamp is not None
                else None
            ),
        },
        "event_signals": event_signals,
        "retention": {
            **retention,
            "earliest_observed_sample_utc": (
                utc(datetime.fromtimestamp(earliest_retained, timezone.utc))
                if earliest_retained is not None
                else None
            ),
        },
        "request_performance": {
            "request_count": len(client.timings),
            "mean_seconds": round(sum(client.timings) / len(client.timings), 4),
            "maximum_seconds": round(max(client.timings), 4),
        },
        "interpretation_limits": [
            "No confirmed incident timestamps were available.",
            "A non-zero operational signal is not treated as a confirmed incident.",
            "This run does not establish anomaly-detection or RCA accuracy.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--datasource-uid", default=DEFAULT_DATASOURCE_UID)
    args = parser.parse_args()

    client = ReadOnlyGrafana(
        args.base_url, args.datasource_uid, load_token(args.token_file)
    )
    result = collect(client, datetime.now(timezone.utc).replace(microsecond=0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "successful": result["connection"]["successful"],
                "request_count": result["request_performance"]["request_count"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
