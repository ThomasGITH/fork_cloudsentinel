"""Create and fetch a service-oriented topology-scaling dataset.

Run this inside the active data-ingestion container. The script intentionally
uses the public Data Catalogue HTTP API rather than reading catalogue storage.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sys
import time

import requests


BASE_URL = "http://127.0.0.1:5001"
NAMESPACE = "boutique-scale-test"
SERVICES = [
    "currencyservice",
    "frontend",
    "paymentservice",
    "productcatalogservice",
    "shippingservice",
]


def utc_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def response_json(response: requests.Response) -> dict:
    try:
        body = response.json()
    except ValueError:
        body = {"raw_body": response.text[:2000]}
    if not response.ok:
        raise RuntimeError(
            f"{response.request.method} {response.url} returned "
            f"HTTP {response.status_code}: {body}"
        )
    return body


def main() -> int:
    dataset_id = sys.argv[1] if len(sys.argv) > 1 else "topology-scale-transition-20261026"
    window_minutes = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    now = datetime.now(timezone.utc).replace(microsecond=0)
    end = now - timedelta(seconds=now.second % 30)
    start = end - timedelta(minutes=window_minutes)

    payload = {
        "dataset_id": dataset_id,
        "display_name": "Online Boutique topology scaling transition",
        "description": (
            "Service-level CPU and memory telemetry across a change from one "
            "to three replicas for five stateless services."
        ),
        "purpose": ["training", "evaluation"],
        "source": {
            "type": "prometheus",
            "prometheus_source_id": "cluster-default",
            "application": "Online Boutique",
            "namespace": NAMESPACE,
            "requested_targets": {
                "services": SERVICES,
                "pods": [],
                "label_selectors": ["stable service-name regex"],
            },
            "start_time": utc_timestamp(start),
            "end_time": utc_timestamp(end),
            "sampling_interval_seconds": 30,
            "queries": [
                {
                    "query_id": "service_cpu",
                    "display_name": "Service CPU use",
                    "feature_name": "cpu_cores",
                    "promql_template": (
                        'sum(rate(container_cpu_usage_seconds_total{namespace="${namespace}",'
                        'pod=~"${service}(-.*)?"}[1m]))'
                    ),
                    "required": True,
                    "execution_mode": "per_target",
                    "target_type": "service",
                    "identity_labels": [],
                },
                {
                    "query_id": "service_memory",
                    "display_name": "Service memory use",
                    "feature_name": "memory_bytes",
                    "promql_template": (
                        'sum(container_memory_usage_bytes{namespace="${namespace}",'
                        'pod=~"${service}(-.*)?"})'
                    ),
                    "required": True,
                    "execution_mode": "per_target",
                    "target_type": "service",
                    "identity_labels": [],
                },
            ],
        },
        "workload_context": {
            "workload_intensity": "Variable",
            "dominant_workload_characteristic": "Mixed",
            "anomaly_scenario": "Normal operation",
        },
        "partition": {"mode": "none"},
    }

    created = response_json(requests.post(f"{BASE_URL}/datasets", json=payload, timeout=20))
    print("== Create response ==")
    print(json.dumps(created, indent=2, sort_keys=True))

    started = response_json(
        requests.post(f"{BASE_URL}/datasets/{dataset_id}/versions/1/fetch", timeout=20)
    )
    print("== Fetch start response ==")
    print(json.dumps(started, indent=2, sort_keys=True))

    status = {}
    for _ in range(60):
        status = response_json(
            requests.get(
                f"{BASE_URL}/datasets/{dataset_id}/fetch-status",
                params={"version": 1},
                timeout=20,
            )
        )
        if status.get("phase") in {"available", "failed"}:
            break
        time.sleep(2)
    else:
        raise RuntimeError("catalogue fetch did not finish within 120 seconds")

    print("== Final fetch status ==")
    print(json.dumps(status, indent=2, sort_keys=True))
    if status.get("phase") != "available":
        return 2

    detail = response_json(
        requests.get(f"{BASE_URL}/datasets/{dataset_id}/versions/1", timeout=20)
    )
    preview = response_json(
        requests.get(
            f"{BASE_URL}/datasets/{dataset_id}/preview",
            params={"version": 1, "limit": 5},
            timeout=20,
        )
    )
    print("== Stored version ==")
    print(json.dumps(detail, indent=2, sort_keys=True))
    print("== Preview ==")
    print(json.dumps(preview, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
