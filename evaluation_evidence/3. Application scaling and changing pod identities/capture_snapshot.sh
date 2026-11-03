#!/usr/bin/env bash
set -euo pipefail

label="${1:?snapshot label is required}"
evidence_root="${2:-/home/ubuntu/boutique/evaluation_evidence/topology-scaling-20261026}"
namespace="boutique-scale-test"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output="${evidence_root}/${label}-snapshot-${timestamp}.txt"

mkdir -p "${evidence_root}"

{
  echo "label=${label}"
  echo "captured_at=${timestamp}"
  echo
  echo "== Nodes =="
  kubectl get nodes -o wide
  echo
  echo "== Deployments =="
  kubectl get deployments -n "${namespace}" -o wide
  echo
  echo "== Pods =="
  kubectl get pods -n "${namespace}" -o wide
  echo
  echo "== Pod resource use =="
  kubectl top pods -n "${namespace}"
  echo
  echo "== Node resource use =="
  kubectl top nodes
  echo
  echo "== Host memory and disk =="
  free -m
  df -h /
  echo
  echo "== Frontend smoke test =="
  node_ip="$(minikube ip)"
  node_port="$(kubectl get service frontend-external -n "${namespace}" -o jsonpath='{.spec.ports[0].nodePort}')"
  echo "url=http://${node_ip}:${node_port}/"
  curl -sS -o /dev/null \
    -w 'http_status=%{http_code} bytes=%{size_download} time_seconds=%{time_total}\n' \
    --max-time 20 "http://${node_ip}:${node_port}/"
  echo
  echo "== Prometheus service-oriented query check =="
  ingestion_pod="$(kubectl get pods -n cloudsentinel -o name | grep data-ingestion | head -1 | cut -d/ -f2)"
  kubectl exec -i -n cloudsentinel "${ingestion_pod}" -- python - <<'PY'
import json
import requests

endpoint = (
    "http://kube-prometheus-kube-prome-prometheus.monitoring.svc.cluster.local:9090"
    "/api/v1/query"
)
namespace = "boutique-scale-test"
services = [
    "adservice",
    "cartservice",
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "frontend",
    "paymentservice",
    "productcatalogservice",
    "recommendationservice",
    "redis-cart",
    "shippingservice",
]

results = []
for service in services:
    selector = f'namespace="{namespace}",pod=~"{service}(-.*)?"'
    queries = {
        "cpu_cores": f"sum(rate(container_cpu_usage_seconds_total{{{selector}}}[1m]))",
        "memory_bytes": f"sum(container_memory_usage_bytes{{{selector}}})",
    }
    item = {"service": service, "pod_pattern": f"{service}(-.*)?", "values": {}}
    for name, query in queries.items():
        response = requests.get(endpoint, params={"query": query}, timeout=15)
        response.raise_for_status()
        payload = response.json()
        series = payload.get("data", {}).get("result", [])
        item["values"][name] = series[0]["value"][1] if series else None
        item["values"][f"{name}_query"] = query
    results.append(item)

identity_query = f'count(kube_pod_info{{namespace="{namespace}"}}) by (pod, node)'
identity_response = requests.get(
    endpoint, params={"query": identity_query}, timeout=15
)
identity_response.raise_for_status()

print(
    json.dumps(
        {
            "service_aggregates": results,
            "pod_identity_query": identity_query,
            "pod_identities": identity_response.json().get("data", {}).get("result", []),
        },
        indent=2,
        sort_keys=True,
    )
)
PY
} | tee "${output}"

echo "snapshot_file=${output}"
