#!/usr/bin/env python3
"""Run a bounded NaaVRE catalogue and live-input case through CloudSentinel."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import uuid

from flask import Flask

from data_ingestion.catalogue.api import configure_catalogue_defaults
from data_ingestion.catalogue.fetching import (
    FetchLimits,
    build_query_executions,
    execute_catalogue_fetch,
    start_fetch_attempt,
)
from data_ingestion.catalogue.models import build_draft_dataset
from data_ingestion.catalogue.repository import FileCatalogueRepository
from data_ingestion.live_monitoring import collect_recipe_window
from evaluation.naavre.collect_read_only_evidence import (
    DEFAULT_BASE_URL,
    DEFAULT_DATASOURCE_UID,
    load_token,
    utc,
)


METRICS = {
    "cpu_utilisation": (
        'instance:node_cpu_utilisation:rate5m{job="node-exporter",'
        'instance="k8s-staging-1"}'
    ),
    "disk_io_utilisation": (
        'instance_device:node_disk_io_time_seconds:rate5m{job="node-exporter",'
        'instance="k8s-staging-1",device="sda"}'
    ),
    "network_receive_bytes": (
        'instance:node_network_receive_bytes_excluding_lo:rate5m{job="node-exporter",'
        'instance="k8s-staging-1"}'
    ),
    "network_transmit_bytes": (
        'instance:node_network_transmit_bytes_excluding_lo:rate5m{job="node-exporter",'
        'instance="k8s-staging-1"}'
    ),
    "memory_utilisation": (
        'instance:node_memory_utilisation:ratio{job="node-exporter",'
        'instance="k8s-staging-1"}'
    ),
}


def query_specs() -> list[dict[str, object]]:
    return [
        {
            "query_id": name,
            "display_name": name.replace("_", " ").title(),
            "promql_template": expression,
            "feature_name": name,
            "required": True,
            "execution_mode": "single",
            "identity_labels": ["instance"],
        }
        for name, expression in METRICS.items()
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    token = load_token(args.token_file)
    token_environment = "NAAVRE_EVALUATION_TOKEN"
    os.environ[token_environment] = token
    end = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=2)
    start = end - timedelta(hours=args.dataset_hours)
    dataset_id = f"naavre-portability-{end.strftime('%Y%m%d-%H%M%S')}"

    with tempfile.TemporaryDirectory(prefix="cloudsentinel-naavre-") as directory:
        app = Flask(__name__)
        configure_catalogue_defaults(app)
        app.config.update(
            CATALOGUE_STORAGE_ROOT=directory,
            LEGACY_DATASETS_ROOT=str(Path(directory) / "legacy"),
            CATALOGUE_MAX_RETRIES=0,
            CATALOGUE_PROMETHEUS_SOURCES={
                "cluster-default": "http://prometheus.invalid",
                "naavre": {
                    "transport": "grafana-datasource-proxy",
                    "base_url": args.base_url,
                    "datasource_uid": args.datasource_uid,
                    "bearer_token_env": token_environment,
                },
            },
        )
        repository = FileCatalogueRepository(directory)
        payload = {
            "dataset_id": dataset_id,
            "display_name": "NaaVRE operational telemetry portability case",
            "description": (
                "Unlabelled operational telemetry used to test external-source "
                "portability; it is not presented as a confirmed incident."
            ),
            "purpose": ["portability evaluation", "live-input smoke test"],
            "source": {
                "type": "prometheus",
                "prometheus_source_id": "naavre",
                "application": "NaaVRE",
                "namespace": "external-operation",
                "requested_targets": {
                    "services": [],
                    "pods": [],
                    "label_selectors": [],
                },
                "start_time": utc(start),
                "end_time": utc(end),
                "sampling_interval_seconds": 65,
                "queries": query_specs(),
            },
            "workload_context": {
                "workload_intensity": "Variable",
                "dominant_workload_characteristic": "Mixed",
                "anomaly_scenario": "Unknown",
                "anomaly_scenario_detail": (
                    "No confirmed incident window; portability evidence only."
                ),
            },
            "ground_truth": {
                "labels_available": False,
                "known_incident_windows": [],
            },
            "partition": {"mode": "none"},
        }
        record, version = build_draft_dataset(payload)
        repository.create(record, version)
        executions = build_query_executions(
            version, FetchLimits.from_config(dict(app.config))
        )
        attempt_id = f"attempt-{uuid.uuid4().hex}"
        task_id = f"catalogue-fetch-{uuid.uuid4().hex}"
        start_fetch_attempt(
            repository, dataset_id, 1, attempt_id, task_id, len(executions)
        )
        execute_catalogue_fetch(
            dataset_id, 1, attempt_id, dict(app.config)
        )
        fetched = repository.read_version(dataset_id, 1)

        live_end = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=1)
        live_start = live_end - timedelta(minutes=args.live_minutes)
        live = collect_recipe_window(
            fetched["live_input_recipe"],
            live_start.timestamp(),
            live_end.timestamp(),
            dict(app.config),
        )
        query_diagnostics = [
            item.get("transport_diagnostics", {})
            for item in fetched["provenance"]["resolved_queries"]
        ]
        request_durations = [
            item.get("request_duration_seconds")
            for item in query_diagnostics
            if isinstance(item.get("request_duration_seconds"), (int, float))
        ]
        kept_artifacts = {
            "canonical/observations.csv",
            "canonical/feature_matrix.csv",
            "schema.json",
            "provenance.json",
            "live_input_recipe.json",
        }
        artifact_checksums = {
            name: value
            for name, value in fetched["checksums"].items()
            if name in kept_artifacts
        }
        summary = {
            "schema_version": 1,
            "case": "NaaVRE read-only portability and live-input smoke test",
            "completed_at_utc": utc(datetime.now(timezone.utc)),
            "dataset": {
                "dataset_id": dataset_id,
                "window": {"start": utc(start), "end": utc(end)},
                "sampling_interval_seconds": 65,
                "status": fetched["status"],
                "observation_count": fetched["technical_schema"]["observation_count"],
                "feature_count": fetched["technical_schema"]["feature_count"],
                "feature_names": list(METRICS),
                "feature_order_sha256": fetched["live_input_recipe"][
                    "feature_order_sha256"
                ],
                "missing_values": fetched["technical_schema"]["missing_values"],
                "labels_available": False,
                "artifact_checksums": artifact_checksums,
                "live_input_recipe_sha256": fetched["live_input_recipe"]["recipe_sha256"],
            },
            "catalogue_fetch": {
                "transport": "grafana-datasource-proxy",
                "query_count": len(query_diagnostics),
                "total_series": sum(item.get("series_count", 0) for item in query_diagnostics),
                "total_samples": sum(item.get("sample_count", 0) for item in query_diagnostics),
                "mean_request_duration_seconds": (
                    round(sum(request_durations) / len(request_durations), 4)
                    if request_durations
                    else None
                ),
                "maximum_request_duration_seconds": (
                    round(max(request_durations), 4) if request_durations else None
                ),
            },
            "live_input": {
                "window": {"start": utc(live_start), "end": utc(live_end)},
                "observation_count": len(live["rows"]),
                "feature_order_matches_catalogue": (
                    live["feature_order"] == fetched["technical_schema"]["feature_order"]
                ),
                "missing_values": live["missing_values"],
                "fetch_diagnostics": live["fetch_diagnostics"],
            },
            "code_boundary": {
                "branch": "codex/naavre-grafana-proxy",
                "main_branch_changed": False,
                "adapter_scope": "evaluation-only",
            },
            "interpretation_limits": [
                "The telemetry has no confirmed incident timestamps or row labels.",
                "The case tests data-source portability and live-input compatibility.",
                "It does not establish anomaly-detection or RCA accuracy.",
                "CloudSentinel was not deployed inside NaaVRE.",
            ],
        }
    os.environ.pop(token_environment, None)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--datasource-uid", default=DEFAULT_DATASOURCE_UID)
    parser.add_argument("--dataset-hours", type=int, default=24)
    parser.add_argument("--live-minutes", type=int, default=15)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["dataset"]["status"],
                "observations": result["dataset"]["observation_count"],
                "features": result["dataset"]["feature_count"],
                "live_observations": result["live_input"]["observation_count"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
