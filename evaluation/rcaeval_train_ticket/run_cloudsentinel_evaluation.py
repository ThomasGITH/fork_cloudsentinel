#!/usr/bin/env python3
"""Run the frozen RCAEval sample through deployed CloudSentinel HTTP APIs.

The script uses only the Python standard library so it can run on the EC2 host.
It expects local port-forwards to Data Ingestion (18001) and Learning &
Adaptation (18002), and is resumable through its state JSON file.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import uuid


TERMINAL_RUN = {"completed", "partial_success", "failed"}
TERMINAL_COMPARISON = {"completed", "partial", "failed"}


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def request_json(method: str, url: str, payload: object | None = None, timeout: int = 120):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body)
        except json.JSONDecodeError:
            detail = body
        raise RuntimeError(f"{method} {url} returned HTTP {exc.code}: {detail}") from exc


def post_multipart(url: str, metadata: dict[str, object], csv_path: Path):
    boundary = f"cloudsentinel-{uuid.uuid4().hex}"
    parts = []
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"metadata\"\r\n"
        "Content-Type: application/json\r\n\r\n"
        f"{json.dumps(metadata, separators=(',', ':'))}\r\n".encode("utf-8")
    )
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{csv_path.name}\"\r\nContent-Type: text/csv\r\n\r\n".encode("utf-8")
        + csv_path.read_bytes()
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    request = Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=180) as response:
            return time.perf_counter() - started, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"CSV import returned HTTP {exc.code}: {detail}") from exc


def workload(fault: str) -> dict[str, str]:
    return {
        "workload_intensity": "High",
        "dominant_workload_characteristic": {
            "cpu": "CPU-intensive",
            "mem": "Memory-intensive",
            "delay": "Latency-intensive",
            "loss": "High traffic",
            "disk": "Mixed",
        }[fault],
        "anomaly_scenario": {
            "cpu": "CPU stress",
            "mem": "Memory stress",
            "delay": "Network delay",
            "loss": "Unknown",
            "disk": "Unknown",
        }[fault],
    }


def case_window(csv_path: Path) -> tuple[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    first_anomaly = next(row["timestamp"] for row in rows if row["label"] == "1")
    return first_anomaly, rows[-1]["timestamp"]


def wait_for(url: str, terminal: set[str], *, maximum_seconds: int = 900):
    deadline = time.monotonic() + maximum_seconds
    while True:
        _status, value = request_json("GET", url)
        if value.get("status") in terminal:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {url}; last status: {value.get('status')}")
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--data-url", default="http://127.0.0.1:18001")
    parser.add_argument("--learning-url", default="http://127.0.0.1:18002")
    arguments = parser.parse_args()
    prepared = arguments.prepared_dir.resolve()
    state_path = arguments.state.resolve()
    selected_path = prepared / "evidence" / "selected_cases.csv"
    with selected_path.open("r", encoding="utf-8", newline="") as source:
        selected = list(csv.DictReader(source))
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {
        "protocol_version": "cloudsentinel-rcaeval-train-ticket/v1",
        "started_at": utc_now(),
        "datasets": {},
        "training": None,
        "comparisons": {},
    }

    _status, existing_page = request_json("GET", f"{arguments.data_url}/datasets?page_size=100")
    existing_by_name = {item["display_name"]: item for item in existing_page.get("items", [])}
    for index, row in enumerate(selected, start=1):
        case = row["case"]
        if case in state["datasets"]:
            continue
        display_name = f"RCAEval RE1 Train Ticket | {case}"
        csv_path = prepared / "canonical" / f"{case}.csv"
        existing = existing_by_name.get(display_name)
        if existing:
            _status, imported = request_json("GET", f"{arguments.data_url}/datasets/{existing['dataset_id']}")
            import_seconds = None
        else:
            metadata = {
                "display_name": display_name,
                "description": (
                    f"RCAEval RE1 Train Ticket case with known {row['fault']} injection on "
                    f"{row['root_cause_service']}; fixed shared 100-feature profile."
                ),
                "purpose": ["training", "evaluation"],
                "application": "RCAEval Train Ticket",
                "namespace": "external-benchmark",
                "timestamp_column": "timestamp",
                "label_column": "label",
                "train_split_percent": 30,
                "workload_context": workload(row["fault"]),
            }
            import_seconds, imported = post_multipart(
                f"{arguments.data_url}/datasets/import/csv", metadata, csv_path
            )
        detail = imported
        dataset_id = detail["dataset_id"]
        version = detail["version"]["version"]
        partition_id = detail["version"]["partition"]["partition_id"]
        injection, end = case_window(csv_path)
        incident_id = f"rcaeval-{case}"[:128]
        known_incidents = detail["version"].get("ground_truth", {}).get("known_incident_windows", [])
        if not any(item.get("incident_id") == incident_id for item in known_incidents):
            _status, incident = request_json(
                "POST",
                f"{arguments.data_url}/datasets/{dataset_id}/versions/{version}/incidents",
                {
                    "incident_id": incident_id,
                    "start_time": injection,
                    "end_time": end,
                    "scenario": workload(row["fault"])["anomaly_scenario"],
                    "affected_services": [row["root_cause_service"]],
                    "affected_metrics": [],
                    "annotation": f"Published RCAEval {row['fault']} injection interval.",
                    "source": "RCAEval cases.parquet and inject_time.txt",
                },
            )
        else:
            incident = known_incidents[0]
        state["datasets"][case] = {
            "dataset_id": dataset_id,
            "version": version,
            "partition_id": partition_id,
            "root_cause_service": row["root_cause_service"],
            "fault": row["fault"],
            "repetition": int(row["repetition"]),
            "import_seconds": import_seconds,
            "observation_count": detail["version"]["technical_schema"]["observation_count"],
            "feature_count": detail["version"]["technical_schema"]["feature_count"],
            "incident": incident,
        }
        write_json(state_path, state)
        print(f"Imported {index:02d}/{len(selected)} {case}", flush=True)

    base_case = selected[0]["case"]
    base = state["datasets"][base_case]
    if state["training"] is None:
        training_payload = {
            "dataset": {
                "source": "catalogue",
                "dataset_id": base["dataset_id"],
                "version": base["version"],
                "partition_id": base["partition_id"],
            },
            "detectors": [
                {"detector_id": "isolation-forest", "parameters": {}},
                {"detector_id": "local-outlier-factor", "parameters": {}},
            ],
            "client_request_id": "rcaeval-re1-tt-base-models-v1",
            "run_name": "RCAEval RE1 Train Ticket external validation base models",
        }
        _status, training = request_json("POST", f"{arguments.learning_url}/training_runs", training_payload)
        state["training"] = {"run_id": training["run_id"], "submitted_at": utc_now()}
        write_json(state_path, state)
    run_id = state["training"]["run_id"]
    training = wait_for(f"{arguments.learning_url}/training_runs/{run_id}", TERMINAL_RUN)
    state["training"]["result"] = training
    state["training"]["model_ids"] = [
        child["model_id"] for child in training["children"] if child.get("status") == "completed"
    ]
    write_json(state_path, state)
    if len(state["training"]["model_ids"]) != 2:
        raise RuntimeError(f"Expected two completed models, got {state['training']['model_ids']}")

    model_ids = state["training"]["model_ids"]
    for index, row in enumerate(selected, start=1):
        case = row["case"]
        if case in state["comparisons"] and state["comparisons"][case].get("result"):
            continue
        dataset = state["datasets"][case]
        payload = {
            "name": f"RCAEval context | {row['root_cause_service']} | {row['fault']} | rep {row['repetition']}",
            "evaluation_dataset": {
                "dataset_id": dataset["dataset_id"],
                "version": dataset["version"],
                "partition_id": dataset["partition_id"],
            },
            "model_ids": model_ids,
            "client_request_id": f"rcaeval-re1-tt-comparison-v1-{case}",
        }
        _status, comparison = request_json("POST", f"{arguments.learning_url}/api/comparisons", payload)
        comparison_id = comparison["comparison_id"]
        state["comparisons"][case] = {"comparison_id": comparison_id, "submitted_at": utc_now()}
        write_json(state_path, state)
        result = wait_for(
            f"{arguments.learning_url}/api/comparisons/{comparison_id}", TERMINAL_COMPARISON
        )
        state["comparisons"][case]["result"] = result
        write_json(state_path, state)
        print(f"Compared {index:02d}/{len(selected)} {case}: {result['status']}", flush=True)

    final_comparison = state["comparisons"][selected[-1]["case"]]["comparison_id"]
    _status, robustness = request_json(
        "GET", f"{arguments.learning_url}/api/comparisons/{final_comparison}/robustness"
    )
    state["robustness"] = robustness
    state["completed_at"] = utc_now()
    write_json(state_path, state)
    print(json.dumps({
        "datasets": len(state["datasets"]),
        "training_status": training["status"],
        "models": model_ids,
        "comparisons": len(state["comparisons"]),
        "completed_at": state["completed_at"],
    }, indent=2))


if __name__ == "__main__":
    main()
