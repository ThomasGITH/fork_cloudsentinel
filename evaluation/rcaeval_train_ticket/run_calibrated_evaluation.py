#!/usr/bin/env python3
"""Run the fixed five-case calibrated AD evaluation through CloudSentinel."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


TERMINAL_RUN = {"completed", "partial_success", "failed"}
TERMINAL_COMPARISON = {"completed", "partial", "failed"}


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def request_json(method: str, url: str, payload: object | None = None, timeout: int = 180):
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
        raise RuntimeError(f"{method} {url} returned HTTP {exc.code}: {body}") from exc


def wait_for(url: str, terminal: set[str], maximum_seconds: int = 7200):
    deadline = time.monotonic() + maximum_seconds
    last = None
    while True:
        _status, value = request_json("GET", url)
        status = value.get("status")
        progress = tuple(
            (child.get("detector_id"), child.get("status"), child.get("progress", {}).get("detail"))
            for child in value.get("children", [])
        )
        current = (status, progress)
        if current != last:
            print(json.dumps({"status": status, "children": progress}), flush=True)
            last = current
        if status in terminal:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {url}; last status: {status}")
        time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-state", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--learning-url", default="http://127.0.0.1:18002")
    arguments = parser.parse_args()

    original = json.loads(arguments.original_state.read_text(encoding="utf-8"))
    prepared = arguments.prepared_dir.resolve()
    evidence = prepared / "evidence"
    with (evidence / "pairs.csv").open("r", encoding="utf-8", newline="") as source:
        pairs = list(csv.DictReader(source))
    test_rows = [row for row in pairs if row["role"] == "test"]
    selection = json.loads((evidence / "parameter_selection.json").read_text(encoding="utf-8"))
    protocol = json.loads((evidence / "protocol.json").read_text(encoding="utf-8"))
    state_path = arguments.state.resolve()
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {
        "protocol_version": protocol["protocol_version"],
        "started_at": utc_now(),
        "training_runs": {},
        "comparisons": {},
    }

    for index, row in enumerate(test_rows, start=1):
        case = row["case"]
        pair_id = row["pair_id"]
        dataset = original["datasets"][case]
        if_parameters = dict(selection[pair_id]["isolation-forest"]["selected"]["parameters"])
        lof_parameters = dict(selection[pair_id]["local-outlier-factor"]["selected"]["parameters"])
        for parameters in (if_parameters, lof_parameters):
            if parameters.get("contamination") != "auto":
                parameters["contamination"] = f"{float(parameters['contamination']):.2f}"

        if case not in state["training_runs"]:
            payload = {
                "dataset": {
                    "source": "catalogue",
                    "dataset_id": dataset["dataset_id"],
                    "version": dataset["version"],
                    "partition_id": dataset["partition_id"],
                },
                "detectors": [
                    {"detector_id": "isolation-forest", "parameters": if_parameters},
                    {"detector_id": "local-outlier-factor", "parameters": lof_parameters},
                    {"detector_id": "cgnn", "parameters": dict(protocol["cgnn_parameters"])},
                ],
                "client_request_id": f"rcaeval-calibrated-v4-{case}",
                "run_name": f"RCAEval calibrated temporal hold-out | {pair_id}",
            }
            _status, submitted = request_json(
                "POST", f"{arguments.learning_url}/training_runs", payload
            )
            state["training_runs"][case] = {
                "run_id": submitted["run_id"],
                "submitted_at": utc_now(),
                "parameters": {
                    "isolation-forest": if_parameters,
                    "local-outlier-factor": lof_parameters,
                    "cgnn": protocol["cgnn_parameters"],
                },
            }
            write_json(state_path, state)

        run = state["training_runs"][case]
        result = wait_for(f"{arguments.learning_url}/training_runs/{run['run_id']}", TERMINAL_RUN)
        run["result"] = result
        model_ids = [
            child["model_id"]
            for child in result.get("children", [])
            if child.get("status") == "completed" and child.get("model_status") == "available"
        ]
        run["model_ids"] = model_ids
        write_json(state_path, state)
        if len(model_ids) != 3:
            raise RuntimeError(f"{case}: expected three completed models, got {model_ids}")

        if case not in state["comparisons"]:
            payload = {
                "name": f"RCAEval calibrated test | {pair_id}",
                "evaluation_dataset": {
                    "dataset_id": dataset["dataset_id"],
                    "version": dataset["version"],
                    "partition_id": dataset["partition_id"],
                },
                "model_ids": model_ids,
                "client_request_id": f"rcaeval-calibrated-comparison-v4-{case}",
            }
            _status, submitted = request_json(
                "POST", f"{arguments.learning_url}/api/comparisons", payload
            )
            state["comparisons"][case] = {
                "comparison_id": submitted["comparison_id"],
                "submitted_at": utc_now(),
            }
            write_json(state_path, state)
        comparison = state["comparisons"][case]
        comparison["result"] = wait_for(
            f"{arguments.learning_url}/api/comparisons/{comparison['comparison_id']}",
            TERMINAL_COMPARISON,
        )
        write_json(state_path, state)
        print(f"Finished {index}/{len(test_rows)} {case}", flush=True)

    state["completed_at"] = utc_now()
    write_json(state_path, state)


if __name__ == "__main__":
    main()
