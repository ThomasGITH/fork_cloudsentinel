#!/usr/bin/env python3
"""Repeat comparisons after all detector runtimes have been aligned."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


TERMINAL = {"completed", "partial", "failed"}


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def request_json(method: str, url: str, payload: object | None = None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=180) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} returned HTTP {exc.code}: {body}") from exc


def wait_for(url: str, maximum_seconds: int = 1800) -> dict:
    deadline = time.monotonic() + maximum_seconds
    last = None
    while True:
        value = request_json("GET", url)
        status = value.get("status")
        if status != last:
            print(json.dumps({"status": status}), flush=True)
            last = status
        if status in TERMINAL:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {url}; last status: {status}")
        time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-state", type=Path, required=True)
    parser.add_argument("--original-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt", default="v5")
    parser.add_argument("--learning-url", default="http://127.0.0.1:18002")
    arguments = parser.parse_args()

    run_state = json.loads(arguments.run_state.read_text(encoding="utf-8"))
    original = json.loads(arguments.original_state.read_text(encoding="utf-8"))
    output = (
        json.loads(arguments.output.read_text(encoding="utf-8"))
        if arguments.output.is_file()
        else {"attempt": arguments.attempt, "started_at": utc_now(), "comparisons": {}}
    )

    for case, run in run_state["training_runs"].items():
        result = run.get("result") or {}
        if result.get("status") != "completed" or len(run.get("model_ids") or []) != 3:
            raise RuntimeError(f"{case}: training run is not complete with three models")
        dataset = original["datasets"][case]
        if case not in output["comparisons"]:
            submitted = request_json(
                "POST",
                f"{arguments.learning_url}/api/comparisons",
                {
                    "name": f"RCAEval calibrated test | {case} | aligned runtime",
                    "evaluation_dataset": {
                        "dataset_id": dataset["dataset_id"],
                        "version": dataset["version"],
                        "partition_id": dataset["partition_id"],
                    },
                    "model_ids": run["model_ids"],
                    "client_request_id": (
                        f"rcaeval-calibrated-comparison-{arguments.attempt}-{case}"
                    ),
                },
            )
            output["comparisons"][case] = {
                "comparison_id": submitted["comparison_id"],
                "submitted_at": utc_now(),
            }
            write_json(arguments.output, output)
        comparison = output["comparisons"][case]
        comparison["result"] = wait_for(
            f"{arguments.learning_url}/api/comparisons/{comparison['comparison_id']}"
        )
        write_json(arguments.output, output)
        print(f"Finished {case}", flush=True)

    output["completed_at"] = utc_now()
    write_json(arguments.output, output)


if __name__ == "__main__":
    main()
