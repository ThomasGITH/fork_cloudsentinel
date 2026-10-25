#!/usr/bin/env python3
"""Create compact tables from the retained deployed-CloudSentinel API record."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics
import csv


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarise(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "contexts": len(rows),
        "precision": distribution([float(row["precision"]) for row in rows]),
        "recall": distribution([float(row["recall"]) for row in rows]),
        "f1": distribution([float(row["f1"]) for row in rows]),
        "runtime_ms": distribution([float(row["runtime_ms"]) for row in rows]),
        "predicted_anomalies": distribution([float(row["predicted_anomalies"]) for row in rows]),
        "predicted_anomaly_fraction": distribution(
            [float(row["predicted_anomaly_fraction"]) for row in rows]
        ),
        "false_positive_rate": distribution([float(row["false_positive_rate"]) for row in rows]),
        "balanced_accuracy": distribution([float(row["balanced_accuracy"]) for row in rows]),
        "lead_time_statuses": dict(Counter(str(row["lead_time_status"]) for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    arguments = parser.parse_args()
    record = json.loads(arguments.record.read_text(encoding="utf-8"))
    datasets = record["datasets"]
    comparisons = record["comparisons"]
    rows = []
    hashes_by_model: dict[str, set[str]] = defaultdict(set)
    feature_hashes = set()
    winner_counts = Counter()
    for case, stored in comparisons.items():
        comparison = stored["result"]
        context = datasets[case]
        feature_hashes.add(comparison["feature_identity"]["sha256"])
        succeeded = [item for item in comparison["results"] if item["status"] == "completed"]
        if len(succeeded) != 2:
            raise RuntimeError(f"{case} did not retain two successful model results")
        best_f1 = max(item["metrics"]["f1_score"] for item in succeeded)
        for item in succeeded:
            detector = item["detector_id"]
            hashes_by_model[item["model_id"]].add(item["artifact_identity"]["artifact_manifest_sha256"])
            if item["metrics"]["f1_score"] == best_f1:
                winner_counts[detector] += 1
            matrix = item["metrics"]["confusion_matrix"]
            false_positive_rate = matrix["fp"] / (matrix["fp"] + matrix["tn"])
            true_negative_rate = 1.0 - false_positive_rate
            recall = item["metrics"]["recall"]
            rows.append({
                "case": case,
                "root_cause_service": context["root_cause_service"],
                "fault": context["fault"],
                "repetition": context["repetition"],
                "detector_id": detector,
                "model_id": item["model_id"],
                "artifact_manifest_sha256": item["artifact_identity"]["artifact_manifest_sha256"],
                "precision": item["metrics"]["precision"],
                "recall": item["metrics"]["recall"],
                "f1": item["metrics"]["f1_score"],
                "false_positive_rate": false_positive_rate,
                "balanced_accuracy": (recall + true_negative_rate) / 2.0,
                "tp": matrix["tp"],
                "fp": matrix["fp"],
                "tn": matrix["tn"],
                "fn": matrix["fn"],
                "input_observations": item["observation_count"],
                "predicted_anomalies": item["anomaly_count"],
                "predicted_anomaly_fraction": item["anomaly_count"] / item["prediction_count"],
                "runtime_ms": item["runtime_ms"],
                "lead_time_status": item["lead_time_status"],
                "lead_time_seconds": item["lead_time_seconds"],
            })
    if len(feature_hashes) != 1:
        raise RuntimeError(f"Expected one fixed feature identity, found {feature_hashes}")
    if any(len(hashes) != 1 for hashes in hashes_by_model.values()):
        raise RuntimeError("A model artifact identity changed across contexts")

    target_csv = arguments.record.with_name("cloudsentinel_per_context.csv")
    with target_csv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    training = record["training"]["result"]
    children = {}
    for child in training["children"]:
        children[child["detector_id"]] = {
            "status": child["status"],
            "model_id": child["model_id"],
            "duration_seconds": (
                parse_time(child["completed_at"]) - parse_time(child["started_at"])
            ).total_seconds(),
            "model_status": child["model_status"],
        }
    import_times = [
        float(item["import_seconds"])
        for item in datasets.values()
        if item.get("import_seconds") is not None
    ]
    by_detector = {}
    for detector in sorted({row["detector_id"] for row in rows}):
        detector_rows = [row for row in rows if row["detector_id"] == detector]
        by_detector[detector] = {
            "overall": summarise(detector_rows),
            "by_fault": {
                fault: summarise([row for row in detector_rows if row["fault"] == fault])
                for fault in sorted({row["fault"] for row in detector_rows})
            },
        }
    aggregate = {
        "protocol_version": record["protocol_version"],
        "completed_at": record["completed_at"],
        "dataset_import": {
            "successful_datasets": len(datasets),
            "feature_counts": dict(Counter(str(item["feature_count"]) for item in datasets.values())),
            "observation_counts": dict(Counter(str(item["observation_count"]) for item in datasets.values())),
            "import_seconds": distribution(import_times),
        },
        "training": {
            "status": training["status"],
            "source_case": next(iter(datasets)),
            "children": children,
            "snapshot_id": training["snapshot_id"],
            "snapshot_sha256": training["snapshot_sha256"],
            "feature_order_sha256": training["dataset"]["feature_order_sha256"],
        },
        "comparison": {
            "successful_contexts": sum(
                item["result"]["status"] == "completed" for item in comparisons.values()
            ),
            "context_count": len(comparisons),
            "fixed_feature_order_sha256": next(iter(feature_hashes)),
            "model_artifact_hash_count": {
                model: len(hashes) for model, hashes in hashes_by_model.items()
            },
            "winner_or_tie_counts": dict(winner_counts),
            "by_detector": by_detector,
        },
        "robustness": {
            "eligible_comparison_count": record["robustness"]["eligible_comparison_count"],
            "coverage_warnings": record["robustness"]["coverage_warnings"],
            "model_insights": record["robustness"]["model_insights"],
        },
        "evidence_boundary": (
            "The two immutable models were fitted once on the first case and evaluated unchanged "
            "on all 25 contexts. This tests historical data and artifact portability, not live "
            "connection to a second cluster. The 100-feature adaptation remains part of the result."
        ),
    }
    write_json(arguments.record.with_name("cloudsentinel_aggregate.json"), aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
