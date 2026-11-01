#!/usr/bin/env python3
"""Recalculate comparison metrics without importing CloudSentinel code.

Usage:
    python independent_score_check.py labels.csv comparison-detail.json output.json

The comparison detail is the public JSON returned by
GET /api/comparisons/<comparison_id>. The public timeline contains every
prediction for the evaluated dataset in this evaluation (921 points maximum,
below the configured 1000-point bound). labels.csv is the independently
downloaded evaluation-bundle label vector before detector warm-up exclusion.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


def calculate(labels: list[int], predictions: list[int]) -> dict[str, object]:
    if len(labels) != len(predictions):
        raise ValueError("label and prediction counts differ")
    tp = sum(a == p == 1 for a, p in zip(labels, predictions))
    tn = sum(a == p == 0 for a, p in zip(labels, predictions))
    fp = sum(a == 0 and p == 1 for a, p in zip(labels, predictions))
    fn = sum(a == 1 and p == 0 for a, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_score = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def main() -> None:
    labels_path, comparison_path, output_path = map(Path, sys.argv[1:4])
    with labels_path.open(newline="", encoding="utf-8") as source:
        all_labels = [int(row[0]) for row in csv.reader(source)]
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    report: dict[str, object] = {"comparison_id": comparison["comparison_id"], "models": []}
    for result in comparison["results"]:
        predictions = [int(point["prediction"]) for point in result["timeline"]]
        warmup = int(result["observation_count"]) - int(result["prediction_count"])
        labels = all_labels[warmup:]
        independent = calculate(labels, predictions)
        application = result.get("metrics", {})
        report["models"].append(
            {
                "model_id": result["model_id"],
                "detector_id": result["detector_id"],
                "warmup_observations": warmup,
                "label_count": len(labels),
                "prediction_count": len(predictions),
                "application": application,
                "independent": independent,
                "exact_match": application == independent,
            }
        )
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

