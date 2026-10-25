#!/usr/bin/env python3
"""Create compact, auditable tables from the aligned RCAEval comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def division(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def finite_mean(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None and math.isfinite(value)]
    return mean(available) if available else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparisons", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    source = json.loads(arguments.comparisons.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for case, comparison in source["comparisons"].items():
        result = comparison["result"]
        for model in result.get("results", []):
            metrics = model.get("metrics") or {}
            confusion = metrics.get("confusion_matrix") or {}
            tp = int(confusion.get("tp", 0))
            tn = int(confusion.get("tn", 0))
            fp = int(confusion.get("fp", 0))
            fn = int(confusion.get("fn", 0))
            tpr = division(tp, tp + fn)
            tnr = division(tn, tn + fp)
            rows.append(
                {
                    "case": case,
                    "detector_id": model.get("detector_id"),
                    "status": model.get("status"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1_score": metrics.get("f1_score"),
                    "false_positive_rate": division(fp, fp + tn),
                    "balanced_accuracy": (
                        (tpr + tnr) / 2 if tpr is not None and tnr is not None else None
                    ),
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "prediction_count": model.get("prediction_count"),
                    "anomaly_count": model.get("anomaly_count"),
                    "runtime_ms": model.get("runtime_ms"),
                    "safe_error_summary": model.get("safe_error_summary"),
                }
            )

    detector_ids = sorted({row["detector_id"] for row in rows if row["detector_id"]})
    aggregate = []
    for detector_id in detector_ids:
        selected = [row for row in rows if row["detector_id"] == detector_id]
        aggregate.append(
            {
                "detector_id": detector_id,
                "completed_cases": sum(row["status"] == "completed" for row in selected),
                "total_cases": len(selected),
                **{
                    f"mean_{name}": finite_mean([row[name] for row in selected])
                    for name in (
                        "precision",
                        "recall",
                        "f1_score",
                        "false_positive_rate",
                        "balanced_accuracy",
                        "runtime_ms",
                    )
                },
            }
        )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with (arguments.output_dir / "per_case_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (arguments.output_dir / "summary.json").write_text(
        json.dumps({"per_case": rows, "aggregate": aggregate}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
