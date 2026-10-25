#!/usr/bin/env python3
"""Evaluate BARO RobustScorer rankings on the frozen Train Ticket sample."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from baro.root_cause_analysis import robust_scorer
import numpy as np
import pandas as pd


SUFFIXES = ("latency-50", "latency-90", "workload", "error", "cpu", "mem")


def service_of(feature: str) -> str:
    for suffix in SUFFIXES:
        marker = f"_{suffix}"
        if feature.endswith(marker):
            return feature[: -len(marker)]
    return feature


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", type=Path, default=Path("prepared"))
    arguments = parser.parse_args()
    prepared = arguments.prepared_dir.resolve()
    evidence_dir = prepared / "evidence"
    selected = pd.read_csv(evidence_dir / "selected_cases.csv")

    results = []
    for row in selected.to_dict(orient="records"):
        case = row["case"]
        frame = pd.read_csv(prepared / "canonical" / f"{case}.csv")
        labels = frame.pop("label").to_numpy(dtype=int)
        frame.pop("timestamp")
        anomaly_indices = np.flatnonzero(labels == 1)
        if len(anomaly_indices) == 0:
            raise RuntimeError(f"{case} has no labelled injection point")
        change_point = int(anomaly_indices[0])
        started = time.perf_counter()
        output = robust_scorer(frame, anomalies=[change_point], dk_select_useful=False)
        elapsed = time.perf_counter() - started
        feature_ranks = output.get("ranks", [])
        if not feature_ranks:
            raise RuntimeError(f"BARO returned no ranking for {case}")
        service_ranks = []
        for feature in feature_ranks:
            service = service_of(feature)
            if service not in service_ranks:
                service_ranks.append(service)
        root = row["root_cause_service"]
        service_rank = service_ranks.index(root) + 1 if root in service_ranks else None
        root_feature_ranks = [
            position
            for position, feature in enumerate(feature_ranks, start=1)
            if service_of(feature) == root
        ]
        result = {
            "case": case,
            "root_cause_service": root,
            "fault": row["fault"],
            "repetition": int(row["repetition"]),
            "change_point_index": change_point,
            "change_point_source": "RCAEval published inject_time converted to the canonical row index",
            "runtime_seconds": elapsed,
            "root_service_rank": service_rank,
            "reciprocal_rank": 0.0 if service_rank is None else 1.0 / service_rank,
            "service_top_1": bool(service_rank is not None and service_rank <= 1),
            "service_top_3": bool(service_rank is not None and service_rank <= 3),
            "service_top_5": bool(service_rank is not None and service_rank <= 5),
            "best_root_feature_rank": min(root_feature_ranks) if root_feature_ranks else None,
            "top_10_features": feature_ranks[:10],
            "top_10_services": service_ranks[:10],
        }
        results.append(result)
        write_json(evidence_dir / "baro" / f"{case}.json", result)

    def summary(items: list[dict[str, object]]) -> dict[str, object]:
        present_ranks = [item["root_service_rank"] for item in items if item["root_service_rank"]]
        return {
            "cases": len(items),
            "service_top_1_rate": float(np.mean([item["service_top_1"] for item in items])),
            "service_top_3_rate": float(np.mean([item["service_top_3"] for item in items])),
            "service_top_5_rate": float(np.mean([item["service_top_5"] for item in items])),
            "mean_reciprocal_rank": float(np.mean([item["reciprocal_rank"] for item in items])),
            "median_root_service_rank": float(np.median(present_ranks)) if present_ranks else None,
            "mean_runtime_seconds": float(np.mean([item["runtime_seconds"] for item in items])),
        }

    aggregate = {
        "method": "fse-baro RobustScorer 0.2.2",
        "protocol_version": "cloudsentinel-rcaeval-train-ticket/v1",
        "created_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "change_point_source": "published benchmark injection time",
        "overall": summary(results),
        "by_fault": {
            fault: summary([item for item in results if item["fault"] == fault])
            for fault in sorted(Counter(item["fault"] for item in results))
        },
        "by_root_service": {
            service: summary([item for item in results if item["root_cause_service"] == service])
            for service in sorted(Counter(item["root_cause_service"] for item in results))
        },
        "evidence_boundary": (
            "The published injection time is supplied to RobustScorer. These results evaluate "
            "ranking conditional on a known change point, not anomaly or change-point detection."
        ),
    }
    write_json(evidence_dir / "baro_aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
