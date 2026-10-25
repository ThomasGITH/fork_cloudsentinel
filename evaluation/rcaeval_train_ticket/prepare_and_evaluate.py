#!/usr/bin/env python3
"""Prepare a frozen RCAEval Train Ticket sample for CloudSentinel evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import LocalOutlierFactor


BASE_URL = "https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/main"
INDEX_URL = f"{BASE_URL}/cases.parquet"
ROOT_SERVICES = [
    "ts-auth-service",
    "ts-order-service",
    "ts-route-service",
    "ts-train-service",
    "ts-travel-service",
]
FAULTS = ["cpu", "delay", "disk", "loss", "mem"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download(url: str, target: Path) -> None:
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    temporary.replace(target)


def chosen_cases(index: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for service_index, service in enumerate(ROOT_SERVICES):
        for fault_index, fault in enumerate(FAULTS):
            repetition = ((service_index + fault_index) % 5) + 1
            matches = index[
                (index["suite"] == "RE1")
                & (index["system"] == "tt")
                & (index["root_cause_service"] == service)
                & (index["fault"] == fault)
                & (index["repetition"] == repetition)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one case for {service}/{fault}/rep {repetition}, got {len(matches)}"
                )
            selected.append(matches.iloc[0])
    return pd.DataFrame(selected).reset_index(drop=True)


def feature_parts(name: str) -> tuple[str, str]:
    suffixes = ("latency-50", "latency-90", "workload", "error", "cpu", "mem")
    for suffix in suffixes:
        marker = f"_{suffix}"
        if name.endswith(marker):
            return name[: -len(marker)], suffix
    return name, "other"


def feature_priority(name: str) -> tuple[int, int, str, str]:
    entity, family = feature_parts(name)
    root = entity in ROOT_SERVICES
    database = entity.endswith("-mongo") or entity.endswith("-mysql")
    family_order = {
        "workload": 0,
        "latency-50": 1,
        "latency-90": 2,
        "cpu": 3,
        "mem": 4,
        "error": 5,
        "other": 6,
    }
    if root:
        group = 0
    elif not database and family in {"workload", "latency-50", "latency-90"}:
        group = 1
    elif not database and family == "error":
        group = 2
    elif not database and family in {"cpu", "mem"}:
        group = 3
    else:
        group = 4
    return group, family_order[family], entity, name


def diagnostic_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_anomalies": int(predictions.sum()),
        "test_observations": int(len(labels)),
        "actual_anomalies": int(labels.sum()),
    }


def run_diagnostics(frame: pd.DataFrame, inject_time: int, features: list[str]) -> dict[str, object]:
    split_index = max(2, min(len(frame) - 2, round(len(frame) * 0.30)))
    train = frame.iloc[:split_index][features].to_numpy(dtype=float)
    test = frame.iloc[split_index:][features].to_numpy(dtype=float)
    labels = (frame.iloc[split_index:]["time"].to_numpy(dtype=int) >= inject_time).astype(int)
    started = time.perf_counter()
    isolation_forest = IsolationForest(
        n_estimators=100, contamination="auto", random_state=42, n_jobs=1
    ).fit(train)
    fit_if = time.perf_counter() - started
    started = time.perf_counter()
    if_predictions = (isolation_forest.predict(test) == -1).astype(int)
    predict_if = time.perf_counter() - started

    started = time.perf_counter()
    lof = LocalOutlierFactor(n_neighbors=20, contamination="auto", novelty=True, n_jobs=1).fit(train)
    fit_lof = time.perf_counter() - started
    started = time.perf_counter()
    lof_predictions = (lof.predict(test) == -1).astype(int)
    predict_lof = time.perf_counter() - started
    return {
        "split_index": split_index,
        "train_observations": int(len(train)),
        "normal_test_observations": int((labels == 0).sum()),
        "anomalous_test_observations": int(labels.sum()),
        "isolation_forest": {
            **diagnostic_metrics(labels, if_predictions),
            "fit_seconds": fit_if,
            "predict_seconds": predict_if,
        },
        "local_outlier_factor": {
            **diagnostic_metrics(labels, lof_predictions),
            "fit_seconds": fit_lof,
            "predict_seconds": predict_lof,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("prepared"))
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    raw = output / "raw"
    canonical = output / "canonical"
    evidence = output / "evidence"
    for directory in (raw, canonical, evidence):
        directory.mkdir(parents=True, exist_ok=True)

    index_path = raw / "cases.parquet"
    download(INDEX_URL, index_path)
    index = pd.read_parquet(index_path)
    selection = chosen_cases(index)
    selection.to_csv(evidence / "selected_cases.csv", index=False)

    frames: dict[str, pd.DataFrame] = {}
    source_hashes: dict[str, object] = {
        "cases.parquet": {"sha256": sha256(index_path), "source": INDEX_URL}
    }
    shared_features: set[str] | None = None
    for row in selection.to_dict(orient="records"):
        case = row["case"]
        case_dir = raw / case
        metrics_path = case_dir / "metrics.parquet"
        inject_path = case_dir / "inject_time.txt"
        metrics_url = f"{BASE_URL}/{case}/metrics.parquet"
        inject_url = f"{BASE_URL}/{case}/inject_time.txt"
        download(metrics_url, metrics_path)
        download(inject_url, inject_path)
        frame = pd.read_parquet(metrics_path)
        if "time" not in frame or frame["time"].duplicated().any() or not frame["time"].is_monotonic_increasing:
            raise RuntimeError(f"{case} has an invalid time column")
        frames[case] = frame
        columns = set(frame.columns) - {"time"}
        shared_features = columns if shared_features is None else shared_features & columns
        source_hashes[case] = {
            "metrics_sha256": sha256(metrics_path),
            "inject_time_sha256": sha256(inject_path),
            "metrics_source": metrics_url,
            "inject_time_source": inject_url,
        }

    eligible_features = {
        feature
        for feature in (shared_features or ())
        if all(
            np.isfinite(frame[feature].dropna().to_numpy(dtype=float)).any()
            for frame in frames.values()
        )
    }
    features = sorted(eligible_features, key=feature_priority)[:100]
    if len(features) != 100:
        raise RuntimeError(f"Expected at least 100 shared metrics, found {len(features)}")
    write_json(
        evidence / "feature_profile.json",
        {
            "profile_version": "rcaeval-re1-tt-shared-100/v1",
            "shared_feature_count_before_limit": len(shared_features or ()),
            "eligible_feature_count_before_limit": len(eligible_features),
            "selected_feature_count": len(features),
            "features": features,
            "families": dict(Counter(feature_parts(item)[1] for item in features)),
            "selection_rule": "feature_priority in prepare_and_evaluate.py; lexical tie-break",
        },
    )
    write_json(evidence / "source_hashes.json", source_hashes)

    case_evidence = []
    for row in selection.to_dict(orient="records"):
        case = row["case"]
        frame = frames[case].copy()
        inject_time = int(row["inject_time"])
        missing_before = int(frame[features].isna().sum().sum())
        frame[features] = frame[features].interpolate(method="linear", limit_direction="both").ffill().bfill()
        if frame[features].isna().any().any():
            raise RuntimeError(f"{case} still contains missing selected values after the frozen rule")
        timestamps = [
            datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat().replace("+00:00", "Z")
            for value in frame["time"]
        ]
        labels = (frame["time"].to_numpy(dtype=int) >= inject_time).astype(int)
        prepared = frame[features].copy()
        prepared.insert(0, "timestamp", timestamps)
        prepared["label"] = labels
        target = canonical / f"{case}.csv"
        prepared.to_csv(target, index=False, float_format="%.12g")
        diagnostics = run_diagnostics(frame, inject_time, features)
        evidence_item = {
            "case": case,
            "root_cause_service": row["root_cause_service"],
            "fault": row["fault"],
            "repetition": int(row["repetition"]),
            "inject_time": inject_time,
            "observations": int(len(frame)),
            "source_feature_count": int(len(frame.columns) - 1),
            "selected_feature_count": len(features),
            "missing_selected_values_before_imputation": missing_before,
            "canonical_csv_sha256": sha256(target),
            "diagnostics": diagnostics,
        }
        case_evidence.append(evidence_item)
        write_json(evidence / "cases" / f"{case}.json", evidence_item)

    aggregate: dict[str, object] = {
        "protocol_version": "cloudsentinel-rcaeval-train-ticket/v1",
        "created_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "case_count": len(case_evidence),
        "feature_count": len(features),
        "fault_counts": dict(Counter(item["fault"] for item in case_evidence)),
        "root_service_counts": dict(Counter(item["root_cause_service"] for item in case_evidence)),
        "repetition_counts": dict(Counter(str(item["repetition"]) for item in case_evidence)),
    }
    for method in ("isolation_forest", "local_outlier_factor"):
        values = [item["diagnostics"][method] for item in case_evidence]
        aggregate[method] = {
            metric: {
                "mean": float(np.mean([value[metric] for value in values])),
                "median": float(np.median([value[metric] for value in values])),
                "minimum": float(np.min([value[metric] for value in values])),
                "maximum": float(np.max([value[metric] for value in values])),
            }
            for metric in ("precision", "recall", "f1", "fit_seconds", "predict_seconds")
        }
    write_json(evidence / "aggregate_diagnostics.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
