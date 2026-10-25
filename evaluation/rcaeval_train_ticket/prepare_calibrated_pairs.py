#!/usr/bin/env python3
"""Prepare a leakage-controlled, context-calibrated RCAEval evaluation.

Five service/fault combinations are used.  For each combination, one repetition
is used for fitting and validation and a different repetition is retained as an
untouched test case.  Detector parameters are selected only on the validation
repetition; the held-out repetition is inspected only after selection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_URL = "https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/main"
PAIRS = (
    ("ts-auth-service", "cpu", 2, 1),
    ("ts-order-service", "delay", 4, 3),
    ("ts-route-service", "disk", 1, 5),
    ("ts-train-service", "loss", 3, 2),
    ("ts-travel-service", "mem", 5, 4),
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download(url: str, target: Path) -> None:
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    with urlopen(url, timeout=180) as response, temporary.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    temporary.replace(target)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measures(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(1.0 - specificity),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def matrix(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = max(2, min(len(frame) - 2, round(len(frame) * 0.30)))
    train = frame.iloc[:split][features].to_numpy(dtype=float)
    evaluate = frame.iloc[split:][features].to_numpy(dtype=float)
    labels = frame.iloc[split:]["label"].to_numpy(dtype=np.int8)
    return train, evaluate, labels


def if_pipeline(parameters: dict[str, object]) -> Pipeline:
    return Pipeline((
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(**parameters)),
    ))


def lof_pipeline(parameters: dict[str, object]) -> Pipeline:
    return Pipeline((
        ("scaler", StandardScaler()),
        ("detector", LocalOutlierFactor(novelty=True, **parameters)),
    ))


def select_parameters(
    train: np.ndarray, validation: np.ndarray, labels: np.ndarray
) -> dict[str, object]:
    candidates: dict[str, list[dict[str, object]]] = {
        "isolation-forest": [
            {
                "n_estimators": 300,
                "max_samples": "auto",
                "contamination": contamination,
                "max_features": fraction,
                "bootstrap": bootstrap,
                "random_state": 42,
                "n_jobs": 1,
            }
            for contamination in ("auto", 0.01, 0.02, 0.05)
            for fraction in (0.5, 0.75, 1.0)
            for bootstrap in (False, True)
        ],
        "local-outlier-factor": [
            {
                "n_neighbors": neighbors,
                "metric": metric,
                "contamination": contamination,
                "algorithm": "auto",
                "leaf_size": 30,
                "p": 2,
            }
            for contamination in ("auto", 0.01, 0.02, 0.05)
            for neighbors in (10, 20, 35, 50)
            for metric in ("minkowski", "manhattan")
        ],
    }
    selected: dict[str, object] = {}
    for detector, parameter_sets in candidates.items():
        scored = []
        for parameters in parameter_sets:
            pipeline = if_pipeline(parameters) if detector == "isolation-forest" else lof_pipeline(parameters)
            pipeline.fit(train)
            predictions = (pipeline.predict(validation) == -1).astype(np.int8)
            result = measures(labels, predictions)
            scored.append({"parameters": parameters, "metrics": result})
        scored.sort(
            key=lambda item: (
                item["metrics"]["balanced_accuracy"],
                item["metrics"]["f1"],
                -item["metrics"]["false_positive_rate"],
            ),
            reverse=True,
        )
        selected[detector] = {"selected": scored[0], "candidates": scored}
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", type=Path, required=True)
    arguments = parser.parse_args()
    original = arguments.prepared_dir.resolve()
    output = original / "calibrated"
    raw = output / "raw"
    canonical = output / "canonical"
    evidence = output / "evidence"
    for directory in (raw, canonical, evidence):
        directory.mkdir(parents=True, exist_ok=True)

    features = json.loads((original / "evidence" / "feature_profile.json").read_text())["features"]
    index = pd.read_parquet(original / "raw" / "cases.parquet")
    rows = []
    hashes: dict[str, object] = {}
    frames: dict[str, pd.DataFrame] = {}
    for service, fault, validation_rep, test_rep in PAIRS:
        for role, repetition in (("validation", validation_rep), ("test", test_rep)):
            match = index[
                (index["suite"] == "RE1")
                & (index["system"] == "tt")
                & (index["root_cause_service"] == service)
                & (index["fault"] == fault)
                & (index["repetition"] == repetition)
            ]
            if len(match) != 1:
                raise RuntimeError(f"Expected one {service}/{fault}/{repetition} case")
            record = match.iloc[0].to_dict()
            case = record["case"]
            case_raw = raw / case
            metrics_path = case_raw / "metrics.parquet"
            injection_path = case_raw / "inject_time.txt"
            download(f"{BASE_URL}/{case}/metrics.parquet", metrics_path)
            download(f"{BASE_URL}/{case}/inject_time.txt", injection_path)
            frame = pd.read_parquet(metrics_path)
            missing = sorted(set(features).difference(frame.columns))
            if missing:
                raise RuntimeError(f"{case} misses {len(missing)} frozen features")
            frame[features] = frame[features].interpolate(method="linear", limit_direction="both").ffill().bfill()
            if frame[features].isna().any().any():
                raise RuntimeError(f"{case} still contains missing values")
            inject_time = int(record["inject_time"])
            prepared = frame[features].copy()
            prepared.insert(
                0,
                "timestamp",
                [
                    datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat().replace("+00:00", "Z")
                    for value in frame["time"]
                ],
            )
            prepared["label"] = (frame["time"].to_numpy(dtype=int) >= inject_time).astype(np.int8)
            target = canonical / f"{case}.csv"
            prepared.to_csv(target, index=False, float_format="%.12g")
            frames[case] = prepared
            hashes[case] = {
                "canonical_sha256": sha256(target),
                "metrics_sha256": sha256(metrics_path),
                "inject_time_sha256": sha256(injection_path),
            }
            rows.append({
                "pair_id": f"{service}_{fault}",
                "role": role,
                "case": case,
                "root_cause_service": service,
                "fault": fault,
                "repetition": repetition,
                "inject_time": inject_time,
            })

    pd.DataFrame(rows).to_csv(evidence / "pairs.csv", index=False)
    parameter_selection: dict[str, object] = {}
    for service, fault, validation_rep, test_rep in PAIRS:
        validation_case = f"re1tt_{service}_{fault}_{validation_rep}"
        test_case = f"re1tt_{service}_{fault}_{test_rep}"
        train, validation, labels = matrix(frames[validation_case], features)
        selection = select_parameters(train, validation, labels)
        parameter_selection[f"{service}_{fault}"] = selection

    # Select one configuration per detector across all five validation cases.
    # The held-out cases and their labels are not consulted for this choice.
    global_parameters: dict[str, object] = {}
    for detector in ("isolation-forest", "local-outlier-factor"):
        aggregates: dict[str, dict[str, object]] = {}
        for pair in parameter_selection.values():
            for candidate in pair[detector]["candidates"]:
                key = json.dumps(candidate["parameters"], sort_keys=True)
                entry = aggregates.setdefault(
                    key,
                    {"parameters": candidate["parameters"], "metrics": []},
                )
                entry["metrics"].append(candidate["metrics"])
        summaries = []
        for entry in aggregates.values():
            summaries.append({
                "parameters": entry["parameters"],
                "mean_balanced_accuracy": float(np.mean([
                    item["balanced_accuracy"] for item in entry["metrics"]
                ])),
                "mean_f1": float(np.mean([item["f1"] for item in entry["metrics"]])),
                "mean_false_positive_rate": float(np.mean([
                    item["false_positive_rate"] for item in entry["metrics"]
                ])),
            })
        summaries.sort(
            key=lambda item: (
                item["mean_balanced_accuracy"],
                item["mean_f1"],
                -item["mean_false_positive_rate"],
            ),
            reverse=True,
        )
        global_parameters[detector] = {
            "selected": summaries[0],
            "candidates": summaries,
        }

    # Temporal hold-out preview: each final model is fitted only on the first
    # 30% (normal baseline) of its held-out run, and evaluated on later points.
    held_out_preview: dict[str, object] = {}
    for service, fault, _validation_rep, test_rep in PAIRS:
        test_case = f"re1tt_{service}_{fault}_{test_rep}"
        test_train, test_matrix, test_labels = matrix(frames[test_case], features)
        preview = {}
        for detector in ("isolation-forest", "local-outlier-factor"):
            # Context-specific settings come from the separate validation
            # repetition for this service/fault pair, never from the test run.
            parameters = parameter_selection[f"{service}_{fault}"][detector]["selected"]["parameters"]
            pipeline = if_pipeline(parameters) if detector == "isolation-forest" else lof_pipeline(parameters)
            pipeline.fit(test_train)
            predictions = (pipeline.predict(test_matrix) == -1).astype(np.int8)
            preview[detector] = measures(test_labels, predictions)
        held_out_preview[f"{service}_{fault}"] = preview

    write_json(evidence / "source_hashes.json", hashes)
    write_json(evidence / "parameter_selection.json", parameter_selection)
    write_json(evidence / "global_parameters.json", global_parameters)
    write_json(evidence / "held_out_offline_preview.json", held_out_preview)
    write_json(
        evidence / "protocol.json",
        {
            "protocol_version": "cloudsentinel-rcaeval-calibrated-pairs/v2",
            "selection_fixed_before_deployed_run": True,
            "feature_profile": "rcaeval-re1-tt-shared-100/v1",
            "pairs": [
                {
                    "service": service,
                    "fault": fault,
                    "validation_repetition": validation_rep,
                    "held_out_test_repetition": test_rep,
                }
                for service, fault, validation_rep, test_rep in PAIRS
            ],
            "cgnn_parameters": {
                "epochs": 20,
                "lookback": 100,
                "bs": 64,
                "init_lr": 0.001,
                "val_split": 0.15,
                "shuffle_dataset": True,
                "dropout": 0.4,
                "use_cuda": True,
                "feature_importance": False,
                "log_tensorboard": False,
            },
        },
    )
    print(f"Prepared {len(rows)} cases in {output}")


if __name__ == "__main__":
    main()
