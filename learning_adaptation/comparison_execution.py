"""Detector-agnostic execution and metric calculation for AD comparisons."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
import shutil
from typing import Any

try:
    from learning_adaptation.comparison_evaluation import (
        ComparisonEvaluationError,
        EvaluationBundleClient,
        GenericEvaluationClient,
    )
    from learning_adaptation.comparison_storage import ComparisonStore
    from learning_adaptation.training_lifecycle import safe_failure_summary
except ModuleNotFoundError:
    from comparison_evaluation import (
        ComparisonEvaluationError,
        EvaluationBundleClient,
        GenericEvaluationClient,
    )
    from comparison_storage import ComparisonStore
    from training_lifecycle import safe_failure_summary


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def labelled_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    if len(labels) != len(predictions):
        raise ComparisonEvaluationError("prediction count does not match ground truth")
    tp = sum(1 for actual, predicted in zip(labels, predictions) if actual == predicted == 1)
    tn = sum(1 for actual, predicted in zip(labels, predictions) if actual == predicted == 0)
    fp = sum(1 for actual, predicted in zip(labels, predictions) if actual == 0 and predicted == 1)
    fn = sum(1 for actual, predicted in zip(labels, predictions) if actual == 1 and predicted == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def detection_timing(
    predictions: list[int],
    timestamps: list[str] | None,
    incident: dict[str, Any] | None,
) -> dict[str, Any]:
    if not timestamps or len(timestamps) != len(predictions):
        return {
            "first_detection_timestamp": None,
            "lead_time_seconds": None,
            "lead_time_minutes": None,
            "lead_time_status": "unavailable",
        }
    first = next((timestamps[index] for index, value in enumerate(predictions) if value == 1), None)
    result = {
        "first_detection_timestamp": first,
        "lead_time_seconds": None,
        "lead_time_minutes": None,
        "lead_time_status": "unavailable",
    }
    if first is None:
        result["lead_time_status"] = "no_detection"
        return result
    if not isinstance(incident, dict) or not incident.get("start_time"):
        return result
    try:
        seconds = (_parse_timestamp(incident["start_time"]) - _parse_timestamp(first)).total_seconds()
    except (TypeError, ValueError):
        return result
    if seconds >= 0:
        result.update(
            lead_time_seconds=seconds,
            lead_time_minutes=seconds / 60.0,
            lead_time_status="before_or_at_incident_start",
        )
    else:
        result["lead_time_status"] = "after_incident_start"
    return result


def _bounded_timeline(
    predictions: list[int], scores: list[float], timestamps: list[str] | None, maximum: int
) -> list[dict[str, Any]]:
    if not timestamps or len(timestamps) != len(predictions):
        return []
    step = max(1, math.ceil(len(predictions) / maximum))
    indices = list(range(0, len(predictions), step))[:maximum]
    if predictions and indices and indices[-1] != len(predictions) - 1 and len(indices) < maximum:
        indices.append(len(predictions) - 1)
    return [
        {
            "timestamp": timestamps[index],
            "prediction": predictions[index],
            "score": scores[index],
        }
        for index in indices
    ]


def execute_comparison(
    comparison_id: str,
    *,
    storage_root: str,
    catalogue_url: str,
    generic_runtime_url: str,
    maximum_bundle_bytes: int = 250 * 1024 * 1024,
    maximum_timeline_points: int = 1000,
    bundle_client: EvaluationBundleClient | None = None,
    runtime_client: GenericEvaluationClient | None = None,
) -> dict[str, Any]:
    store = ComparisonStore(storage_root)
    now = utc_now()
    store.update(
        comparison_id,
        lambda record: record.update(status="running", started_at=now, updated_at=now),
    )
    record = store.read(comparison_id)
    bundle = None
    try:
        bundle = (bundle_client or EvaluationBundleClient(
            catalogue_url, maximum_bytes=maximum_bundle_bytes
        )).download(record["evaluation_dataset"])
        manifest = bundle["manifest"]
        if manifest["feature_order_sha256"] != record["feature_identity"]["sha256"]:
            raise ComparisonEvaluationError("evaluation feature identity changed after submission")
        runtime = runtime_client or GenericEvaluationClient(generic_runtime_url)
        results: list[dict[str, Any]] = []
        for result_index, selected in enumerate(record["selected_models"]):
            started = utc_now()
            base = {
                "model_id": selected["model_id"],
                "detector_id": selected["detector_id"],
                "detector_version": selected["detector_version"],
                "display_name": selected["display_name"],
                "artifact_identity": selected["artifact_identity"],
                "status": "failed",
                "started_at": started,
                "completed_at": None,
                "safe_error_summary": None,
            }
            def mark_model_running(stored: dict[str, Any]) -> None:
                current = stored.setdefault("results", [])
                while len(current) <= result_index:
                    current.append({})
                current[result_index] = deepcopy(base)
                current[result_index]["status"] = "running"
                stored["updated_at"] = started

            store.update(comparison_id, mark_model_running)
            try:
                payload = runtime.evaluate(
                    selected["model_id"],
                    bundle["matrix_path"],
                    manifest["feature_order"],
                    manifest["feature_order_sha256"],
                    bundle["timestamps"],
                )
                if payload.get("artifact_manifest_sha256") != selected[
                    "artifact_identity"
                ]["artifact_manifest_sha256"]:
                    raise ComparisonEvaluationError(
                        "Saved Model artifact identity changed after submission"
                    )
                predictions = payload.get("binary_predictions")
                scores = payload.get("anomaly_scores")
                warmup = payload.get("warmup_observations", 0)
                input_count = payload.get("input_observation_count")
                expected_count = manifest.get("counts", {}).get("observations")
                runtime_ms = payload.get("runtime_ms")
                if (
                    not isinstance(predictions, list)
                    or not isinstance(scores, list)
                    or len(predictions) != len(scores)
                    or any(type(value) is not int or value not in {0, 1} for value in predictions)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        for value in scores
                    )
                    or not isinstance(warmup, int)
                    or isinstance(warmup, bool)
                    or warmup < 0
                    or input_count != expected_count
                    or warmup + len(predictions) != expected_count
                    or isinstance(runtime_ms, bool)
                    or not isinstance(runtime_ms, (int, float))
                    or not math.isfinite(runtime_ms)
                    or runtime_ms < 0
                ):
                    raise ComparisonEvaluationError("generic runtime returned invalid evaluation output")
                labels = bundle["labels"][warmup:] if bundle["labels"] is not None else None
                timestamps = bundle["timestamps"][warmup:] if bundle["timestamps"] is not None else None
                metrics = labelled_metrics(labels, predictions) if labels is not None else {}
                incident = (manifest.get("known_incident_windows") or [None])[0]
                timing = detection_timing(predictions, timestamps, incident)
                if labels is None:
                    timing.update(
                        lead_time_seconds=None,
                        lead_time_minutes=None,
                        lead_time_status="ground_truth_unavailable",
                    )
                base.update(
                    status="completed",
                    runtime_ms=float(runtime_ms),
                    observation_count=int(input_count),
                    prediction_count=len(predictions),
                    anomaly_count=sum(predictions),
                    ground_truth_available=labels is not None,
                    metrics=metrics,
                    **timing,
                    timeline=_bounded_timeline(
                        predictions, [float(value) for value in scores], timestamps, maximum_timeline_points
                    ),
                )
            except Exception as exc:
                base["safe_error_summary"] = safe_failure_summary(exc) or "model evaluation failed"
            base["completed_at"] = utc_now()
            results.append(base)

            def persist_model(stored: dict[str, Any]) -> None:
                stored["results"][result_index] = deepcopy(base)
                stored["updated_at"] = base["completed_at"]

            store.update(comparison_id, persist_model)

        succeeded = sum(item["status"] == "completed" for item in results)
        terminal_status = "completed" if succeeded == len(results) else ("partial" if succeeded else "failed")
        labelled = [
            item for item in results
            if item["status"] == "completed" and item.get("ground_truth_available") and item.get("metrics")
        ]
        summary = None
        if labelled:
            highest = max(item["metrics"]["f1_score"] for item in labelled)
            winners = [item for item in labelled if item["metrics"]["f1_score"] == highest]
            if len(winners) == 1:
                summary = f"{winners[0]['display_name']} has the highest F1-score."
        completed_at = utc_now()

        def finish(stored: dict[str, Any]) -> None:
            stored.update(
                status=terminal_status,
                updated_at=completed_at,
                completed_at=completed_at,
                results=results,
                result_reference=f"comparison-result-{comparison_id}",
                deterministic_summary=summary,
                ground_truth_available=manifest["ground_truth_available"],
                known_incident_window=(manifest.get("known_incident_windows") or [None])[0],
                known_incident_windows=deepcopy(manifest.get("known_incident_windows") or []),
            )

        return store.update(comparison_id, finish)
    except Exception as exc:
        failed_at = utc_now()

        def fail(stored: dict[str, Any]) -> None:
            stored.update(
                status="failed",
                updated_at=failed_at,
                completed_at=failed_at,
                safe_failure_summary=safe_failure_summary(exc) or "comparison execution failed",
                results=[],
            )

        return store.update(comparison_id, fail)
    finally:
        if bundle is not None:
            shutil.rmtree(bundle["temporary_root"], ignore_errors=True)
