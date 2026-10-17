"""Deterministic read-only decision support over one ComparisonRun."""

from __future__ import annotations

import math
from typing import Any


GOALS = (
    {
        "id": "balanced_detection",
        "label": "Balanced detection",
        "description": "Balance missed incidents and false alarms using F1-score.",
        "metric_key": "f1_score",
        "metric_label": "F1-score",
        "direction": "higher",
        "unit": "ratio",
    },
    {
        "id": "catch_incidents",
        "label": "Catch as many incidents as possible",
        "description": "Prioritize finding labelled incidents using recall.",
        "metric_key": "recall",
        "metric_label": "Recall",
        "direction": "higher",
        "unit": "ratio",
    },
    {
        "id": "minimize_false_alarms",
        "label": "Minimize false alarms",
        "description": "Prioritize precise alerts using precision.",
        "metric_key": "precision",
        "metric_label": "Precision",
        "direction": "higher",
        "unit": "ratio",
    },
    {
        "id": "earliest_detection",
        "label": "Detect incidents earliest",
        "description": "Prefer the longest valid lead time before the known incident start.",
        "metric_key": "lead_time_seconds",
        "metric_label": "Detection lead time",
        "direction": "higher",
        "unit": "seconds",
    },
    {
        "id": "minimize_compute_cost",
        "label": "Minimize compute cost",
        "description": "Use measured inference runtime as the available compute-cost proxy.",
        "metric_key": "runtime_ms",
        "metric_label": "Inference runtime",
        "direction": "lower",
        "unit": "milliseconds",
    },
)


def _finite(value: Any, *, minimum: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        return None
    return number


def _metric(result: dict[str, Any], key: str) -> float | None:
    if key in {"precision", "recall", "f1_score"}:
        if result.get("ground_truth_available") is not True:
            return None
        metrics = result.get("metrics")
        return _finite(metrics.get(key), minimum=0.0) if isinstance(metrics, dict) else None
    if key == "runtime_ms":
        return _finite(result.get("runtime_ms"), minimum=0.0)
    if key == "lead_time_seconds":
        if result.get("lead_time_status") != "before_or_at_incident_start":
            return None
        seconds = _finite(result.get("lead_time_seconds"), minimum=0.0)
        if seconds is not None:
            return seconds
        minutes = _finite(result.get("lead_time_minutes"), minimum=0.0)
        return minutes * 60.0 if minutes is not None else None
    return None


def _display(value: float, unit: str) -> str:
    if unit == "ratio":
        return f"{value:.3f}"
    if unit == "milliseconds":
        return f"{value:.1f} ms"
    if unit == "seconds":
        return f"{value / 60.0:.1f} min"
    return str(value)


def _model_projection(result: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    live = selected.get("live_monitoring")
    live = live if isinstance(live, dict) else {"status": "not_ready"}
    return {
        "model_id": result.get("model_id"),
        "display_name": result.get("display_name") or selected.get("display_name") or result.get("model_id"),
        "detector_id": result.get("detector_id") or selected.get("detector_id"),
        "detector_version": result.get("detector_version") or selected.get("detector_version"),
        "live_monitoring": {
            "status": live.get("status") or "not_ready",
            "reason": live.get("reason"),
        },
    }


def build_decision_support(
    comparison: Any, selected_goal_id: str | None = None
) -> dict[str, Any]:
    """Project goal-specific recommendations without mutating stored results."""
    payload = comparison if isinstance(comparison, dict) else {}
    selected_models = {
        item.get("model_id"): item
        for item in payload.get("selected_models", [])
        if isinstance(item, dict) and isinstance(item.get("model_id"), str)
    }
    results = [
        item
        for item in payload.get("results", [])
        if isinstance(item, dict) and item.get("status") == "completed"
    ]
    supported_ids = {goal["id"] for goal in GOALS}
    selected_id = selected_goal_id if selected_goal_id in supported_ids else GOALS[0]["id"]
    projected_goals = []
    for definition in GOALS:
        participants = []
        for result in results:
            value = _metric(result, definition["metric_key"])
            if value is None:
                continue
            model = _model_projection(
                result, selected_models.get(result.get("model_id"), {})
            )
            participants.append({"value": value, "model": model})
        reverse = definition["direction"] == "higher"
        participants.sort(key=lambda item: item["value"], reverse=reverse)
        goal = {**definition, "status": "not_enough_data", "best_models": []}
        if participants:
            best_value = participants[0]["value"]
            winners = [item for item in participants if item["value"] == best_value]
            runner_up = next(
                (item for item in participants if item["value"] != best_value), None
            )
            difference = None
            if runner_up is not None:
                difference = (
                    best_value - runner_up["value"]
                    if reverse
                    else runner_up["value"] - best_value
                )
            tie = len(winners) > 1
            names = ", ".join(str(item["model"]["display_name"]) for item in winners)
            comparison_word = "highest" if reverse else "lowest"
            rationale = (
                f"{names} share the best measured {definition['metric_label']} in this evaluation."
                if tie
                else f"{names} has the {comparison_word} measured {definition['metric_label']} among successful models with available data."
            )
            goal.update(
                status="available",
                best_models=[item["model"] for item in winners],
                metric_value=best_value,
                metric_value_display=_display(best_value, definition["unit"]),
                difference_to_second=difference,
                difference_to_second_display=(
                    _display(difference, definition["unit"])
                    if difference is not None
                    else None
                ),
                tie=tie,
                rationale=rationale,
                eligible_model_count=len(participants),
            )
        projected_goals.append(goal)
    selected = next(goal for goal in projected_goals if goal["id"] == selected_id)
    return {
        "selected_goal_id": selected_id,
        "selected_goal": selected,
        "goals": projected_goals,
        "evaluation_notice": (
            "This result applies to the selected evaluation dataset and workload context. "
            "Performance may differ in another environment."
        ),
    }
