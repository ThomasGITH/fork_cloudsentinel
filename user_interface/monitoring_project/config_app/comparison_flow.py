"""Small shared projections for moving from training to Comparison."""

from __future__ import annotations

from typing import Any


TERMINAL_TRAINING_STATUSES = {
    "completed",
    "partial_success",
    "failed",
    "dispatch_failed",
    "validation_failed",
}


def successful_training_models(run: Any) -> list[dict[str, str]]:
    """Return only durable, available models produced by a finished child run."""
    if not isinstance(run, dict):
        return []
    models = []
    seen = set()
    children = run.get("children")
    if not isinstance(children, list):
        return []
    for child in children:
        if not isinstance(child, dict):
            continue
        model_id = child.get("model_id")
        if (
            child.get("status") != "completed"
            or child.get("model_status") != "available"
            or not isinstance(model_id, str)
            or not model_id.strip()
            or model_id in seen
        ):
            continue
        seen.add(model_id)
        display_name = child.get("model_display_name")
        if not isinstance(display_name, str) or not display_name.strip():
            display_name = str(child.get("detector_id") or model_id)
        models.append(
            {
                "model_id": model_id,
                "display_name": display_name.strip(),
                "detector_id": str(child.get("detector_id") or ""),
            }
        )
    return models


def training_comparison_context(run: Any) -> dict[str, Any] | None:
    if not isinstance(run, dict) or run.get("status") not in TERMINAL_TRAINING_STATUSES:
        return None
    models = successful_training_models(run)
    return {
        "source_run_id": run.get("run_id"),
        "source_run_name": run.get("run_name") or run.get("run_id") or "Training run",
        "successful_models": models,
        "successful_count": len(models),
        "can_compare": len(models) >= 2,
    }
