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
        # One model is enough to enter the wizard; a second compatible saved
        # model can be selected there before the comparison is submitted.
        "can_compare": len(models) >= 1,
    }


def comparison_source_context(comparison: Any) -> dict[str, Any] | None:
    """Project a stored comparison into safe, server-resolved wizard seed state."""
    if not isinstance(comparison, dict):
        return None
    models = []
    seen = set()
    for model in comparison.get("selected_models") or []:
        if not isinstance(model, dict):
            continue
        model_id = model.get("model_id")
        artifact = model.get("artifact_identity") or {}
        if (
            not isinstance(model_id, str)
            or not model_id.strip()
            or model_id in seen
            or not isinstance(artifact, dict)
            or not isinstance(artifact.get("artifact_id"), str)
            or not artifact.get("artifact_id")
            or not isinstance(artifact.get("artifact_manifest_sha256"), str)
            or not artifact.get("artifact_manifest_sha256")
        ):
            continue
        seen.add(model_id)
        models.append(
            {
                "model_id": model_id,
                "display_name": str(model.get("display_name") or model_id),
                "detector_id": str(model.get("detector_id") or ""),
                "artifact_id": artifact["artifact_id"],
                "artifact_manifest_sha256": artifact["artifact_manifest_sha256"],
            }
        )
    dataset = comparison.get("evaluation_dataset") or {}
    return {
        "source_comparison_id": comparison.get("comparison_id"),
        "source_comparison_name": comparison.get("name")
        or comparison.get("comparison_id")
        or "Comparison",
        "successful_models": models,
        "successful_count": len(models),
        "can_compare": len(models) >= 2,
        "evaluation_context": {
            key: dataset.get(key)
            for key in ("dataset_id", "version", "partition_id", "partition_checksum")
        },
    }
