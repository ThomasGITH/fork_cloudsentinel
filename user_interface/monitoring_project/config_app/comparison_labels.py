"""Readable presentation labels for immutable Saved Model identities."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _detector_label(value: Any) -> str:
    words = [word for word in str(value or "").replace("_", "-").split("-") if word]
    if not words:
        return "Saved Model"
    return " ".join(word.upper() if len(word) <= 4 else word.capitalize() for word in words)


def _created_label(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if "T" not in value:
        return parsed.strftime("%d %b %Y")
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime("%d %b %Y, %H:%M UTC")


def readable_model_name(model: dict[str, Any]) -> str:
    """Prefer an explicit name and otherwise derive a stable readable label."""
    model_id = model.get("model_id")
    explicit = model.get("display_name")
    if (
        isinstance(explicit, str)
        and explicit.strip()
        and explicit.strip() != model_id
    ):
        return explicit.strip()

    detector = _detector_label(model.get("detector_id"))
    version = model.get("detector_version")
    base = f"{detector} v{version}" if version else detector
    created = _created_label(model.get("model_created_at") or model.get("created_at"))
    return f"{base} · {created}" if created else f"{base} Saved Model"


def humanize_model_items(items: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """Mutate safe UI projections and return old/new names keyed by model ID."""
    names: dict[str, tuple[str, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = item.get("model_id")
        old = item.get("display_name") or model_id
        new = readable_model_name(item)
        item["display_name"] = new
        if isinstance(model_id, str):
            names[model_id] = (str(old or model_id), new)
    return names


def humanize_comparison(payload: dict[str, Any]) -> None:
    selected = payload.get("selected_models")
    selected = selected if isinstance(selected, list) else []
    names = humanize_model_items(selected)

    results = payload.get("results")
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            pair = names.get(result.get("model_id"))
            if pair:
                result["display_name"] = pair[1]
            else:
                result["display_name"] = readable_model_name(result)

    winner = payload.get("f1_winner")
    if isinstance(winner, str):
        for old, new in names.values():
            if winner == old:
                payload["f1_winner"] = new
                break

    summary = payload.get("deterministic_summary")
    if isinstance(summary, str):
        for old, new in names.values():
            prefix = f"{old} has the highest F1-score."
            if summary == prefix:
                payload["deterministic_summary"] = f"{new} has the highest F1-score."
                break


def humanize_robustness(payload: dict[str, Any]) -> None:
    models = payload.get("models")
    models = models if isinstance(models, list) else []
    names = humanize_model_items(models)

    warnings = payload.get("coverage_warnings")
    if isinstance(warnings, list):
        readable_coverage = "; ".join(
            (
                f"{model.get('display_name')} has "
                f"{model.get('labelled_context_count', 0)} labelled "
                f"{'context' if model.get('labelled_context_count') == 1 else 'contexts'}"
            )
            for model in models
        )
        payload["coverage_warnings"] = [
            (
                f"Comparison coverage differs: {readable_coverage}."
                if isinstance(warning, str)
                and warning.startswith("Comparison coverage differs:")
                and readable_coverage
                else warning
            )
            for warning in warnings
        ]
    for model in models:
        contexts = model.get("contexts") if isinstance(model, dict) else None
        if not isinstance(contexts, list):
            continue
        for row in contexts:
            if isinstance(row, dict) and row.get("model_id") in names:
                row["display_name"] = names[row["model_id"]][1]
    shared = payload.get("shared_context_matrix")
    if isinstance(shared, list):
        for context in shared:
            rows = context.get("models") if isinstance(context, dict) else None
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict) and row.get("model_id") in names:
                    row["display_name"] = names[row["model_id"]][1]
