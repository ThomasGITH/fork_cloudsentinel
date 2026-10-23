"""Signed UI context for carrying one experiment through the workflow screens."""

from __future__ import annotations

import uuid
from typing import Any

from django.core import signing


PARAMETER = "flow"
_SALT = "cloudsentinel.guided-experiment.v1"
_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
_ALLOWED_KEYS = {
    "schema_version",
    "flow_id",
    "dataset_id",
    "dataset_version",
    "dataset_name",
    "partition_id",
    "training_run_id",
    "comparison_id",
    "model_id",
    "monitoring_session_id",
}
_STEPS = (
    ("dataset", "Dataset"),
    ("training", "Training"),
    ("comparison", "Comparison"),
    ("monitoring", "Live monitoring"),
    ("evidence", "Evidence"),
)


def new_context(**values: Any) -> dict[str, Any]:
    context: dict[str, Any] = {
        "schema_version": 1,
        "flow_id": uuid.uuid4().hex,
    }
    return update_context(context, **values)


def update_context(context: dict[str, Any] | None, **values: Any) -> dict[str, Any]:
    result = dict(context) if context else {
        "schema_version": 1,
        "flow_id": uuid.uuid4().hex,
    }
    for key, value in values.items():
        if key not in _ALLOWED_KEYS or value in {None, ""}:
            continue
        if key == "dataset_version":
            try:
                result[key] = int(value)
            except (TypeError, ValueError):
                continue
        else:
            result[key] = str(value)[:200]
    result["schema_version"] = 1
    result.setdefault("flow_id", uuid.uuid4().hex)
    return result


def encode_context(context: dict[str, Any] | None) -> str:
    return signing.dumps(update_context(context), salt=_SALT, compress=True)


def decode_context(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        value = signing.loads(raw, salt=_SALT, max_age=_MAX_AGE_SECONDS)
    except signing.BadSignature:
        return None
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return None
    clean = {key: item for key, item in value.items() if key in _ALLOWED_KEYS}
    if not isinstance(clean.get("flow_id"), str):
        return None
    return update_context(clean)


def context_from_request(request) -> dict[str, Any] | None:
    return decode_context(request.GET.get(PARAMETER))


def progress_context(
    context: dict[str, Any] | None,
    current_step: str,
) -> dict[str, Any] | None:
    if not context or current_step not in {key for key, _ in _STEPS}:
        return None
    current_index = next(index for index, item in enumerate(_STEPS) if item[0] == current_step)
    steps = []
    for index, (key, label) in enumerate(_STEPS):
        state = (
            "complete"
            if index < current_index
            else "current"
            if index == current_index
            else "upcoming"
        )
        steps.append({"key": key, "label": label, "number": index + 1, "state": state})
    return {
        "flow_id": context["flow_id"],
        "token": encode_context(context),
        "steps": steps,
        "current_step": current_step,
        "dataset_name": context.get("dataset_name") or context.get("dataset_id"),
        "dataset_version": context.get("dataset_version"),
        "training_run_id": context.get("training_run_id"),
        "comparison_id": context.get("comparison_id"),
        "model_id": context.get("model_id"),
        "monitoring_session_id": context.get("monitoring_session_id"),
    }
