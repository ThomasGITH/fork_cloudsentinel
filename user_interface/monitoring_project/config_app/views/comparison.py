"""Server-rendered anomaly-detection Comparison module."""

from __future__ import annotations

import uuid

from django.contrib import messages
from django.core import signing
from django.http import Http404, HttpRequest, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_http_methods

from ..learning_client import LearningClientError, get_learning_client, safe_learning_message


_SALT = "cloudsentinel.comparison-wizard.v1"


def _safe_error(exc: LearningClientError) -> str:
    return safe_learning_message(
        str(exc), "Comparison service is temporarily unavailable."
    )


def _state(raw: str | None) -> dict:
    if not raw:
        return {"schema_version": 1, "nonce": uuid.uuid4().hex}
    try:
        value = signing.loads(raw, salt=_SALT, max_age=3600)
    except signing.BadSignature as exc:
        raise ValueError("Comparison wizard state expired. Start again.") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Comparison wizard state is invalid. Start again.")
    return value


def _token(value: dict) -> str:
    return signing.dumps(value, salt=_SALT, compress=True)


def _page_links(payload: dict, filters: dict) -> tuple[dict | None, dict | None]:
    page = int(payload.get("page", 1))
    pages = int(payload.get("pages", 0))
    previous = {**filters, "page": page - 1} if page > 1 else None
    following = {**filters, "page": page + 1} if page < pages else None
    return previous, following


@require_GET
def comparison_overview(request: HttpRequest):
    allowed = {"search", "modality", "workload", "status", "page", "page_size"}
    filters = {key: value for key, value in request.GET.items() if key in allowed and value}
    context = {"filters": filters, "error": None, "retryable": False}
    try:
        payload = get_learning_client().list_comparisons(filters)
        previous, following = _page_links(payload, filters)
        context.update(comparisons=payload, previous=previous, next=following)
    except LearningClientError as exc:
        context.update(error=_safe_error(exc), retryable=exc.retryable)
    return render(request, "config_app/comparison/overview.html", context)


@require_http_methods(["GET", "POST"])
def comparison_new(request: HttpRequest):
    client = get_learning_client()
    status = 200
    try:
        state = _state(request.POST.get("wizard_state") if request.method == "POST" else None)
        step = int(request.POST.get("step", "1")) if request.method == "POST" else 1
        error = None
        if request.method == "POST":
            if step == 1:
                selection = request.POST.get("dataset_selection", "")
                try:
                    dataset_id, raw_version = selection.rsplit("|", 1)
                    version = int(raw_version)
                except (ValueError, AttributeError) as exc:
                    raise ValueError("Select a valid evaluation dataset.") from exc
                datasets = client.list_evaluation_datasets().get("items", [])
                selected = next(
                    (item for item in datasets if item.get("dataset_id") == dataset_id and item.get("version") == version),
                    None,
                )
                if selected is None:
                    raise ValueError("Select a valid evaluation dataset.")
                state["dataset"] = selected
                step = 2
            elif step == 2:
                selected_ids = request.POST.getlist("model_ids")
                dataset = state.get("dataset") or {}
                models_payload = client.list_compatible_models(
                    dataset["dataset_id"], dataset["version"], dataset["partition_id"]
                )
                compatible = {
                    item["model_id"]: item
                    for item in models_payload.get("items", [])
                    if item.get("compatible")
                }
                if len(selected_ids) < 2 or len(set(selected_ids)) != len(selected_ids):
                    raise ValueError("Select at least two unique compatible Saved Models.")
                if any(model_id not in compatible for model_id in selected_ids):
                    raise ValueError("One or more selected Saved Models are incompatible.")
                state["model_ids"] = selected_ids
                state["models"] = [compatible[model_id] for model_id in selected_ids]
                step = 3
            elif step == 3:
                if state.get("submitted"):
                    raise ValueError("This comparison has already been submitted.")
                name = request.POST.get("name", "").strip()
                if not name or len(name) > 200:
                    raise ValueError("Comparison name must contain 1 to 200 characters.")
                payload = {
                    "name": name,
                    "evaluation_dataset": {
                        key: state["dataset"][key]
                        for key in ("dataset_id", "version", "partition_id")
                    },
                    "model_ids": state["model_ids"],
                    "client_request_id": state["nonce"],
                }
                result = client.create_comparison(payload)
                return redirect("comparison_detail", comparison_id=result["comparison_id"])
            else:
                raise ValueError("Unknown comparison wizard step.")
        context = {"step": step, "state": state, "state_token": _token(state), "error": error}
        if step == 1:
            context["datasets"] = client.list_evaluation_datasets().get("items", [])
        elif step == 2:
            dataset = state["dataset"]
            context["models_payload"] = client.list_compatible_models(
                dataset["dataset_id"], dataset["version"], dataset["partition_id"]
            )
        return render(request, "config_app/comparison/new.html", context, status=status)
    except (ValueError, KeyError, TypeError, LearningClientError) as exc:
        status = 400 if isinstance(exc, (ValueError, KeyError, TypeError)) else 502
        try:
            state_value = state
        except UnboundLocalError:
            state_value = {"schema_version": 1, "nonce": uuid.uuid4().hex}
        context = {
            "step": min(max(locals().get("step", 1), 1), 3),
            "state": state_value,
            "state_token": _token(state_value),
            "error": str(exc),
        }
        if context["step"] == 1:
            try:
                context["datasets"] = client.list_evaluation_datasets().get("items", [])
            except LearningClientError:
                context["datasets"] = []
        elif context["step"] == 2 and state_value.get("dataset"):
            try:
                dataset = state_value["dataset"]
                context["models_payload"] = client.list_compatible_models(
                    dataset["dataset_id"], dataset["version"], dataset["partition_id"]
                )
            except LearningClientError:
                context["models_payload"] = {"items": []}
        return render(request, "config_app/comparison/new.html", context, status=status)


@require_GET
def comparison_detail(request: HttpRequest, comparison_id: str):
    tab = request.GET.get("tab", "overview")
    if tab not in {"overview", "timeline", "models", "robustness"}:
        tab = "overview"
    try:
        comparison = get_learning_client().get_comparison(comparison_id)
        comparison["is_active"] = comparison.get("status") in {"queued", "running"}
        comparison["has_timeline"] = any(
            bool(result.get("timeline")) for result in comparison.get("results", [])
        )
        results_by_model = {
            result.get("model_id"): result for result in comparison.get("results", [])
        }
        for model in comparison.get("selected_models", []):
            model["evaluation_result"] = results_by_model.get(model.get("model_id"), {})
        return render(
            request,
            "config_app/comparison/detail.html",
            {"comparison": comparison, "active_tab": tab},
        )
    except LearningClientError as exc:
        if exc.status_code == 404:
            raise Http404("Comparison not found") from exc
        return render(
            request,
            "config_app/comparison/detail.html",
            {"error": _safe_error(exc), "retryable": exc.retryable},
            status=502,
        )


@require_GET
def comparison_status(request: HttpRequest, comparison_id: str):
    try:
        payload = get_learning_client().get_comparison_status(comparison_id)
        return JsonResponse(payload)
    except LearningClientError as exc:
        return JsonResponse(
            {"error": _safe_error(exc), "retryable": exc.retryable},
            status=exc.status_code if exc.status_code in {400, 404, 409} else 502,
        )
