"""Server-rendered anomaly-detection Comparison module."""

from __future__ import annotations

import uuid
from urllib.parse import urlencode

from django.contrib import messages
from django.core import signing
from django.http import Http404, HttpRequest, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_http_methods

from ..learning_client import LearningClientError, get_learning_client, safe_learning_message
from ..comparison_labels import (
    humanize_comparison,
    humanize_model_items,
    humanize_robustness,
)
from ..comparison_decision import build_decision_support
from ..comparison_flow import comparison_source_context, training_comparison_context
from ..workload_context import ANOMALY_SCENARIOS, WORKLOAD_INTENSITY_DEFINITIONS
from ..workflow_context import (
    context_from_request,
    encode_context,
    new_context,
    progress_context,
    update_context,
)


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


def _source_run_state(client, raw_run_id: str | None) -> tuple[dict | None, str | None]:
    """Resolve URL input through the Learning API and retain only safe model identities."""
    if not raw_run_id:
        return None, None
    if not isinstance(raw_run_id, str) or not raw_run_id.strip() or len(raw_run_id) > 128:
        return None, "The requested source training run is invalid. You can continue without preselection."
    try:
        run = client.get_training_run(raw_run_id.strip())
    except LearningClientError:
        return None, "The requested source training run is unavailable. You can continue without preselection."
    source = training_comparison_context(run)
    if source is None:
        return None, "The source training run is not finished. You can continue without preselection."
    return source, None


def _source_comparison_state(client, raw_id: str | None) -> tuple[dict | None, str | None]:
    if not raw_id:
        return None, None
    if not isinstance(raw_id, str) or not raw_id.strip() or len(raw_id) > 128:
        return None, "The requested source comparison is invalid. You can continue without preselection."
    try:
        comparison = client.get_comparison(raw_id.strip())
    except LearningClientError:
        return None, "The requested source comparison is unavailable. You can continue without preselection."
    source = comparison_source_context(comparison)
    if source is None or not source.get("successful_models"):
        return None, "The source comparison has no usable immutable model artefacts. You can continue without preselection."
    return source, None


def _same_evaluation_context(source: dict, selected: dict) -> bool:
    original = source.get("evaluation_context") or {}
    required = ("dataset_id", "version", "partition_id", "partition_checksum")
    return all(original.get(key) is not None and original.get(key) == selected.get(key) for key in required)


def _prepare_model_selection(state: dict, payload: dict, checked_ids=None) -> dict:
    items = payload.get("items") if isinstance(payload, dict) else None
    items = items if isinstance(items, list) else []
    humanize_model_items(items)
    source_kind = "training_run" if isinstance(state.get("source_run"), dict) else "comparison"
    source = state.get("source_run") or state.get("source_comparison") or {}
    source_models = source.get("successful_models")
    source_models = source_models if isinstance(source_models, list) else []
    source_by_id = {
        item.get("model_id"): item
        for item in source_models
        if isinstance(item, dict) and isinstance(item.get("model_id"), str)
    }
    available_ids = {
        item.get("model_id") for item in items if isinstance(item, dict)
    }
    requested = set(checked_ids) if checked_ids is not None else set(source_by_id)
    incompatible = []
    for item in items:
        model_id = item.get("model_id")
        from_source = model_id in source_by_id
        source_identity_matches = True
        if from_source and source_kind == "comparison":
            expected = source_by_id[model_id]
            actual = item.get("artifact_identity") or {}
            source_identity_matches = (
                actual.get("artifact_id") == expected.get("artifact_id")
                and actual.get("artifact_manifest_sha256")
                == expected.get("artifact_manifest_sha256")
            )
            if not source_identity_matches:
                item["source_preselection_reasons"] = [
                    "Saved Model artifact identity no longer matches the source comparison"
                ]
        item["from_source_run"] = from_source and source_kind == "training_run"
        item["from_source_comparison"] = from_source and source_kind == "comparison"
        item["preselected"] = bool(
            model_id in requested
            and item.get("compatible") is True
            and source_identity_matches
        )
        if from_source and (
            item.get("compatible") is not True or not source_identity_matches
        ):
            incompatible.append(item)
    missing = [source_by_id[model_id] for model_id in source_by_id if model_id not in available_ids]
    payload["items"] = items
    return {
        "models_payload": payload,
        "source_incompatible_models": incompatible,
        "source_missing_models": missing,
    }


def _page_links(payload: dict, filters: dict) -> tuple[dict | None, dict | None]:
    page = int(payload.get("page", 1))
    pages = int(payload.get("pages", 0))
    previous = {**filters, "page": page - 1} if page > 1 else None
    following = {**filters, "page": page + 1} if page < pages else None
    return previous, following


@require_GET
def comparison_overview(request: HttpRequest):
    allowed = {
        "search", "modality", "workload", "status", "page", "page_size",
        "model_id", "artifact_id", "artifact_manifest_sha256",
    }
    filters = {key: value for key, value in request.GET.items() if key in allowed and value}
    context = {"filters": filters, "error": None, "retryable": False}
    try:
        payload = get_learning_client().list_comparisons(filters)
        for item in payload.get("items", []):
            if isinstance(item, dict):
                humanize_comparison(item)
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
        source_notice = None
        if request.method == "GET":
            state["workflow"] = context_from_request(request)
            raw_run = request.GET.get("source_run")
            raw_comparison = request.GET.get("source_comparison")
            if raw_run and raw_comparison:
                source_notice = "Choose either a source training run or a source comparison, not both. Continue without preselection."
            elif raw_run:
                source, source_notice = _source_run_state(client, raw_run)
                if source is not None:
                    state["source_run"] = source
            elif raw_comparison:
                source, source_notice = _source_comparison_state(client, raw_comparison)
                if source is not None:
                    state["source_comparison"] = source
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
                state["workflow"] = update_context(
                    state.get("workflow") or new_context(),
                    dataset_id=selected.get("dataset_id"),
                    dataset_version=selected.get("version"),
                    dataset_name=selected.get("display_name"),
                    partition_id=selected.get("partition_id"),
                )
                source_comparison = state.get("source_comparison")
                state["repeated_source_context"] = bool(
                    isinstance(source_comparison, dict)
                    and _same_evaluation_context(source_comparison, selected)
                )
                step = 2
            elif step == 2:
                selected_ids = request.POST.getlist("model_ids")
                dataset = state.get("dataset") or {}
                models_payload = client.list_compatible_models(
                    dataset["dataset_id"], dataset["version"], dataset["partition_id"]
                )
                humanize_model_items(models_payload.get("items", []))
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
                comparison_id = result["comparison_id"]
                workflow = update_context(
                    state.get("workflow") or new_context(), comparison_id=comparison_id
                )
                target = reverse("comparison_detail", args=[comparison_id])
                return redirect(f"{target}?{urlencode({'flow': encode_context(workflow)})}")
            else:
                raise ValueError("Unknown comparison wizard step.")
        context = {
            "step": step,
            "state": state,
            "state_token": _token(state),
            "error": error,
            "source_notice": source_notice,
            "workflow_progress": progress_context(state.get("workflow"), "comparison"),
        }
        if step == 1:
            datasets = client.list_evaluation_datasets().get("items", [])
            workflow = state.get("workflow") or {}
            for item in datasets:
                item["workflow_preselected"] = (
                    item.get("dataset_id") == workflow.get("dataset_id")
                    and item.get("version") == workflow.get("dataset_version")
                )
            context["datasets"] = datasets
        elif step == 2:
            dataset = state["dataset"]
            models_payload = client.list_compatible_models(
                dataset["dataset_id"], dataset["version"], dataset["partition_id"]
            )
            context.update(_prepare_model_selection(state, models_payload))
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
                models_payload = client.list_compatible_models(
                    dataset["dataset_id"], dataset["version"], dataset["partition_id"]
                )
                context.update(
                    _prepare_model_selection(
                        state_value,
                        models_payload,
                        locals().get("selected_ids"),
                    )
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
        client = get_learning_client()
        comparison = client.get_comparison(comparison_id)
        humanize_comparison(comparison)
        comparison["is_active"] = comparison.get("status") in {"queued", "running"}
        comparison["has_timeline"] = any(
            bool(result.get("timeline")) for result in comparison.get("results", [])
        )
        incident_windows = comparison.get("known_incident_windows")
        if not isinstance(incident_windows, list):
            incident = comparison.get("known_incident_window")
            incident_windows = [incident] if isinstance(incident, dict) else []
        comparison["timeline_incident_windows"] = incident_windows
        results_by_model = {
            result.get("model_id"): result for result in comparison.get("results", [])
        }
        for model in comparison.get("selected_models", []):
            model["evaluation_result"] = results_by_model.get(model.get("model_id"), {})
        comparison["decision_support"] = build_decision_support(
            comparison, request.GET.get("goal")
        )
        evaluation = comparison.get("evaluation_dataset") or {}
        workflow = update_context(
            context_from_request(request) or new_context(),
            dataset_id=evaluation.get("dataset_id"),
            dataset_version=evaluation.get("version"),
            dataset_name=evaluation.get("display_name"),
            partition_id=evaluation.get("partition_id"),
            comparison_id=comparison_id,
        )
        workflow_token = encode_context(workflow)
        robustness = None
        robustness_error = None
        robustness_filters = {}
        if tab == "robustness":
            allowed = {"workload", "scenario", "labelled_only", "shared_only"}
            robustness_filters = {
                key: value
                for key, value in request.GET.items()
                if key in allowed and value
            }
            try:
                robustness = client.get_comparison_robustness(
                    comparison_id, robustness_filters
                )
                humanize_robustness(robustness)
            except LearningClientError as exc:
                robustness_error = _safe_error(exc)
        return render(
            request,
            "config_app/comparison/detail.html",
            {
                "comparison": comparison,
                "active_tab": tab,
                "robustness": robustness,
                "robustness_error": robustness_error,
                "robustness_filters": robustness_filters,
                "robustness_scenarios": ANOMALY_SCENARIOS,
                "workload_intensity_definitions": WORKLOAD_INTENSITY_DEFINITIONS,
                "workflow_progress": progress_context(workflow, "comparison"),
                "workflow_token": workflow_token,
            },
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
