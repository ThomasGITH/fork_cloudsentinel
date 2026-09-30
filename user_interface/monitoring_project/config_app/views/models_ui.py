"""Server-rendered Models and Training UI."""

from __future__ import annotations

import re
import uuid
from urllib.parse import urlencode

from django.contrib import messages
from django.core import signing
from django.core.cache import cache
from django.http import Http404, HttpRequest, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_http_methods

from config_app.catalogue_client import CatalogueClientError, get_catalogue_client
from config_app.learning_client import (
    LearningClientError,
    get_learning_client,
    safe_learning_message,
)


_WIZARD_SALT = "cloudsentinel.models.training.v1"
_SAFE_KEYS_TO_REMOVE = {
    "entry_point", "task_id", "artifact", "artifacts", "artifact_dir",
    "path", "base_url", "url", "broker", "redis", "traceback",
    "credentials", "token", "secret",
}
_INTERNAL_URL = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_INTERNAL_PATH = re.compile(r"(?<![A-Za-z0-9])/(?:app|tmp|var|home|Users)/[^\s]*")
_ACTIVE_CHILD_STATUSES = {"queued", "running"}


def _safe(value):
    """Remove internal transport/runtime details before rendering or proxying."""
    if isinstance(value, dict):
        return {
            key: _safe(item)
            for key, item in value.items()
            if str(key).lower() not in _SAFE_KEYS_TO_REMOVE
        }
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, str):
        value = _INTERNAL_URL.sub("[internal service]", value)
        return _INTERNAL_PATH.sub("[internal path]", value)
    return value


def _error(exc: Exception) -> str:
    return safe_learning_message(str(exc), "The requested information is unavailable.")


def _filters(request: HttpRequest, allowed: set[str]) -> dict[str, str]:
    return {
        key: request.GET[key].strip()
        for key in allowed
        if request.GET.get(key, "").strip()
    }


def _page_links(payload: dict, filters: dict, *, tab: str) -> tuple[str | None, str | None]:
    try:
        page, pages = int(payload.get("page", 1)), int(payload.get("pages", 0))
    except (TypeError, ValueError):
        return None, None
    base = {**filters, "tab": tab}
    previous = f"?{urlencode({**base, 'page': page - 1})}" if page > 1 else None
    following = f"?{urlencode({**base, 'page': page + 1})}" if page < pages else None
    return previous, following


def _training_capability(detector: dict) -> dict:
    return detector.get("capabilities", {}).get("training", {})


def _detector_ui(detector: dict) -> dict:
    item = _safe(detector)
    training = _training_capability(detector)
    runtime_trainable = (
        training.get("enabled") is True
        and training.get("protocol") == "cloudsentinel.training/v1"
        and training.get("input_profile") == "metrics-partition-v1"
        and "metrics" in detector.get("supported_modalities", [])
    )
    item["training"] = {
        "enabled": training.get("enabled") is True,
        "trainable": runtime_trainable,
        "protocol": training.get("protocol"),
        "input_profile": training.get("input_profile"),
    }
    requirements = detector.get("input_requirements", {})
    training_requirements = requirements.get("training", requirements)
    summary_parts = []
    if isinstance(training_requirements, dict):
        for key in ("format", "shape", "observation", "feature_identity"):
            value = training_requirements.get(key)
            if value:
                summary_parts.append(str(value))
        inputs = training_requirements.get("inputs")
        if isinstance(inputs, list) and inputs:
            summary_parts.append("Inputs: " + ", ".join(str(value) for value in inputs))
    item["input_summary"] = " · ".join(summary_parts) or "See detector details for input requirements."
    parameters = []
    for name, metadata in detector.get("training_parameters", {}).items():
        parameter = {"name": name, **_safe(metadata)}
        parameter["advanced"] = bool(metadata.get("advanced", False))
        parameters.append(parameter)
    item["basic_parameters"] = [value for value in parameters if not value["advanced"]]
    item["advanced_parameters"] = [value for value in parameters if value["advanced"]]
    item["parameter_count"] = len(parameters)
    return item


def _detector_map() -> tuple[list[dict], list[dict]]:
    payload = get_learning_client().list_detectors()
    detectors = [_detector_ui(item) for item in payload.get("detectors", [])]
    errors = []
    for raw in payload.get("discovery_errors", []):
        if isinstance(raw, dict):
            errors.append({
                "plugin": raw.get("plugin") or raw.get("name") or "Unknown plugin",
                "error": safe_learning_message(raw.get("error") or raw.get("message"), "Invalid detector manifest."),
            })
        else:
            errors.append({"plugin": "Unknown plugin", "error": safe_learning_message(raw, "Invalid detector manifest.")})
    return detectors, errors


def _add_dataset_display_names(items: list[dict]) -> None:
    """Add catalogue display names without making Models depend on their availability."""
    catalogue = get_catalogue_client()
    names: dict[str, str] = {}
    for item in items:
        dataset = item.get("dataset")
        if not isinstance(dataset, dict) or dataset.get("source") != "catalogue":
            continue
        dataset_id = dataset.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            continue
        if dataset_id not in names:
            try:
                record = catalogue.get_dataset(dataset_id)
                display_name = record.get("display_name")
                names[dataset_id] = (
                    display_name.strip()
                    if isinstance(display_name, str) and display_name.strip()
                    else dataset_id
                )
            except CatalogueClientError:
                names[dataset_id] = dataset_id
        dataset["display_name"] = names[dataset_id]


@require_GET
def models_overview(request: HttpRequest):
    tab = request.GET.get("tab", "detectors")
    if tab not in {"detectors", "saved", "history"}:
        tab = "detectors"
    context = {"active_tab": tab, "error": None, "retryable": False}
    try:
        if tab == "detectors":
            detectors, discovery_errors = _detector_map()
            context.update(detectors=detectors, discovery_errors=discovery_errors)
        elif tab == "saved":
            allowed = {"page", "page_size", "detector_id", "status", "dataset_id", "sort"}
            filters = _filters(request, allowed)
            payload = _safe(get_learning_client().list_models(filters))
            _add_dataset_display_names(payload.get("items", []))
            previous, following = _page_links(payload, filters, tab=tab)
            context.update(models=payload, filters=filters, previous=previous, next=following)
        else:
            allowed = {"page", "page_size", "status", "detector_id", "dataset_id", "source", "sort"}
            filters = _filters(request, allowed)
            payload = _safe(get_learning_client().list_training_runs(filters))
            _add_dataset_display_names(payload.get("items", []))
            for item in payload.get("items", []):
                statuses = {}
                for child in item.get("children", []):
                    status = child.get("status", "unknown")
                    statuses[status] = statuses.get(status, 0) + 1
                item["child_summary"] = statuses
            previous, following = _page_links(payload, filters, tab=tab)
            context.update(runs=payload, filters=filters, previous=previous, next=following)
    except LearningClientError as exc:
        context.update(error=_error(exc), retryable=exc.retryable)
    return render(request, "config_app/models/overview.html", context)


@require_GET
def detector_detail(request: HttpRequest, detector_id: str):
    try:
        detector = _detector_ui(get_learning_client().get_detector(detector_id))
        return render(request, "config_app/models/detector_detail.html", {"detector": detector})
    except LearningClientError as exc:
        if exc.status_code == 404:
            raise Http404("Detector not found") from exc
        return render(request, "config_app/models/detector_detail.html", {"error": _error(exc), "retryable": exc.retryable}, status=502)


def _count(partition_part: object) -> int | None:
    if not isinstance(partition_part, dict):
        return None
    value = partition_part.get("observation_count", partition_part.get("row_count"))
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _dataset_readiness(version: dict) -> tuple[bool, list[str], dict]:
    reasons = []
    partition = version.get("partition") or {}
    schema = version.get("technical_schema") or {}
    ground_truth = version.get("ground_truth") or {}
    usage = version.get("usages", {}).get("labeled_evaluation", {})
    if version.get("status") != "available":
        reasons.append("Dataset version is not available.")
    if partition.get("mode") in {None, "none"} or not partition.get("partition_id"):
        reasons.append("An active explicit partition is required.")
    train_count, test_count = _count(partition.get("train")), _count(partition.get("test"))
    if train_count is None or train_count < 2:
        reasons.append("The partition needs at least two training observations.")
    if test_count is None or test_count < 1:
        reasons.append("The partition needs at least one test observation.")
    label_source = partition.get("labels_source") or ground_truth.get("active_label_source")
    labels = partition.get("labels") or {}
    labels_available = bool(ground_truth.get("labels_available") or labels.get("artifact") or label_source)
    if not labels_available:
        reasons.append("Row-level or incident-derived labels are required.")
    if usage.get("supported") is not True:
        reasons.extend(str(reason) for reason in usage.get("reasons", []) if reason)
    try:
        if int(schema.get("feature_count", 0)) < 1:
            reasons.append("At least one feature is required.")
    except (TypeError, ValueError):
        reasons.append("Feature metadata is invalid.")
    reasons = list(dict.fromkeys(reasons))
    summary = {
        "partition_id": partition.get("partition_id"),
        "partition_checksum": partition.get("checksum"),
        "train_count": train_count,
        "test_count": test_count,
        "feature_count": schema.get("feature_count"),
        "feature_order": schema.get("feature_order", []),
        "feature_order_sha256": (partition.get("feature_order_reference") or {}).get("sha256"),
        "label_source": label_source or ("predefined" if labels.get("artifact") else None),
        "partition_mode": partition.get("mode"),
    }
    return not reasons, reasons, summary


def _catalogue_choices() -> list[dict]:
    payload = get_catalogue_client().list_datasets({"page": "1", "page_size": "100"})
    choices = []
    for raw in payload.get("items", []):
        item = _safe(raw)
        preliminary = []
        if item.get("status") != "available":
            preliminary.append("Version is not available.")
        if item.get("partition_mode") in {None, "none"}:
            preliminary.append("No active partition.")
        if not item.get("labels_available"):
            preliminary.append("No labels available.")
        if not item.get("usages", {}).get("labeled_evaluation", {}).get("supported"):
            preliminary.extend(item.get("usages", {}).get("labeled_evaluation", {}).get("reasons", []))
        item["readiness_reasons"] = list(dict.fromkeys(preliminary))
        item["preliminarily_ready"] = not item["readiness_reasons"]
        choices.append(item)
    return choices


def _load_selected_dataset(dataset_id: str, version_number: int) -> dict:
    catalogue = get_catalogue_client()
    detail = catalogue.get_dataset(dataset_id)
    version = catalogue.get_version(dataset_id, version_number)
    ready, reasons, summary = _dataset_readiness(version)
    return {
        "dataset_id": dataset_id,
        "display_name": detail.get("display_name") or dataset_id,
        "purpose": detail.get("purpose", []),
        "workload_context": version.get("workload_context") or detail.get("workload_context", {}),
        "version": version_number,
        "ready": ready,
        "reasons": reasons,
        **summary,
    }


def _compatible(detector: dict) -> tuple[bool, str]:
    training = _training_capability(detector)
    if training.get("enabled") is not True:
        return False, "Training is disabled for this detector."
    if training.get("protocol") != "cloudsentinel.training/v1":
        return False, "The training protocol is not supported by this UI."
    if training.get("input_profile") != "metrics-partition-v1":
        return False, "This detector does not accept the selected metrics partition profile."
    if "metrics" not in detector.get("supported_modalities", []):
        return False, "This detector does not support metrics data."
    return True, "Compatible with the selected labelled metrics partition."


def _decode_state(value: str | None) -> dict:
    if not value:
        return {"schema_version": 1, "nonce": uuid.uuid4().hex}
    try:
        state = signing.loads(value, salt=_WIZARD_SALT, max_age=3600)
    except signing.BadSignature as exc:
        raise ValueError("Training wizard state expired or is invalid. Start again.") from exc
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("Training wizard state is invalid. Start again.")
    return state


def _encode_state(state: dict) -> str:
    return signing.dumps(state, salt=_WIZARD_SALT, compress=True)


def _submitted_step(request: HttpRequest) -> int:
    try:
        return max(1, min(4, int(request.POST.get("step", "1") or 1)))
    except (TypeError, ValueError):
        return 1


def _parameter_fields(detector: dict) -> tuple[list[dict], list[dict]]:
    fields = []
    for name, metadata in detector.get("training_parameters", {}).items():
        if detector.get("id") == "cgnn" and name == "feature_importance":
            continue
        field = {"name": name, **metadata}
        default = metadata.get("default")
        field["default_text"] = "" if default is None else str(default).lower() if isinstance(default, bool) else str(default)
        fields.append(field)
    return ([field for field in fields if not field.get("advanced")], [field for field in fields if field.get("advanced")])


def _parse_parameter(raw: str | None, metadata: dict, name: str):
    if metadata.get("nullable") and raw in {None, "", "__none__"}:
        return None
    kind = metadata.get("type")
    try:
        if kind == "boolean":
            if raw not in {"true", "false"}:
                raise ValueError
            value = raw == "true"
        elif kind == "integer":
            value = int(raw)
        elif kind == "number":
            value = float(raw)
        else:
            value = "" if raw is None else str(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} has an invalid {kind} value.") from exc
    allowed = metadata.get("allowed_values")
    if allowed is not None and value not in allowed:
        raise ValueError(f"{name} must be one of the allowed values.")
    if value in metadata.get("excluded_values", []):
        raise ValueError(f"{name} uses an excluded value.")
    if kind in {"integer", "number"}:
        invalid = (
            ("minimum" in metadata and value < metadata["minimum"])
            or ("exclusive_minimum" in metadata and value <= metadata["exclusive_minimum"])
            or ("maximum" in metadata and value > metadata["maximum"])
            or ("exclusive_maximum" in metadata and value >= metadata["exclusive_maximum"])
        )
        if invalid:
            raise ValueError(f"{name} violates its numeric constraints.")
    return value


def _manifest_by_id() -> tuple[dict[str, dict], list[dict]]:
    detectors, errors = _detector_map()
    return {item["id"]: item for item in detectors}, errors


def _wizard_context(step: int, state: dict, *, error: str | None = None) -> dict:
    context = {"step": step, "state_token": _encode_state(state), "error": error, "selected_dataset": state.get("dataset"), "selected_ids": state.get("detector_ids", []), "parameters": state.get("parameters", {})}
    if step == 1:
        context["datasets"] = _catalogue_choices()
        context["preselected_dataset"] = state.get("preselected_dataset")
        context["preselected_version"] = state.get("preselected_version")
    elif step in {2, 3, 4}:
        manifests, discovery_errors = _manifest_by_id()
        detector_items = []
        for detector in manifests.values():
            compatible, reason = _compatible(detector)
            detector["compatible"] = compatible
            detector["compatibility_reason"] = reason
            detector_items.append(detector)
        context.update(detectors=detector_items, detector_map=manifests, discovery_errors=discovery_errors)
        if step == 3:
            configured = []
            for detector_id in state.get("detector_ids", []):
                detector = manifests.get(detector_id)
                if detector:
                    basic, advanced = _parameter_fields(detector)
                    configured.append({**detector, "basic_fields": basic, "advanced_fields": advanced})
            context["configured_detectors"] = configured
        if step == 4:
            context["review_detectors"] = [
                {"detector": manifests.get(detector_id, {"id": detector_id, "name": detector_id}), "parameters": state.get("parameters", {}).get(detector_id, {})}
                for detector_id in state.get("detector_ids", [])
            ]
    return context


@require_http_methods(["GET", "POST"])
def train_models(request: HttpRequest):
    try:
        if request.method == "GET":
            state = _decode_state(None)
            state["preselected_detector"] = request.GET.get("detector")
            state["preselected_dataset"] = request.GET.get("dataset")
            state["preselected_version"] = request.GET.get("version")
            return render(request, "config_app/models/train.html", _wizard_context(1, state))

        state = _decode_state(request.POST.get("wizard_state"))
        step = _submitted_step(request)
        action = request.POST.get("action", "continue")
        if action == "cancel":
            return redirect("models_overview")
        if action == "back":
            return render(request, "config_app/models/train.html", _wizard_context(max(1, step - 1), state))

        if step == 1:
            dataset_id = request.POST.get("dataset_id", "").strip()
            try:
                version = int(request.POST.get("version", ""))
            except ValueError as exc:
                raise ValueError("Select a valid dataset version.") from exc
            selected = _load_selected_dataset(dataset_id, version)
            if not selected["ready"]:
                raise ValueError("This dataset version is not ready: " + " ".join(selected["reasons"]))
            state["dataset"] = selected
            state["detector_ids"] = [state["preselected_detector"]] if state.get("preselected_detector") else []
            return render(request, "config_app/models/train.html", _wizard_context(2, state))

        if not state.get("dataset"):
            raise ValueError("Select a dataset before continuing.")
        # Recheck the exact immutable source on every subsequent step.
        selected = _load_selected_dataset(state["dataset"]["dataset_id"], int(state["dataset"]["version"]))
        if not selected["ready"] or selected.get("partition_id") != state["dataset"].get("partition_id") or selected.get("partition_checksum") != state["dataset"].get("partition_checksum"):
            raise ValueError("The selected dataset partition changed or is no longer training-ready. Start again.")
        state["dataset"] = selected
        manifests, _ = _manifest_by_id()

        if step == 2:
            selected_ids = list(dict.fromkeys(request.POST.getlist("detectors")))
            if not selected_ids:
                raise ValueError("Select at least one compatible detector.")
            for detector_id in selected_ids:
                detector = manifests.get(detector_id)
                if not detector:
                    raise ValueError(f"Detector {detector_id!r} is unavailable.")
                compatible, reason = _compatible(detector)
                if not compatible:
                    raise ValueError(reason)
            state["detector_ids"] = selected_ids
            return render(request, "config_app/models/train.html", _wizard_context(3, state))

        if step == 3:
            parameters = {}
            for detector_id in state.get("detector_ids", []):
                detector = manifests.get(detector_id)
                if not detector:
                    raise ValueError(f"Detector {detector_id!r} is unavailable.")
                detector_parameters = {}
                for name, metadata in detector.get("training_parameters", {}).items():
                    if detector_id == "cgnn" and name == "feature_importance":
                        detector_parameters[name] = False
                        continue
                    key = f"param__{detector_id}__{name}"
                    raw = request.POST.get(key)
                    if raw is None:
                        raw = metadata.get("default")
                        if isinstance(raw, bool):
                            raw = str(raw).lower()
                        elif raw is not None:
                            raw = str(raw)
                    detector_parameters[name] = _parse_parameter(raw, metadata, f"{detector_id}.{name}")
                parameters[detector_id] = detector_parameters
            state["parameters"] = parameters
            return render(request, "config_app/models/train.html", _wizard_context(4, state))

        if step == 4:
            nonce = state.get("nonce")
            cache_key = f"models-training-submit:{nonce}"
            if not nonce or not cache.add(cache_key, True, timeout=300):
                raise ValueError("This training request was already submitted.")
            try:
                dataset = state["dataset"]
                payload = {
                    "dataset": {"source": "catalogue", "dataset_id": dataset["dataset_id"], "version": dataset["version"], "partition_id": dataset["partition_id"]},
                    "detectors": [{"detector_id": detector_id, "parameters": state["parameters"][detector_id]} for detector_id in state["detector_ids"]],
                    "client_request_id": nonce,
                }
                run_name = request.POST.get("run_name", "").strip()
                if run_name:
                    payload["run_name"] = run_name
                result = get_learning_client().create_training_run(payload)
                run_id = result.get("run_id")
                if not isinstance(run_id, str) or not run_id:
                    raise LearningClientError("Learning service did not return a run ID.")
            except Exception:
                cache.delete(cache_key)
                raise
            messages.success(request, "Training run started.")
            return redirect("models_training_run_detail", run_id=run_id)
        raise ValueError("Invalid wizard step.")
    except (ValueError, KeyError, TypeError, CatalogueClientError, LearningClientError) as exc:
        try:
            state
        except UnboundLocalError:
            state = _decode_state(None)
        step = _submitted_step(request) if request.method == "POST" else 1
        try:
            context = _wizard_context(step, state, error=_error(exc))
        except (CatalogueClientError, LearningClientError):
            context = {"step": step, "error": _error(exc), "state_token": _encode_state(state)}
        return render(request, "config_app/models/train.html", context, status=400)


def _run_ui(run: dict) -> dict:
    run = _safe(run)
    for child in run.get("children", []):
        metadata = child.get("result_metadata") or {}
        child["promotion_status"] = (metadata.get("promotion") or {}).get("status") or child.get("model_status")
        child["display_detail"] = _child_progress_text(child)
    run["is_active"] = any(child.get("status") in _ACTIVE_CHILD_STATUSES for child in run.get("children", []))
    return run


def _child_progress_text(child: dict) -> str:
    """Render trusted, bounded task metadata without exposing object reprs."""
    phase = str(child.get("progress_phase") or "").strip().upper()
    phase_label = phase.replace("_", " ").title() if phase else ""
    detail = child.get("status_detail") or child.get("detail")
    if isinstance(detail, str):
        detail = detail.strip()
        if detail and phase_label and detail.casefold() != phase_label.casefold():
            return f"{phase_label} — {detail}"
        return detail or phase_label
    if isinstance(detail, dict):
        message = detail.get("message")
        if isinstance(message, str) and message.strip():
            return f"{phase_label} — {message.strip()}" if phase_label else message.strip()
        values = detail.get("values")
        if isinstance(values, list) and len(values) >= 4:
            outer, total_outer, inner, total_inner = values[:4]
            if all(isinstance(value, int) and not isinstance(value, bool) for value in values[:4]):
                outer_label = "Epoch" if phase in {"", "TRAINING"} else "Step"
                prefix = phase_label or "Training"
                return (
                    f"{prefix} — {outer_label} {outer + 1} of {total_outer} · "
                    f"batch {inner + 1} of {total_inner}"
                )
    return phase_label


@require_GET
def training_run_detail(request: HttpRequest, run_id: str):
    try:
        run = _run_ui(get_learning_client().get_training_run(run_id))
        return render(request, "config_app/models/run_detail.html", {"run": run})
    except LearningClientError as exc:
        if exc.status_code == 404:
            raise Http404("Training run not found") from exc
        return render(request, "config_app/models/run_detail.html", {"error": _error(exc), "retryable": exc.retryable}, status=502)


@require_GET
def training_run_status(request: HttpRequest, run_id: str):
    try:
        return JsonResponse(_run_ui(get_learning_client().get_training_run(run_id)))
    except LearningClientError as exc:
        return JsonResponse({"error": _error(exc), "retryable": exc.retryable}, status=exc.status_code)


@require_GET
def saved_model_detail(request: HttpRequest, model_id: str):
    try:
        model = _safe(get_learning_client().get_model(model_id))
        return render(request, "config_app/models/model_detail.html", {"model": model})
    except LearningClientError as exc:
        if exc.status_code == 404:
            raise Http404("Saved model not found") from exc
        return render(request, "config_app/models/model_detail.html", {"error": _error(exc), "retryable": exc.retryable}, status=502)
