"""Django pages and safe proxy endpoints for the Data Catalogue."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import re
from typing import Any
from urllib.parse import urlencode

from django.contrib import messages
from django.http import HttpRequest, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_POST

from ..catalogue_client import CatalogueClientError, get_catalogue_client
from ..workflow_context import encode_context, new_context, progress_context
from ..workload_context import (
    ANOMALY_SCENARIOS,
    ANOMALY_SCENARIO_DETAIL_MAX_LENGTH,
    ANOMALY_SCENARIO_HELP,
    DOMINANT_CHARACTERISTIC_HELP,
    INCIDENT_SCENARIOS,
    USER_SUPPLIED_HELP,
    WORKLOAD_CHARACTERISTICS,
    WORKLOAD_INTENSITIES,
    WORKLOAD_INTENSITY_DEFINITIONS,
)

DATASET_STATUSES = ["draft", "fetching", "validating", "available", "failed"]
FILTER_FIELDS = {
    "search",
    "status",
    "purpose",
    "workload_intensity",
    "dominant_workload_characteristic",
    "anomaly_scenario",
    "labels_available",
    "page",
    "page_size",
}
SENSITIVE_RESPONSE_KEYS = {
    "artifact",
    "artifacts",
    "path",
    "url",
    "base_url",
    "credentials",
    "token",
    "secret",
}
_URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def _safe_error(exc: Exception) -> str:
    return _URL_PATTERN.sub("[internal service]", str(exc))[:500]


def _catalogue_context() -> dict[str, Any]:
    return {
        "workload_intensities": WORKLOAD_INTENSITIES,
        "workload_characteristics": WORKLOAD_CHARACTERISTICS,
        "anomaly_scenarios": ANOMALY_SCENARIOS,
        "incident_scenarios": INCIDENT_SCENARIOS,
        "workload_intensity_definitions": WORKLOAD_INTENSITY_DEFINITIONS,
        "dominant_characteristic_help": DOMINANT_CHARACTERISTIC_HELP,
        "anomaly_scenario_help": ANOMALY_SCENARIO_HELP,
        "workload_user_supplied_help": USER_SUPPLIED_HELP,
        "anomaly_scenario_detail_max_length": ANOMALY_SCENARIO_DETAIL_MAX_LENGTH,
        "dataset_statuses": DATASET_STATUSES,
    }


def _split_values(value: str | None) -> list[str]:
    if not value:
        return []
    normalized = value.replace("\n", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def _utc_timestamp(value: str | None, field: str) -> str:
    if not value:
        raise ValueError(f"{field} is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid date and time") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _integer(value: str | None, field: str, *, minimum: int = 0) -> int:
    try:
        result = int(value or "")
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _queries(post: Any) -> list[dict[str, Any]]:
    raw = post.get("queries_json", "[]")
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Query configuration is not valid JSON") from exc
    if not isinstance(values, list) or not values:
        raise ValueError("At least one Prometheus query is required")
    queries = []
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            raise ValueError(f"Query {index} must be an object")
        query = {
            "query_id": value.get("query_id", "").strip(),
            "display_name": value.get("display_name", "").strip(),
            "promql_template": value.get("promql_template", "").strip(),
            "required": bool(value.get("required", False)),
            "execution_mode": value.get("execution_mode", "single"),
            "identity_labels": _split_values(value.get("identity_labels")),
            "target_context": {},
            "parameters": {},
        }
        for field in ("feature_name", "expected_modality", "expected_result_type"):
            submitted = value.get(field)
            if isinstance(submitted, str) and submitted.strip():
                query[field] = submitted.strip()
        target_type = value.get("target_type")
        if target_type:
            query["target_type"] = target_type
        for field in ("pod", "service"):
            submitted = value.get(f"target_context_{field}")
            if isinstance(submitted, str) and submitted.strip():
                query["target_context"][field] = submitted.strip()
        parameters = value.get("parameters")
        if isinstance(parameters, str) and parameters.strip():
            try:
                parameters = json.loads(parameters)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Query {index} parameters must be a JSON object") from exc
        if parameters:
            if not isinstance(parameters, dict):
                raise ValueError(f"Query {index} parameters must be a JSON object")
            query["parameters"] = parameters
        queries.append(query)
    return queries


def _workload_context(post: Any, prefix: str = "") -> dict[str, str]:
    intensity = post.get(f"{prefix}workload_intensity", "")
    characteristic = post.get(f"{prefix}dominant_workload_characteristic", "")
    scenario = post.get(f"{prefix}anomaly_scenario", "")
    if intensity not in WORKLOAD_INTENSITIES:
        raise ValueError("Select a valid workload intensity")
    if characteristic not in WORKLOAD_CHARACTERISTICS:
        raise ValueError("Select a valid dominant workload characteristic")
    if scenario not in ANOMALY_SCENARIOS:
        raise ValueError("Select a valid anomaly scenario")
    result = {
        "workload_intensity": intensity,
        "dominant_workload_characteristic": characteristic,
        "anomaly_scenario": scenario,
    }
    if scenario == "Other":
        detail = post.get(f"{prefix}anomaly_scenario_detail", "")
        if not isinstance(detail, str) or not detail.strip():
            raise ValueError("Custom scenario name is required when Other / custom is selected")
        detail = detail.strip()
        if len(detail) > ANOMALY_SCENARIO_DETAIL_MAX_LENGTH:
            raise ValueError(
                f"Custom scenario name must contain at most {ANOMALY_SCENARIO_DETAIL_MAX_LENGTH} characters"
            )
        result["anomaly_scenario_detail"] = detail
    return result


def _partition(post: Any, prefix: str = "") -> dict[str, Any]:
    mode = post.get(f"{prefix}partition_mode", "none")
    if mode == "none":
        return {"mode": "none"}
    if mode != "time_range":
        raise ValueError("Only none and time-range partitions can be authored")
    result = {
        "mode": "time_range",
        "train": {
            "start_time": _utc_timestamp(
                post.get(f"{prefix}train_start_time_utc")
                or post.get(f"{prefix}train_start_time"),
                "Train start time",
            ),
            "end_time": _utc_timestamp(
                post.get(f"{prefix}train_end_time_utc")
                or post.get(f"{prefix}train_end_time"),
                "Train end time",
            ),
        },
        "test": {
            "start_time": _utc_timestamp(
                post.get(f"{prefix}test_start_time_utc")
                or post.get(f"{prefix}test_start_time"),
                "Test start time",
            ),
            "end_time": _utc_timestamp(
                post.get(f"{prefix}test_end_time_utc")
                or post.get(f"{prefix}test_end_time"),
                "Test end time",
            ),
        },
        "gap_seconds": _integer(
            post.get(f"{prefix}gap_seconds", "0"), "Gap", minimum=0
        ),
    }
    if post.get(f"{prefix}labeled_evaluation") == "on":
        result["labels_source"] = "ground_truth"
    return result


def build_dataset_payload(post: Any) -> dict[str, Any]:
    purpose = post.getlist("purpose") if hasattr(post, "getlist") else []
    purpose.extend(_split_values(post.get("purpose_other")))
    purpose = list(dict.fromkeys(item for item in purpose if item))
    if not purpose:
        raise ValueError("Select or enter at least one purpose")
    incidents = []
    incident_start = post.get("incident_start_time_utc") or post.get("incident_start_time")
    incident_end = post.get("incident_end_time_utc") or post.get("incident_end_time")
    if incident_start or incident_end:
        incidents.append(
            {
                "start_time": _utc_timestamp(incident_start, "Incident start time"),
                "end_time": _utc_timestamp(incident_end, "Incident end time"),
                "scenario": post.get("incident_scenario", "Unknown"),
                "affected_services": _split_values(post.get("incident_services")),
                "affected_metrics": _split_values(post.get("incident_metrics")),
                "annotation": post.get("incident_annotation", ""),
                "source": "catalogue_request",
            }
        )
    return {
        "display_name": post.get("display_name", "").strip(),
        "description": post.get("description", ""),
        "purpose": purpose,
        "source": {
            "type": "prometheus",
            "prometheus_source_id": "cluster-default",
            "application": post.get("application", "").strip(),
            "namespace": post.get("namespace", "").strip(),
            "requested_targets": {
                "services": _split_values(post.get("services")),
                "pods": _split_values(post.get("pods")),
                "label_selectors": _split_values(post.get("label_selectors")),
            },
            "start_time": _utc_timestamp(
                post.get("start_time_utc") or post.get("start_time"), "Start time"
            ),
            "end_time": _utc_timestamp(
                post.get("end_time_utc") or post.get("end_time"), "End time"
            ),
            "sampling_interval_seconds": _integer(
                post.get("sampling_interval_seconds"),
                "Sampling interval",
                minimum=1,
            ),
            "queries": _queries(post),
        },
        "workload_context": _workload_context(post),
        "ground_truth": {
            "labels_available": False,
            "label_semantics": {},
            "known_incident_windows": incidents,
        },
        "partition": _partition(post),
    }


def build_version_payload(post: Any, copy_from_version: int) -> dict[str, Any]:
    return {
        "copy_from_version": copy_from_version,
        "source": {
            "application": post.get("application", "").strip(),
            "namespace": post.get("namespace", "").strip(),
            "requested_targets": {
                "services": _split_values(post.get("services")),
                "pods": _split_values(post.get("pods")),
                "label_selectors": _split_values(post.get("label_selectors")),
            },
            "start_time": _utc_timestamp(
                post.get("start_time_utc") or post.get("start_time"), "Start time"
            ),
            "end_time": _utc_timestamp(
                post.get("end_time_utc") or post.get("end_time"), "End time"
            ),
            "sampling_interval_seconds": _integer(
                post.get("sampling_interval_seconds"),
                "Sampling interval",
                minimum=1,
            ),
            "queries": _queries(post),
        },
        "workload_context": _workload_context(post),
        "partition": _partition(post),
    }


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _redact(item)
            for key, item in value.items()
            if key.lower() not in SENSITIVE_RESPONSE_KEYS
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _version_for_ui(version: dict[str, Any]) -> dict[str, Any]:
    result = _redact(deepcopy(version))
    hashes = version.get("checksums", {})
    result["checksum_identifiers"] = sorted(
        {item for item in hashes.values() if isinstance(item, str)}
    )
    result.pop("checksums", None)
    return result


def _form_partition(value: dict[str, Any] | None) -> dict[str, Any]:
    value = value or {}
    return {
        **value,
        "mode": value.get("mode", "none"),
        "train": value.get("train") or {"start_time": "", "end_time": ""},
        "test": value.get("test") or {"start_time": "", "end_time": ""},
        "gap_seconds": value.get("gap_seconds", 0),
    }


def _detail_url(dataset_id: str, version: int | None = None) -> str:
    url = reverse("data_catalogue_detail", args=[dataset_id])
    return f"{url}?version={version}" if version else url


@require_GET
def data_catalogue_overview(request: HttpRequest):
    params = {
        key: value
        for key in FILTER_FIELDS
        if (value := request.GET.get(key)) not in {None, ""}
    }
    context = {**_catalogue_context(), "filters": params, "catalogue": None}
    try:
        context["catalogue"] = _redact(get_catalogue_client().list_datasets(params))
    except CatalogueClientError as exc:
        context.update(error=_safe_error(exc), retryable=exc.retryable)
    return render(request, "config_app/data_catalogue/overview.html", context)


def data_catalogue_new(request: HttpRequest):
    try:
        initial_queries = json.loads(request.POST.get("queries_json", "[]")) if request.POST else []
    except json.JSONDecodeError:
        initial_queries = []
    default_namespace = ""
    try:
        default_namespace = get_catalogue_client().get_monitored_environment().get(
            "default_namespace", ""
        )
    except CatalogueClientError:
        pass
    context = {
        **_catalogue_context(),
        "submitted": request.POST,
        "source": {
            "application": "",
            "namespace": default_namespace,
            "services": "",
            "pods": "",
            "label_selectors": "",
            "start_time": "",
            "end_time": "",
            "sampling_interval_seconds": 60,
        },
        "partition": {
            "mode": "none",
            "train": {"start_time": "", "end_time": ""},
            "test": {"start_time": "", "end_time": ""},
            "gap_seconds": 0,
        },
        "initial_queries": initial_queries or [{"required": True, "execution_mode": "single"}],
    }
    if request.method == "POST":
        try:
            created = get_catalogue_client().create_dataset(build_dataset_payload(request.POST))
            dataset_id = created["dataset_id"]
            version = int(created["version"]["version"])
            if request.POST.get("action") == "save_and_fetch":
                try:
                    get_catalogue_client().start_fetch(dataset_id, version)
                    messages.success(request, "Dataset draft saved. Fetch has started.")
                except CatalogueClientError as exc:
                    messages.error(
                        request,
                        f"Dataset draft was saved, but fetch could not start: {exc}",
                    )
            else:
                messages.success(request, "Dataset draft saved.")
            return redirect(_detail_url(dataset_id, version))
        except (CatalogueClientError, ValueError, KeyError, TypeError) as exc:
            context["error"] = _safe_error(exc)
    return render(request, "config_app/data_catalogue/new.html", context)


@require_GET
def data_catalogue_detail(request: HttpRequest, dataset_id: str):
    try:
        raw_detail = get_catalogue_client().get_dataset(dataset_id)
        requested_version = request.GET.get("version")
        if requested_version is None:
            version_number = int(raw_detail["latest_version"])
            raw_version = raw_detail["version"]
        else:
            version_number = int(requested_version)
            raw_version = get_catalogue_client().get_version(dataset_id, version_number)
        if "usages" not in raw_version:
            raw_version = {**raw_version, "usages": raw_detail.get("usages", {})}
        detail = _redact(raw_detail)
        detail["version"] = _version_for_ui(raw_version)
        guided_context = new_context(
            dataset_id=dataset_id,
            dataset_version=version_number,
            dataset_name=detail.get("display_name"),
            partition_id=(raw_version.get("partition") or {}).get("partition_id"),
        )
        training_url = f'{reverse("models_train")}?{urlencode({"dataset": dataset_id, "version": version_number, "flow": encode_context(guided_context)})}'
        training_usage = (raw_version.get("usages") or {}).get("labeled_evaluation") or {}
        return render(
            request,
            "config_app/data_catalogue/detail.html",
            {
                **_catalogue_context(),
                "dataset": detail,
                "version": detail["version"],
                "selected_version": version_number,
                "selected_workload_intensity_help": dict(
                    WORKLOAD_INTENSITY_DEFINITIONS
                ).get((detail.get("workload_context") or {}).get("workload_intensity")),
                "workflow_progress": progress_context(guided_context, "dataset"),
                "guided_training_url": training_url,
                "guided_training_ready": bool(training_usage.get("supported")),
            },
        )
    except (CatalogueClientError, ValueError, KeyError, TypeError) as exc:
        status = exc.status_code if isinstance(exc, CatalogueClientError) else 400
        return render(
            request,
            "config_app/data_catalogue/detail.html",
            {"error": _safe_error(exc), "dataset_id": dataset_id},
            status=404 if status == 404 else 502 if status >= 500 else 400,
        )


@require_POST
def data_catalogue_fetch(request: HttpRequest, dataset_id: str, version: int):
    try:
        get_catalogue_client().start_fetch(dataset_id, version)
        messages.success(request, "Dataset fetch started.")
    except CatalogueClientError as exc:
        messages.error(request, _safe_error(exc))
    return redirect(_detail_url(dataset_id, version))


@require_GET
def data_catalogue_fetch_status(request: HttpRequest, dataset_id: str):
    try:
        version = _integer(request.GET.get("version"), "Version", minimum=1)
        return JsonResponse(_redact(get_catalogue_client().fetch_status(dataset_id, version)))
    except ValueError as exc:
        return JsonResponse({"error": _safe_error(exc), "retryable": False}, status=400)
    except CatalogueClientError as exc:
        return JsonResponse(
            {"error": _safe_error(exc), "retryable": exc.retryable}, status=exc.status_code
        )


@require_GET
def data_catalogue_preview(request: HttpRequest, dataset_id: str):
    try:
        version = _integer(request.GET.get("version"), "Version", minimum=1)
        limit = _integer(request.GET.get("limit", "20"), "Limit", minimum=1)
        return JsonResponse(
            _redact(get_catalogue_client().preview(dataset_id, version, limit))
        )
    except ValueError as exc:
        return JsonResponse({"error": _safe_error(exc), "retryable": False}, status=400)
    except CatalogueClientError as exc:
        return JsonResponse(
            {"error": _safe_error(exc), "retryable": exc.retryable}, status=exc.status_code
        )


@require_POST
def data_catalogue_metadata(request: HttpRequest, dataset_id: str):
    try:
        purpose = request.POST.getlist("purpose")
        purpose.extend(_split_values(request.POST.get("purpose_other")))
        payload = {
            "display_name": request.POST.get("display_name", "").strip(),
            "description": request.POST.get("description", ""),
            "purpose": list(dict.fromkeys(item for item in purpose if item)),
            "tags": _split_values(request.POST.get("tags")),
            "workload_context": _workload_context(request.POST, "metadata_"),
        }
        get_catalogue_client().update_metadata(dataset_id, payload)
        messages.success(request, "Dataset metadata updated.")
    except (CatalogueClientError, ValueError) as exc:
        messages.error(request, _safe_error(exc))
    return redirect(_detail_url(dataset_id, request.POST.get("version")))


@require_POST
def data_catalogue_incident(request: HttpRequest, dataset_id: str, version: int):
    try:
        payload = {
            "incident_id": request.POST.get("incident_id", "").strip() or None,
            "start_time": _utc_timestamp(
                request.POST.get("incident_start_time_utc")
                or request.POST.get("incident_start_time"),
                "Incident start time",
            ),
            "end_time": _utc_timestamp(
                request.POST.get("incident_end_time_utc")
                or request.POST.get("incident_end_time"),
                "Incident end time",
            ),
            "scenario": request.POST.get("incident_scenario", "Unknown"),
            "affected_services": _split_values(request.POST.get("incident_services")),
            "affected_metrics": _split_values(request.POST.get("incident_metrics")),
            "annotation": request.POST.get("incident_annotation", ""),
            "source": "manual",
            "allow_overlap": request.POST.get("allow_overlap") == "on",
        }
        if payload["incident_id"] is None:
            payload.pop("incident_id")
        get_catalogue_client().add_incident(dataset_id, version, payload)
        messages.success(request, "Incident window added.")
    except (CatalogueClientError, ValueError) as exc:
        messages.error(request, _safe_error(exc))
    return redirect(f"{_detail_url(dataset_id, version)}#labels")


@require_POST
def data_catalogue_derive_labels(request: HttpRequest, dataset_id: str, version: int):
    try:
        get_catalogue_client().derive_labels(dataset_id, version)
        messages.success(request, "Labels derived from incident windows.")
    except CatalogueClientError as exc:
        messages.error(request, _safe_error(exc))
    return redirect(f"{_detail_url(dataset_id, version)}#labels")


@require_POST
def data_catalogue_partition(request: HttpRequest, dataset_id: str, version: int):
    try:
        payload = _partition(request.POST, "new_")
        payload["labeled_evaluation"] = request.POST.get("new_labeled_evaluation") == "on"
        get_catalogue_client().create_partition(dataset_id, version, payload)
        messages.success(request, "Partition definition saved.")
    except (CatalogueClientError, ValueError) as exc:
        messages.error(request, _safe_error(exc))
    return redirect(f"{_detail_url(dataset_id, version)}#partitions")


def data_catalogue_new_version(request: HttpRequest, dataset_id: str):
    try:
        detail = get_catalogue_client().get_dataset(dataset_id)
        copy_from = int(request.GET.get("copy_from", detail["latest_version"]))
        current = get_catalogue_client().get_version(dataset_id, copy_from)
    except (CatalogueClientError, ValueError, KeyError, TypeError) as exc:
        return render(
            request,
            "config_app/data_catalogue/new_version.html",
            {"error": _safe_error(exc), "dataset_id": dataset_id},
            status=400,
        )
    context = {
        **_catalogue_context(),
        "dataset": _redact(detail),
        "source_version": _version_for_ui(current),
        "source": {
            **current.get("source", {}),
            "services": ", ".join(current.get("source", {}).get("requested_targets", {}).get("services", [])),
            "pods": ", ".join(current.get("source", {}).get("requested_targets", {}).get("pods", [])),
            "label_selectors": ", ".join(current.get("source", {}).get("requested_targets", {}).get("label_selectors", [])),
        },
        "partition": _form_partition(current.get("partition")),
        "copy_from": copy_from,
        "initial_queries": current.get("source", {}).get("queries", []),
    }
    if request.method == "POST":
        try:
            created = get_catalogue_client().create_version(
                dataset_id, build_version_payload(request.POST, copy_from)
            )
            version = int(created["latest_version"])
            messages.success(request, "New draft version created. Fetch it when ready.")
            return redirect(_detail_url(dataset_id, version))
        except (CatalogueClientError, ValueError, KeyError, TypeError) as exc:
            context.update(error=_safe_error(exc), submitted=request.POST)
            try:
                context["initial_queries"] = json.loads(request.POST.get("queries_json", "[]"))
            except json.JSONDecodeError:
                context["initial_queries"] = []
    return render(request, "config_app/data_catalogue/new_version.html", context)
