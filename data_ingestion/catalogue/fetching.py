"""One-shot Prometheus fetch, canonical assembly, and artifact promotion."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable

import requests

from .models import utc_now, version_summary
from .repository import FileCatalogueRepository, atomic_write_json
from .validation import (
    CatalogueValidationError,
    validate_identifier,
    validate_timestamp,
    validate_version,
)


class CatalogueFetchError(RuntimeError):
    """Safe, user-visible failure for a catalogue fetch attempt."""


@dataclass(frozen=True)
class FetchLimits:
    maximum_query_count: int
    maximum_query_length: int
    maximum_window_seconds: int
    minimum_sampling_interval_seconds: int
    maximum_theoretical_samples: int
    maximum_series: int
    maximum_response_bytes: int
    maximum_preview_rows: int
    connect_timeout_seconds: float
    read_timeout_seconds: float
    maximum_retries: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FetchLimits":
        return cls(
            maximum_query_count=int(config["CATALOGUE_MAX_QUERY_COUNT"]),
            maximum_query_length=int(config["CATALOGUE_MAX_QUERY_LENGTH"]),
            maximum_window_seconds=int(config["CATALOGUE_MAX_WINDOW_SECONDS"]),
            minimum_sampling_interval_seconds=int(
                config["CATALOGUE_MIN_SAMPLING_INTERVAL_SECONDS"]
            ),
            maximum_theoretical_samples=int(
                config["CATALOGUE_MAX_THEORETICAL_SAMPLES"]
            ),
            maximum_series=int(config["CATALOGUE_MAX_SERIES"]),
            maximum_response_bytes=int(config["CATALOGUE_MAX_RESPONSE_BYTES"]),
            maximum_preview_rows=int(config["CATALOGUE_MAX_PREVIEW_ROWS"]),
            connect_timeout_seconds=float(config["CATALOGUE_CONNECT_TIMEOUT_SECONDS"]),
            read_timeout_seconds=float(config["CATALOGUE_READ_TIMEOUT_SECONDS"]),
            maximum_retries=int(config["CATALOGUE_MAX_RETRIES"]),
        )


_TEMPLATE_VARIABLE = re.compile(r"\$\{([^}]+)\}")
_ALLOWED_TEMPLATE_VARIABLES = {"namespace", "pod", "service"}


def _escape_promql_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _render_template(template: str, context: dict[str, str]) -> str:
    matches = list(_TEMPLATE_VARIABLE.finditer(template))
    variables = {match.group(1) for match in matches}
    unknown = variables - _ALLOWED_TEMPLATE_VARIABLES
    if unknown:
        raise CatalogueValidationError(
            f"query template contains unsupported variables: {', '.join(sorted(unknown))}"
        )
    missing = variables - set(context)
    if missing:
        raise CatalogueValidationError(
            f"query template is missing target context for: {', '.join(sorted(missing))}"
        )
    for match in matches:
        escaped = False
        inside_string = False
        for character in template[: match.start()]:
            if character == '"' and not escaped:
                inside_string = not inside_string
            escaped = character == "\\" and not escaped
            if character != "\\":
                escaped = False
        if not inside_string:
            raise CatalogueValidationError(
                "query template variables must be used inside PromQL string literals"
            )
    return _TEMPLATE_VARIABLE.sub(
        lambda match: _escape_promql_label_value(context[match.group(1)]), template
    )


def resolve_prometheus_source(source_id: str, config: dict[str, Any]) -> str:
    source_id = validate_identifier(source_id, "prometheus_source_id")
    sources = config.get("CATALOGUE_PROMETHEUS_SOURCES", {})
    if not isinstance(sources, dict) or source_id not in sources:
        raise CatalogueValidationError(f"unknown Prometheus source: {source_id!r}")
    base_url = sources[source_id]
    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
        raise CatalogueValidationError(
            f"Prometheus source {source_id!r} is not configured with an HTTP URL"
        )
    return base_url.rstrip("/")


def build_query_executions(
    version: dict[str, Any], limits: FetchLimits
) -> list[dict[str, Any]]:
    source = version.get("source", {})
    if source.get("type") != "prometheus":
        raise CatalogueValidationError("only Prometheus catalogue versions can be fetched")
    queries = source.get("queries", [])
    if not queries:
        raise CatalogueValidationError("at least one Prometheus query is required")
    if len(queries) > limits.maximum_query_count:
        raise CatalogueValidationError("query count exceeds the configured maximum")
    _start_value, start = validate_timestamp(source.get("start_time"), "source.start_time")
    _end_value, end = validate_timestamp(source.get("end_time"), "source.end_time")
    window_seconds = (end - start).total_seconds()
    if window_seconds <= 0 or window_seconds > limits.maximum_window_seconds:
        raise CatalogueValidationError("source time window exceeds the configured maximum")
    step = source.get("sampling_interval_seconds")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step < limits.minimum_sampling_interval_seconds
    ):
        raise CatalogueValidationError(
            "sampling interval is below the configured minimum"
        )

    namespace = source.get("namespace")
    targets = source.get("requested_targets", {})
    executions: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        template = query.get("promql_template", query.get("promql", ""))
        if len(template) > limits.maximum_query_length:
            raise CatalogueValidationError(
                f"query {query.get('query_id')!r} exceeds the configured maximum length"
            )
        mode = query.get("execution_mode", "single")
        contexts: list[tuple[dict[str, str], dict[str, str] | None]] = []
        if mode == "single":
            context = {"namespace": namespace, **query.get("target_context", {})}
            contexts.append((context, query.get("target_context") or None))
        elif mode == "per_target":
            target_type = query.get("target_type")
            target_key = "pods" if target_type == "pod" else "services"
            resolved_targets = sorted(set(targets.get(target_key, [])))
            if not resolved_targets:
                raise CatalogueValidationError(
                    f"query {query.get('query_id')!r} has no {target_type} targets"
                )
            for target in resolved_targets:
                contexts.append(
                    (
                        {"namespace": namespace, target_type: target},
                        {target_type: target},
                    )
                )
        else:
            raise CatalogueValidationError("unsupported query execution mode")

        for target_index, (context, resolved_target) in enumerate(contexts):
            executions.append(
                {
                    "execution_id": f"q{query_index:04d}-t{target_index:04d}",
                    "query_id": query["query_id"],
                    "display_name": query["display_name"],
                    "feature_name": query.get("feature_name") or query["query_id"],
                    "required": query.get("required", True),
                    "execution_mode": mode,
                    "target_type": query.get("target_type"),
                    "identity_labels": query.get("identity_labels", []),
                    "resolved_target": resolved_target,
                    "query_template": template,
                    "resolved_query": _render_template(template, context),
                }
            )
    grid_count = math.floor(window_seconds / step) + 1
    if grid_count * len(executions) > limits.maximum_theoretical_samples:
        raise CatalogueValidationError(
            "theoretical sample count exceeds the configured maximum"
        )
    return executions


def validate_fetch_request(version: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    limits = FetchLimits.from_config(config)
    resolve_prometheus_source(version.get("source", {}).get("prometheus_source_id"), config)
    return build_query_executions(version, limits)


class PrometheusRangeClient:
    def __init__(
        self,
        base_url: str,
        limits: FetchLimits,
        request_get: Callable[..., Any] = requests.get,
    ):
        self.endpoint = f"{base_url}/api/v1/query_range"
        self.limits = limits
        self.request_get = request_get

    def query_range(self, query: str, start: str, end: str, step: int) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.limits.maximum_retries + 1):
            try:
                response = self.request_get(
                    self.endpoint,
                    params={"query": query, "start": start, "end": end, "step": step},
                    timeout=(
                        self.limits.connect_timeout_seconds,
                        self.limits.read_timeout_seconds,
                    ),
                    stream=True,
                )
                try:
                    if response.status_code < 200 or response.status_code >= 300:
                        raise CatalogueFetchError(
                            f"Prometheus returned HTTP {response.status_code}"
                        )
                    chunks = []
                    size = 0
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        size += len(chunk)
                        if size > self.limits.maximum_response_bytes:
                            raise CatalogueFetchError(
                                "Prometheus response exceeds the configured size limit"
                            )
                        chunks.append(chunk)
                finally:
                    close = getattr(response, "close", None)
                    if callable(close):
                        close()
                try:
                    payload = json.loads(b"".join(chunks).decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise CatalogueFetchError("Prometheus returned invalid JSON") from exc
                if payload.get("status") != "success":
                    error_type = payload.get("errorType", "unknown")
                    raise CatalogueFetchError(
                        f"Prometheus API reported an error ({error_type})"
                    )
                result = payload.get("data", {}).get("result")
                if not isinstance(result, list):
                    raise CatalogueFetchError("Prometheus response has no result list")
                if len(result) > self.limits.maximum_series:
                    raise CatalogueFetchError(
                        "Prometheus response exceeds the configured series limit"
                    )
                warnings = payload.get("warnings", [])
                if not isinstance(warnings, list):
                    warnings = []
                return {
                    "payload": payload,
                    "warnings": [str(item) for item in warnings],
                    "response_bytes": size,
                }
            except (requests.RequestException, CatalogueFetchError) as exc:
                last_error = exc
                if attempt >= self.limits.maximum_retries:
                    break
        if isinstance(last_error, CatalogueFetchError):
            raise last_error
        raise CatalogueFetchError("Prometheus request failed or timed out") from last_error


def _series_sort_key(series: dict[str, Any]) -> str:
    return json.dumps(series.get("metric", {}), sort_keys=True, separators=(",", ":"))


def _feature_id(execution: dict[str, Any], metric: dict[str, Any], series_count: int) -> str:
    parts = [execution["feature_name"]]
    target = execution.get("resolved_target")
    if target:
        for target_type in sorted(target):
            parts.append(f"target.{target_type}={target[target_type]}")
    labels = execution.get("identity_labels") or sorted(metric)
    if series_count > 1 or execution.get("identity_labels"):
        for label in labels:
            if label not in metric:
                raise CatalogueFetchError(
                    f"series is missing configured identity label {label!r}"
                )
            parts.append(f"{label}={metric[label]}")
    return "|".join(parts)


def assemble_time_series(
    version: dict[str, Any], results: list[dict[str, Any]], limits: FetchLimits
) -> dict[str, Any]:
    source = version["source"]
    _start_value, start = validate_timestamp(source["start_time"], "source.start_time")
    _end_value, end = validate_timestamp(source["end_time"], "source.end_time")
    step = source["sampling_interval_seconds"]
    grid: list[datetime] = []
    current = start
    while current <= end:
        grid.append(current)
        current += timedelta(seconds=step)
    grid_keys = {round(item.timestamp() * 1_000_000): index for index, item in enumerate(grid)}

    warnings: list[str] = []
    features: list[str] = []
    columns: list[list[float | None]] = []
    total_series = 0
    provenance_executions = []
    for result in results:
        execution = result["execution"]
        raw_result = result["response"]["payload"]["data"]["result"]
        raw_result = sorted(raw_result, key=_series_sort_key)
        if not raw_result:
            message = f"query {execution['query_id']!r} returned no series"
            if execution["required"]:
                raise CatalogueFetchError(message)
            warnings.append(message)
        total_series += len(raw_result)
        if total_series > limits.maximum_series:
            raise CatalogueFetchError("combined result exceeds the configured series limit")
        sample_count = 0
        for series in raw_result:
            metric = series.get("metric", {})
            values = series.get("values", [])
            if not isinstance(metric, dict) or not isinstance(values, list):
                raise CatalogueFetchError("Prometheus series has an invalid shape")
            feature_id = _feature_id(execution, metric, len(raw_result))
            if feature_id in features:
                raise CatalogueFetchError(f"feature identity collision: {feature_id!r}")
            column: list[float | None] = [None] * len(grid)
            for sample in values:
                if not isinstance(sample, list) or len(sample) != 2:
                    raise CatalogueFetchError("Prometheus sample has an invalid shape")
                try:
                    timestamp_key = round(float(sample[0]) * 1_000_000)
                    number = float(sample[1])
                except (TypeError, ValueError) as exc:
                    raise CatalogueFetchError("Prometheus sample is not numeric") from exc
                position = grid_keys.get(timestamp_key)
                if position is None:
                    warnings.append(
                        f"query {execution['query_id']!r} returned an off-grid sample"
                    )
                    continue
                if math.isfinite(number):
                    column[position] = number
                else:
                    warnings.append(
                        f"query {execution['query_id']!r} returned a non-finite sample"
                    )
                sample_count += 1
            features.append(feature_id)
            columns.append(column)
        warnings.extend(result["response"]["warnings"])
        provenance_executions.append(
            {
                "execution_id": execution["execution_id"],
                "query_id": execution["query_id"],
                "query_template": execution["query_template"],
                "resolved_query": execution["resolved_query"],
                "resolved_target": execution["resolved_target"],
                "start_time": source["start_time"],
                "end_time": source["end_time"],
                "step_seconds": step,
                "duration_seconds": (
                    validate_timestamp(source["end_time"], "end")[1]
                    - validate_timestamp(source["start_time"], "start")[1]
                ).total_seconds(),
                "series_count": len(raw_result),
                "sample_count": sample_count,
                "response_bytes": result["response"]["response_bytes"],
                "warnings": result["response"]["warnings"],
                "status": "success",
            }
        )
    if not features:
        raise CatalogueFetchError("fetch produced no features")
    rows = [
        [columns[column][row] for column in range(len(columns))]
        for row in range(len(grid))
    ]
    missing_by_feature = {
        feature: sum(1 for value in columns[index] if value is None)
        for index, feature in enumerate(features)
    }
    return {
        "timestamps": [item.isoformat().replace("+00:00", "Z") for item in grid],
        "features": features,
        "rows": rows,
        "warnings": list(dict.fromkeys(warnings)),
        "missing_by_feature": missing_by_feature,
        "provenance_executions": provenance_executions,
    }


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if item is None else item for item in row])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_error(exc: Exception) -> dict[str, str]:
    if isinstance(exc, (CatalogueFetchError, CatalogueValidationError)):
        message = str(exc)
    else:
        message = "unexpected catalogue fetch failure"
    return {"type": type(exc).__name__, "message": message[:500]}


def _sync_record(repository: FileCatalogueRepository, version: dict[str, Any]) -> None:
    record = repository.read_record(version["dataset_id"])
    record["status"] = version["status"]
    record["technical_schema"] = version["technical_schema"]
    record["versions"] = [
        version_summary(version) if item["version"] == version["version"] else item
        for item in record["versions"]
    ]
    repository.update_record(version["dataset_id"], record)


def update_fetch_state(
    repository: FileCatalogueRepository,
    dataset_id: str,
    version_number: int,
    attempt_id: str,
    *,
    status: str,
    phase: str,
    completed: int,
    total: int,
    warnings: list[str] | None = None,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    version = repository.read_version(dataset_id, version_number)
    fetch = version["fetch"]
    fetch.update(
        {
            "status": status,
            "phase": phase,
            "progress": {"completed": completed, "total": total},
            "warnings": warnings or [],
            "error": error,
            "updated_at": utc_now(),
        }
    )
    version["status"] = status
    repository.update_version(dataset_id, version_number, version)
    attempt = repository.read_attempt(dataset_id, version_number, attempt_id)
    attempt.update(fetch)
    repository.write_attempt(dataset_id, version_number, attempt_id, attempt)
    _sync_record(repository, version)
    return version


def start_fetch_attempt(
    repository: FileCatalogueRepository,
    dataset_id: str,
    version_number: int,
    attempt_id: str,
    task_id: str,
    total_executions: int,
) -> dict[str, Any]:
    validate_identifier(dataset_id)
    validate_version(version_number)
    validate_identifier(attempt_id, "attempt_id")
    validate_identifier(task_id, "fetch_task_id")
    version = repository.read_version(dataset_id, version_number)
    if version["status"] not in {"draft", "failed"}:
        raise CatalogueValidationError("only draft or failed versions can be fetched")
    if repository.artifact_directory(dataset_id, version_number).exists():
        raise CatalogueValidationError("dataset version already has active artifacts")
    now = utc_now()
    fetch = {
        "attempt_id": attempt_id,
        "task_id": task_id,
        "status": "fetching",
        "phase": "fetching",
        "progress": {"completed": 0, "total": total_executions},
        "warnings": [],
        "error": None,
        "started_at": now,
        "updated_at": now,
        "completed_at": None,
    }
    version["status"] = "fetching"
    version["fetch"] = fetch
    version["provenance"]["configuration_frozen"] = True
    repository.update_version(dataset_id, version_number, version)
    repository.write_attempt(
        dataset_id,
        version_number,
        attempt_id,
        {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "version": version_number,
            **fetch,
        },
    )
    _sync_record(repository, version)
    return version


def fail_fetch_attempt(
    repository: FileCatalogueRepository,
    dataset_id: str,
    version_number: int,
    attempt_id: str,
    exc: Exception,
) -> None:
    try:
        version = repository.read_version(dataset_id, version_number)
        progress = version.get("fetch", {}).get("progress", {"completed": 0, "total": 0})
        safe_error = _safe_error(exc)
        failed = update_fetch_state(
            repository,
            dataset_id,
            version_number,
            attempt_id,
            status="failed",
            phase="failed",
            completed=progress["completed"],
            total=progress["total"],
            warnings=version.get("fetch", {}).get("warnings", []),
            error=safe_error,
        )
        failed["validation"] = {
            "status": "failed",
            "warnings": failed.get("fetch", {}).get("warnings", []),
            "errors": [safe_error["message"]],
        }
        failed.setdefault("provenance", {}).setdefault("fetch_errors", []).append(safe_error)
        failed["fetch"]["completed_at"] = utc_now()
        repository.update_version(dataset_id, version_number, failed)
        _sync_record(repository, failed)
    except Exception:
        # Preserve the original exception; a corrupt state must not hide the fetch cause.
        pass


def execute_catalogue_fetch(
    dataset_id: str,
    version_number: int,
    attempt_id: str,
    config: dict[str, Any],
    *,
    request_get: Callable[..., Any] = requests.get,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    repository = FileCatalogueRepository(config["CATALOGUE_STORAGE_ROOT"])
    version = repository.read_version(dataset_id, version_number)
    if version.get("fetch", {}).get("attempt_id") != attempt_id:
        raise CatalogueFetchError("fetch attempt is no longer active")
    limits = FetchLimits.from_config(config)
    executions = validate_fetch_request(version, config)
    total = len(executions)
    base_url = resolve_prometheus_source(
        version["source"]["prometheus_source_id"], config
    )
    client = PrometheusRangeClient(base_url, limits, request_get=request_get)
    staging = repository.staging_directory(dataset_id, version_number, attempt_id)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    promoted = False
    source = version["source"]
    results = []
    progress_warnings: list[str] = []
    try:
        atomic_write_json(
            staging / "fetch_request.json",
            {
                "schema_version": 1,
                "dataset_id": dataset_id,
                "version": version_number,
                "attempt_id": attempt_id,
                "prometheus_source_id": source["prometheus_source_id"],
                "start_time": source["start_time"],
                "end_time": source["end_time"],
                "sampling_interval_seconds": source["sampling_interval_seconds"],
                "queries": source["queries"],
            },
        )
        for index, execution in enumerate(executions, start=1):
            response = client.query_range(
                execution["resolved_query"],
                source["start_time"],
                source["end_time"],
                source["sampling_interval_seconds"],
            )
            atomic_write_json(
                staging / "raw" / f"{execution['execution_id']}.json",
                response["payload"],
            )
            results.append({"execution": execution, "response": response})
            progress_warnings.extend(response["warnings"])
            if sum(item["response"]["response_bytes"] for item in results) > limits.maximum_response_bytes:
                raise CatalogueFetchError(
                    "combined Prometheus responses exceed the configured size limit"
                )
            update_fetch_state(
                repository,
                dataset_id,
                version_number,
                attempt_id,
                status="fetching",
                phase="fetching",
                completed=index,
                total=total,
                warnings=list(dict.fromkeys(progress_warnings)),
            )
            if progress_callback:
                progress_callback("fetching", index, total)

        update_fetch_state(
            repository,
            dataset_id,
            version_number,
            attempt_id,
            status="validating",
            phase="validating",
            completed=total,
            total=total,
        )
        if progress_callback:
            progress_callback("validating", total, total)
        assembled = assemble_time_series(version, results, limits)
        observations = [
            [timestamp, *row]
            for timestamp, row in zip(assembled["timestamps"], assembled["rows"])
        ]
        _write_csv(
            staging / "canonical" / "observations.csv",
            ["timestamp", *assembled["features"]],
            observations,
        )
        _write_csv(
            staging / "canonical" / "feature_matrix.csv",
            assembled["features"],
            assembled["rows"],
        )
        _write_csv(
            staging / "canonical" / "timestamps.csv",
            ["timestamp"],
            [[item] for item in assembled["timestamps"]],
        )
        schema = {
            "schema_version": 1,
            "modality": "metrics",
            "timestamp_column": "timestamp",
            "timestamp_timezone": "UTC",
            "observation_count": len(assembled["rows"]),
            "feature_count": len(assembled["features"]),
            "feature_identity": "named",
            "feature_order": assembled["features"],
            "missing_values": {
                "total": sum(assembled["missing_by_feature"].values()),
                "by_feature": assembled["missing_by_feature"],
            },
            "normalization": "none",
            "imputation": "none",
        }
        provenance = {
            "schema_version": 1,
            "prometheus_source_id": source["prometheus_source_id"],
            "fetch_timestamp": utc_now(),
            "executions": assembled["provenance_executions"],
            "warnings": assembled["warnings"],
        }
        validation = {
            "status": "passed_with_warnings" if assembled["warnings"] else "passed",
            "warnings": assembled["warnings"],
            "errors": [],
        }
        attempt = repository.read_attempt(dataset_id, version_number, attempt_id)
        attempt.update(
            {
                "status": "available",
                "phase": "available",
                "progress": {"completed": total, "total": total},
                "warnings": assembled["warnings"],
                "error": None,
                "completed_at": utc_now(),
                "updated_at": utc_now(),
            }
        )
        atomic_write_json(staging / "provenance.json", provenance)
        atomic_write_json(staging / "schema.json", schema)
        atomic_write_json(staging / "validation.json", validation)
        atomic_write_json(staging / "attempt.json", attempt)
        if version.get("ground_truth", {}).get("labels_available") or version.get(
            "incident_context"
        ):
            atomic_write_json(
                staging / "ground_truth.json",
                {
                    "ground_truth": version.get("ground_truth", {}),
                    "incident_context": version.get("incident_context", []),
                },
            )

        checksums = {}
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            checksums[str(path.relative_to(staging))] = _sha256(path)
        atomic_write_json(staging / "checksums.json", checksums)
        repository.promote_staging(dataset_id, version_number, attempt_id)
        promoted = True

        active = repository.artifact_directory(dataset_id, version_number)
        version = repository.read_version(dataset_id, version_number)
        version["status"] = "available"
        version["source"]["fetch_timestamp"] = provenance["fetch_timestamp"]
        resolved_targets = {"services": [], "pods": [], "label_selectors": []}
        for execution in executions:
            target = execution.get("resolved_target")
            if not target:
                continue
            for target_type, target_value in sorted(target.items()):
                key = "pods" if target_type == "pod" else "services"
                if target_value not in resolved_targets[key]:
                    resolved_targets[key].append(target_value)
        version["source"]["resolved_targets"] = resolved_targets
        lineage = {
            key: version.get("provenance", {})[key]
            for key in ("copied_from_version", "origin")
            if key in version.get("provenance", {})
        }
        version["provenance"] = {
            **lineage,
            "configuration_frozen": True,
            "resolved_queries": assembled["provenance_executions"],
            "query_results": [
                {
                    "execution_id": item["execution_id"],
                    "series_count": item["series_count"],
                    "sample_count": item["sample_count"],
                    "status": item["status"],
                }
                for item in assembled["provenance_executions"]
            ],
            "fetch_errors": [],
            "query_warnings": assembled["warnings"],
        }
        version["technical_schema"] = schema
        version["validation"] = validation
        version["checksums"] = checksums
        version["artifacts"] = {
            str(path.relative_to(active)): {
                "path": str(path.relative_to(repository.root)),
                "sha256": checksums[str(path.relative_to(active))],
                "bytes": path.stat().st_size,
            }
            for path in sorted(item for item in active.rglob("*") if item.is_file())
            if path.name != "checksums.json"
        }
        version["artifacts"]["checksums.json"] = {
            "path": str((active / "checksums.json").relative_to(repository.root)),
            "bytes": (active / "checksums.json").stat().st_size,
        }
        version["fetch"] = {
            **version["fetch"],
            "status": "available",
            "phase": "available",
            "progress": {"completed": total, "total": total},
            "warnings": assembled["warnings"],
            "error": None,
            "updated_at": utc_now(),
            "completed_at": attempt["completed_at"],
        }
        repository.update_version(dataset_id, version_number, version)
        repository.write_attempt(dataset_id, version_number, attempt_id, attempt)
        _sync_record(repository, version)
        if progress_callback:
            progress_callback("available", total, total)
        return {
            "status": "available",
            "dataset_id": dataset_id,
            "version": version_number,
            "attempt_id": attempt_id,
            "observation_count": schema["observation_count"],
            "feature_count": schema["feature_count"],
        }
    except Exception as exc:
        active = repository.artifact_directory(dataset_id, version_number)
        if promoted and active.is_dir() and not staging.exists():
            try:
                staging.parent.mkdir(parents=True, exist_ok=True)
                os.replace(active, staging)
            except OSError:
                pass
        fail_fetch_attempt(repository, dataset_id, version_number, attempt_id, exc)
        raise


def read_preview(
    repository: FileCatalogueRepository,
    dataset_id: str,
    version_number: int,
    limit: int,
) -> dict[str, Any]:
    version = repository.read_version(dataset_id, version_number)
    if version["status"] != "available":
        raise CatalogueValidationError("preview is available only for available versions")
    observations = repository.artifact_directory(dataset_id, version_number) / "canonical" / "observations.csv"
    if not observations.is_file():
        raise CatalogueFetchError("canonical observations artifact is missing")
    rows = []
    with observations.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            values = {}
            missing = {}
            for feature in version["technical_schema"]["feature_order"]:
                raw = row[feature]
                missing[feature] = raw == ""
                values[feature] = None if raw == "" else float(raw)
            rows.append(
                {"timestamp": row["timestamp"], "values": values, "missing": missing}
            )
            if len(rows) >= limit:
                break
    schema = version["technical_schema"]
    return {
        "dataset_id": dataset_id,
        "version": version_number,
        "feature_names": schema["feature_order"],
        "observation_count": schema["observation_count"],
        "feature_count": schema["feature_count"],
        "returned_rows": len(rows),
        "rows": rows,
    }
