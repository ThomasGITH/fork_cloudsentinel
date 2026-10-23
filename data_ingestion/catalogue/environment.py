"""Persistent runtime settings for the monitored environment."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from .repository import atomic_write_json
from .validation import CatalogueValidationError


_NAMESPACE = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_SETTINGS_FILE = "monitored_environment.json"


def _settings_path(config: dict[str, Any]) -> Path:
    return Path(config["CATALOGUE_STORAGE_ROOT"]).resolve() / _SETTINGS_FILE


def validate_prometheus_url(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogueValidationError("Prometheus URL is required")
    candidate = value.strip().rstrip("/")
    parsed = urlsplit(candidate)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise CatalogueValidationError(
            "Prometheus URL must be an http(s) base URL without credentials, a query, or a fragment"
        )
    return candidate


def validate_namespace(value: Any) -> str:
    if not isinstance(value, str) or not _NAMESPACE.fullmatch(value.strip()):
        raise CatalogueValidationError("Default namespace is not a valid Kubernetes namespace")
    return value.strip()


def default_environment(config: dict[str, Any]) -> dict[str, Any]:
    sources = config.get("CATALOGUE_PROMETHEUS_SOURCES", {})
    return {
        "schema_version": 1,
        "prometheus_url": validate_prometheus_url(sources.get("cluster-default")),
        "default_namespace": validate_namespace(
            config.get("CATALOGUE_DEFAULT_NAMESPACE", "default")
        ),
        "source_id": "cluster-default",
    }


def read_environment(config: dict[str, Any]) -> dict[str, Any]:
    fallback = default_environment(config)
    path = _settings_path(config)
    if not path.exists():
        return fallback
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        return {
            **fallback,
            "prometheus_url": validate_prometheus_url(stored.get("prometheus_url")),
            "default_namespace": validate_namespace(stored.get("default_namespace")),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback


def write_environment(config: dict[str, Any], payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CatalogueValidationError("A JSON settings object is required")
    value = {
        "schema_version": 1,
        "source_id": "cluster-default",
        "prometheus_url": validate_prometheus_url(payload.get("prometheus_url")),
        "default_namespace": validate_namespace(payload.get("default_namespace")),
    }
    atomic_write_json(_settings_path(config), value)
    return value


def effective_prometheus_sources(config: dict[str, Any]) -> dict[str, str]:
    sources = dict(config.get("CATALOGUE_PROMETHEUS_SOURCES", {}))
    sources["cluster-default"] = read_environment(config)["prometheus_url"]
    return sources


def test_prometheus_connection(
    config: dict[str, Any],
    prometheus_url: Any | None = None,
    *,
    request_get: Any = requests.get,
) -> dict[str, Any]:
    base_url = validate_prometheus_url(
        prometheus_url if prometheus_url is not None else read_environment(config)["prometheus_url"]
    )
    try:
        response = request_get(
            f"{base_url}/api/v1/query",
            params={"query": "up"},
            timeout=(3.0, 8.0),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("status") != "success":
            raise ValueError("invalid Prometheus response")
        result = ((payload.get("data") or {}).get("result") or [])
        target_count = len(result) if isinstance(result, list) else 0
        return {
            "status": "connected",
            "target_count": target_count,
            "message": (
                f"Prometheus responded with {target_count} active series."
                if target_count
                else "Prometheus accepted the query."
            ),
        }
    except (requests.RequestException, ValueError, TypeError) as exc:
        raise CatalogueValidationError(
            "CloudSentinel could not connect to this Prometheus URL"
        ) from exc
