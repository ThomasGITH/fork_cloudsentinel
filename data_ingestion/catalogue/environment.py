"""Persistent runtime settings for the monitored environment."""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from .repository import atomic_write_json
from .validation import CatalogueValidationError


_NAMESPACE = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_SETTINGS_FILE = "monitored_environment.json"
_AUTH_TYPES = {"none", "basic", "bearer"}
_SECRET_KEYS = {"username", "password", "bearer_token", "ca_certificate"}


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


def _optional_url(value: Any, field: str) -> str:
    if value in {None, ""}:
        return ""
    try:
        return validate_prometheus_url(value)
    except CatalogueValidationError as exc:
        raise CatalogueValidationError(f"{field} must be a valid http(s) base URL") from exc


def _auth_type(value: Any) -> str:
    candidate = str(value or "none").strip().lower()
    if candidate not in _AUTH_TYPES:
        raise CatalogueValidationError("Authentication must be none, basic, or bearer")
    return candidate


def _secret_name(config: dict[str, Any]) -> str:
    return str(config.get("PROMETHEUS_CREDENTIALS_SECRET", "cloudsentinel-prometheus-credentials"))


def _read_secret(config: dict[str, Any]) -> dict[str, str]:
    """Read the fixed Kubernetes Secret without ever returning it through the API."""
    if not os.getenv("KUBERNETES_SERVICE_HOST"):
        return {
            "username": os.getenv("PROMETHEUS_BASIC_USERNAME", ""),
            "password": os.getenv("PROMETHEUS_BASIC_PASSWORD", ""),
            "bearer_token": os.getenv("PROMETHEUS_BEARER_TOKEN", ""),
            "ca_certificate": os.getenv("PROMETHEUS_CA_CERTIFICATE", ""),
        }
    try:
        from kubernetes import client, config as kubernetes_config

        kubernetes_config.load_incluster_config()
        secret = client.CoreV1Api().read_namespaced_secret(
            _secret_name(config), os.getenv("POD_NAMESPACE", "cloudsentinel")
        )
        return {
            key: base64.b64decode((secret.data or {}).get(key, "")).decode("utf-8")
            for key in _SECRET_KEYS
        }
    except Exception:
        return {key: "" for key in _SECRET_KEYS}


def _write_secret(config: dict[str, Any], values: dict[str, str]) -> None:
    if not os.getenv("KUBERNETES_SERVICE_HOST"):
        if any(values.values()):
            raise CatalogueValidationError(
                "Credentials can only be saved by a Kubernetes deployment with Secret access"
            )
        return
    try:
        from kubernetes import client, config as kubernetes_config

        kubernetes_config.load_incluster_config()
        encoded = {
            key: base64.b64encode(values.get(key, "").encode("utf-8")).decode("ascii")
            for key in _SECRET_KEYS
        }
        client.CoreV1Api().patch_namespaced_secret(
            _secret_name(config),
            os.getenv("POD_NAMESPACE", "cloudsentinel"),
            {"data": encoded},
        )
    except CatalogueValidationError:
        raise
    except Exception as exc:
        raise CatalogueValidationError("Prometheus credentials could not be saved securely") from exc


def default_environment(config: dict[str, Any]) -> dict[str, Any]:
    sources = config.get("CATALOGUE_PROMETHEUS_SOURCES", {})
    return {
        "schema_version": 2,
        "prometheus_url": validate_prometheus_url(sources.get("cluster-default")),
        "default_namespace": validate_namespace(
            config.get("CATALOGUE_DEFAULT_NAMESPACE", "default")
        ),
        "source_id": "cluster-default",
        "auth_type": "none",
        "tls_verify": True,
        "prometheus_browser_url": "",
        "grafana_browser_url": "",
        "grafana_datasource_uid": "prometheus",
    }


def read_environment(config: dict[str, Any]) -> dict[str, Any]:
    fallback = default_environment(config)
    if not config.get("CATALOGUE_STORAGE_ROOT"):
        return {**fallback, "auth_configured": True, "custom_ca_configured": False}
    path = _settings_path(config)
    if not path.exists():
        return {**fallback, "auth_configured": True, "custom_ca_configured": False}
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        value = {
            **fallback,
            "prometheus_url": validate_prometheus_url(stored.get("prometheus_url")),
            "default_namespace": validate_namespace(stored.get("default_namespace")),
            "auth_type": _auth_type(stored.get("auth_type")),
            "tls_verify": stored.get("tls_verify") is not False,
            "prometheus_browser_url": _optional_url(
                stored.get("prometheus_browser_url"), "Prometheus browser URL"
            ),
            "grafana_browser_url": _optional_url(
                stored.get("grafana_browser_url"), "Grafana browser URL"
            ),
            "grafana_datasource_uid": str(
                stored.get("grafana_datasource_uid") or "prometheus"
            )[:128],
        }
        secret = _read_secret(config)
        value["auth_configured"] = (
            value["auth_type"] == "none"
            or value["auth_type"] == "basic" and bool(secret["username"] and secret["password"])
            or value["auth_type"] == "bearer" and bool(secret["bearer_token"])
        )
        value["custom_ca_configured"] = bool(secret["ca_certificate"])
        return value
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {**fallback, "auth_configured": True, "custom_ca_configured": False}


def write_environment(config: dict[str, Any], payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CatalogueValidationError("A JSON settings object is required")
    current = read_environment(config)
    auth_type = _auth_type(payload.get("auth_type", current.get("auth_type")))
    existing_secret = _read_secret(config)
    clear_credentials = payload.get("clear_credentials") is True
    secret = {key: "" if clear_credentials else existing_secret.get(key, "") for key in _SECRET_KEYS}
    for key in _SECRET_KEYS:
        supplied = payload.get(key)
        if isinstance(supplied, str) and supplied.strip():
            secret[key] = supplied.strip()
    if auth_type == "basic" and not (secret["username"] and secret["password"]):
        raise CatalogueValidationError("Basic authentication requires a username and password")
    if auth_type == "bearer" and not secret["bearer_token"]:
        raise CatalogueValidationError("Bearer authentication requires a token")
    if auth_type == "none":
        secret.update(username="", password="", bearer_token="")
    value = {
        "schema_version": 2,
        "source_id": "cluster-default",
        "prometheus_url": validate_prometheus_url(payload.get("prometheus_url")),
        "default_namespace": validate_namespace(payload.get("default_namespace")),
        "auth_type": auth_type,
        "tls_verify": payload.get("tls_verify") is not False,
        "prometheus_browser_url": _optional_url(
            payload.get("prometheus_browser_url"), "Prometheus browser URL"
        ),
        "grafana_browser_url": _optional_url(
            payload.get("grafana_browser_url"), "Grafana browser URL"
        ),
        "grafana_datasource_uid": str(
            payload.get("grafana_datasource_uid") or "prometheus"
        ).strip()[:128],
    }
    _write_secret(config, secret)
    atomic_write_json(_settings_path(config), value)
    return read_environment(config)


def prometheus_request_options(config: dict[str, Any]) -> dict[str, Any]:
    environment = read_environment(config)
    secret = _read_secret(config)
    options: dict[str, Any] = {}
    if environment["auth_type"] == "basic":
        options["auth"] = (secret["username"], secret["password"])
    elif environment["auth_type"] == "bearer":
        options["headers"] = {"Authorization": f"Bearer {secret['bearer_token']}"}
    if not environment.get("tls_verify", True):
        options["verify"] = False
    elif secret.get("ca_certificate"):
        ca_path = Path(tempfile.gettempdir()) / "cloudsentinel-prometheus-ca.pem"
        ca_path.write_text(secret["ca_certificate"], encoding="utf-8")
        os.chmod(ca_path, 0o600)
        options["verify"] = str(ca_path)
    return options


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
            **prometheus_request_options(config),
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
