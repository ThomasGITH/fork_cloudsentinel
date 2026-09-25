"""Compatibility client for the existing CloudSentinel RCA flow."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

import requests


RCA_TIMEOUT = (5.0, 60.0)


class RCAError(RuntimeError):
    """Raised when the existing RCA flow does not accept a trigger."""


def _data_ingestion_url(settings: dict[str, Any]) -> str:
    value = settings.get("API_DATA_INGESTION_URL")
    if not isinstance(value, str) or not value.strip():
        raise RCAError("settings.API_DATA_INGESTION_URL is required for RCA")
    value = value.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RCAError("settings.API_DATA_INGESTION_URL must be an HTTP(S) URL")
    return value


def trigger_rca(test_info: dict[str, Any]) -> str:
    data = test_info["data"]
    payload = {
        "settings": test_info["settings"],
        "data": {
            "task_id": test_info["task_id"],
            "start_time": data["start_time"],
            "end_time": data["end_time"],
            "crca_pods": data["crca_pods"],
            "metrics": data["metrics"],
            "step": data["data_interval"],
        },
    }
    try:
        response = requests.post(
            f"{_data_ingestion_url(test_info['settings'])}/anomaly_rca",
            data={"crca_data": json.dumps(payload)},
            timeout=RCA_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RCAError(f"RCA request failed: {exc}") from exc
    if not 200 <= response.status_code < 300:
        body = response.text.strip()[:500] if isinstance(response.text, str) else ""
        raise RCAError(f"RCA returned HTTP {response.status_code}: {body or 'no response body'}")
    try:
        response_data = response.json()
    except (TypeError, ValueError) as exc:
        raise RCAError("RCA returned invalid JSON") from exc
    if not isinstance(response_data, dict):
        raise RCAError("RCA response must be a JSON object")
    task_id = response_data.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise RCAError("RCA response did not contain a valid task_id")
    return task_id.strip()
