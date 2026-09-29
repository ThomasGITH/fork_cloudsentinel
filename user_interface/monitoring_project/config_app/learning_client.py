"""Small server-side client for the learning-adaptation public API."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote

import requests
from django.conf import settings


_URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9])/(?:app|tmp|var|home|Users)/[^\s]*")
_MAX_JSON_BYTES = 2_000_000


class LearningClientError(RuntimeError):
    """Safe error raised for learning-adaptation transport and API failures."""

    def __init__(self, message: str, *, status_code: int = 502, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def safe_learning_message(value: Any, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    message = _URL_PATTERN.sub("[internal service]", value.strip())
    message = _PATH_PATTERN.sub("[internal path]", message)
    return " ".join(message.split())[:500]


class LearningAdaptationClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        transport: Any = requests,
        connect_timeout: float = 3.0,
        read_timeout: float = 20.0,
    ):
        self.base_url = (base_url or settings.API_LEARNING_ADAPTATION_URL).rstrip("/")
        self.transport = transport
        self.timeout = (connect_timeout, read_timeout)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.transport.request(
                method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs
            )
        except requests.RequestException as exc:
            raise LearningClientError(
                "Learning service is temporarily unavailable.", retryable=True
            ) from exc

        headers = getattr(response, "headers", None)
        content_length = headers.get("Content-Length") if isinstance(headers, dict) else None
        if content_length:
            try:
                if int(content_length) > _MAX_JSON_BYTES:
                    raise LearningClientError("Learning service response is too large.")
            except ValueError:
                pass
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise LearningClientError(
                "Learning service returned invalid JSON.",
                status_code=response.status_code,
                retryable=response.status_code >= 500,
            ) from exc
        try:
            if len(json.dumps(payload, separators=(",", ":")).encode("utf-8")) > _MAX_JSON_BYTES:
                raise LearningClientError("Learning service response is too large.")
        except (TypeError, ValueError) as exc:
            raise LearningClientError("Learning service returned an invalid response.") from exc

        if response.status_code >= 400:
            fallback = f"Learning service request failed with HTTP {response.status_code}."
            message = safe_learning_message(
                payload.get("error") if isinstance(payload, dict) else None, fallback
            )
            raise LearningClientError(
                message,
                status_code=response.status_code,
                retryable=response.status_code >= 500,
            )
        if not isinstance(payload, dict):
            raise LearningClientError("Learning service returned an invalid response.")
        return payload

    @staticmethod
    def _identifier(value: str) -> str:
        return quote(value, safe="")

    def list_detectors(self) -> dict[str, Any]:
        return self._request("GET", "/detectors")

    def get_detector(self, detector_id: str) -> dict[str, Any]:
        # The backend intentionally has no separate detector-detail route.
        result = self.list_detectors()
        for detector in result.get("detectors", []):
            if detector.get("id") == detector_id:
                return detector
        raise LearningClientError("Detector was not found.", status_code=404)

    def list_training_runs(self, filters: dict[str, Any]) -> dict[str, Any]:
        return self._request("GET", "/training_runs", params=filters)

    def get_training_run(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/training_runs/{self._identifier(run_id)}")

    def create_training_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/training_runs", json=payload)

    def list_models(self, filters: dict[str, Any]) -> dict[str, Any]:
        return self._request("GET", "/models", params=filters)

    def get_model(self, model_id: str) -> dict[str, Any]:
        return self._request("GET", f"/models/{self._identifier(model_id)}")


def get_learning_client() -> LearningAdaptationClient:
    return LearningAdaptationClient()
