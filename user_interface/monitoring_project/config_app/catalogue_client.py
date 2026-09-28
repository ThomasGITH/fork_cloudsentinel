"""Small server-side client for the data-ingestion Data Catalogue API."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

import requests
from django.conf import settings


_URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)


class CatalogueClientError(RuntimeError):
    """Safe error raised for catalogue transport and API failures."""

    def __init__(self, message: str, *, status_code: int = 502, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def _safe_message(value: Any, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    return _URL_PATTERN.sub("[internal service]", value.strip())[:500]


class DataCatalogueClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        transport: Any = requests,
        timeout: tuple[float, float] = (3.05, 20.0),
    ):
        self.base_url = (base_url or settings.API_DATA_INGESTION_URL).rstrip("/")
        self.transport = transport
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self.transport.request(
                method,
                f"{self.base_url}{path}",
                params=params,
                json=json,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise CatalogueClientError(
                "Data Catalogue request timed out. Try again.", retryable=True
            ) from exc
        except requests.RequestException as exc:
            raise CatalogueClientError(
                "Data Catalogue service is temporarily unavailable. Try again.",
                retryable=True,
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            if response.status_code >= 400:
                raise CatalogueClientError(
                    "Data Catalogue returned an invalid error response.",
                    status_code=response.status_code,
                    retryable=response.status_code >= 500,
                ) from exc
            raise CatalogueClientError(
                "Data Catalogue returned invalid JSON.", retryable=True
            ) from exc

        if response.status_code >= 400:
            fallback = f"Data Catalogue request failed with HTTP {response.status_code}."
            message = _safe_message(
                payload.get("error") if isinstance(payload, dict) else None,
                fallback,
            )
            raise CatalogueClientError(
                message,
                status_code=response.status_code,
                retryable=response.status_code >= 500,
            )
        if not isinstance(payload, dict):
            raise CatalogueClientError("Data Catalogue returned an invalid response.")
        return payload

    @staticmethod
    def _dataset(dataset_id: str) -> str:
        return quote(dataset_id, safe="")

    def list_datasets(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._request("GET", "/datasets", params=params)

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self._request("GET", f"/datasets/{self._dataset(dataset_id)}")

    def get_version(self, dataset_id: str, version: int) -> dict[str, Any]:
        return self._request(
            "GET", f"/datasets/{self._dataset(dataset_id)}/versions/{version}"
        )

    def create_dataset(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/datasets", json=payload)

    def update_metadata(self, dataset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "PATCH", f"/datasets/{self._dataset(dataset_id)}", json=payload
        )

    def create_version(self, dataset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST", f"/datasets/{self._dataset(dataset_id)}/versions", json=payload
        )

    def start_fetch(self, dataset_id: str, version: int) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/datasets/{self._dataset(dataset_id)}/versions/{version}/fetch",
        )

    def fetch_status(self, dataset_id: str, version: int) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/datasets/{self._dataset(dataset_id)}/fetch-status",
            params={"version": version},
        )

    def preview(self, dataset_id: str, version: int, limit: int = 20) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/datasets/{self._dataset(dataset_id)}/preview",
            params={"version": version, "limit": limit},
        )

    def add_incident(
        self, dataset_id: str, version: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/datasets/{self._dataset(dataset_id)}/versions/{version}/incidents",
            json=payload,
        )

    def derive_labels(self, dataset_id: str, version: int) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/datasets/{self._dataset(dataset_id)}/versions/{version}/labels",
            json={"source": "incident_windows"},
        )

    def create_partition(
        self, dataset_id: str, version: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/datasets/{self._dataset(dataset_id)}/versions/{version}/partitions",
            json=payload,
        )


def get_catalogue_client() -> DataCatalogueClient:
    return DataCatalogueClient()
