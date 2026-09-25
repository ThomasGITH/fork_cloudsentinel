"""Binary promotion of trained Isolation Forest artifacts."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
from typing import Any

import requests


DETECTOR_ID = "isolation-forest"
DEFAULT_DETECTION_SERVICE_URL = "http://127.0.0.1:5014"
DETECTION_SERVICE_URL_ENV = "API_ISOLATION_FOREST_ANOMALY_DETECTION_URL"
CONNECT_TIMEOUT_SECONDS = 5.0
READ_TIMEOUT_SECONDS = 60.0
PROMOTION_MAX_ATTEMPTS = 2
ARTIFACT_FILES = {
    "model": "model.joblib",
    "metadata": "model_metadata.json",
    "evaluation": "model_evaluation.json",
}


class IsolationForestPromotionError(RuntimeError):
    """Raised when trained artifacts cannot be promoted safely."""


def detection_service_url() -> str:
    """Return the configured detection-service base URL."""
    value = os.getenv(DETECTION_SERVICE_URL_ENV, DEFAULT_DETECTION_SERVICE_URL).strip()
    if not value:
        raise IsolationForestPromotionError(
            f"{DETECTION_SERVICE_URL_ENV} must not be empty"
        )
    return value.rstrip("/")


def file_sha256(path: Path) -> str:
    """Calculate a file digest without loading the complete artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_paths(artifact_dir: str | os.PathLike[str]) -> dict[str, Path]:
    directory = Path(artifact_dir)
    paths = {
        field: directory / filename
        for field, filename in ARTIFACT_FILES.items()
    }
    for field, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise IsolationForestPromotionError(
                f"cannot promote Isolation Forest model: missing {ARTIFACT_FILES[field]}"
            )
    return paths


def _response_error_text(response: Any) -> str:
    text = getattr(response, "text", "")
    if not isinstance(text, str) or not text.strip():
        return "no response body"
    return text.strip()[:500]


def promote_isolation_forest_model(
    artifact_dir: str | os.PathLike[str],
    model_id: str,
    *,
    service_url: str | None = None,
    timeout: tuple[float, float] = (
        CONNECT_TIMEOUT_SECONDS,
        READ_TIMEOUT_SECONDS,
    ),
    max_attempts: int = PROMOTION_MAX_ATTEMPTS,
) -> dict[str, str]:
    """Upload one complete model artifact set and validate remote availability."""
    if (
        not isinstance(max_attempts, int)
        or isinstance(max_attempts, bool)
        or max_attempts < 1
    ):
        raise ValueError("max_attempts must be a positive integer")
    try:
        paths = _artifact_paths(artifact_dir)
        checksums = {field: file_sha256(path) for field, path in paths.items()}
    except OSError as exc:
        raise IsolationForestPromotionError(
            f"could not read Isolation Forest promotion artifacts: {exc}"
        ) from exc
    url = f"{(service_url or detection_service_url()).rstrip('/')}/save_model"
    form = {
        "detector_id": DETECTOR_ID,
        "model_id": model_id,
        "model_sha256": checksums["model"],
        "metadata_sha256": checksums["metadata"],
        "evaluation_sha256": checksums["evaluation"],
    }

    last_error: IsolationForestPromotionError | None = None
    for _attempt in range(1, max_attempts + 1):
        try:
            with ExitStack() as stack:
                files = {
                    field: (
                        path.name,
                        stack.enter_context(path.open("rb")),
                        "application/octet-stream"
                        if field == "model"
                        else "application/json",
                    )
                    for field, path in paths.items()
                }
                response = requests.post(
                    url,
                    data=form,
                    files=files,
                    timeout=timeout,
                )
            if not 200 <= response.status_code < 300:
                raise IsolationForestPromotionError(
                    "Isolation Forest model promotion returned "
                    f"HTTP {response.status_code}: {_response_error_text(response)}"
                )
            try:
                payload = response.json()
            except (TypeError, ValueError) as exc:
                raise IsolationForestPromotionError(
                    "Isolation Forest model promotion returned invalid JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise IsolationForestPromotionError(
                    "Isolation Forest model promotion response must be a JSON object"
                )
            if payload.get("status") != "available":
                raise IsolationForestPromotionError(
                    "Isolation Forest model promotion response did not confirm availability"
                )
            if payload.get("detector_id") != DETECTOR_ID:
                raise IsolationForestPromotionError(
                    "Isolation Forest model promotion response has a detector_id mismatch"
                )
            if payload.get("model_id") != model_id:
                raise IsolationForestPromotionError(
                    "Isolation Forest model promotion response has a model_id mismatch"
                )
            return {
                "status": "available",
                "detector_id": DETECTOR_ID,
                "model_id": model_id,
            }
        except requests.RequestException as exc:
            last_error = IsolationForestPromotionError(
                f"Isolation Forest model promotion request failed: {exc}"
            )
            last_error.__cause__ = exc
        except OSError as exc:
            last_error = IsolationForestPromotionError(
                f"could not read Isolation Forest promotion artifacts: {exc}"
            )
            last_error.__cause__ = exc
        except IsolationForestPromotionError as exc:
            last_error = exc

    raise IsolationForestPromotionError(
        f"Isolation Forest model promotion failed after {max_attempts} attempts: "
        f"{last_error}"
    ) from last_error
