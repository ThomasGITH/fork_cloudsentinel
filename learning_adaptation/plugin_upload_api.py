"""Operator-gated HTTP publication for trusted detector plugin packages."""

from __future__ import annotations

import hmac
import os
from pathlib import Path
import tempfile
from typing import Any

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from detectors.plugin_integrity import PluginPackageError
from detectors.plugin_repository import PluginRepository, PluginRepositoryError
from detectors.registry import ManifestValidationError


MAX_PLUGIN_BYTES = 2 * 1024 * 1024
RESERVED_FILES = {"manifest.yaml", "adapter.py", "requirements.lock"}


def _authorized() -> bool:
    expected = os.getenv("PLUGIN_OPERATOR_KEY", "")
    provided = request.headers.get("X-Plugin-Operator-Key", "")
    return bool(expected) and hmac.compare_digest(expected, provided)


def _safe_error(exc: Exception) -> str:
    return " ".join(str(exc).replace("\r", " ").replace("\n", " ").split())[:1000]


def _save_upload(upload: Any, destination: Path, budget: list[int]) -> None:
    data = upload.read(MAX_PLUGIN_BYTES + 1)
    budget[0] += len(data)
    if len(data) > MAX_PLUGIN_BYTES or budget[0] > MAX_PLUGIN_BYTES:
        raise PluginRepositoryError("plugin package exceeds the 2 MiB upload limit")
    if not data:
        raise PluginRepositoryError(f"{destination.name} is empty")
    destination.write_bytes(data)


def create_plugin_upload_blueprint() -> Blueprint:
    blueprint = Blueprint("plugin_upload", __name__)

    @blueprint.post("/detectors/plugins/upload")
    def upload_plugin():
        if not os.getenv("PLUGIN_OPERATOR_KEY"):
            return jsonify({"error": "Browser plugin publication is not configured"}), 503
        if not _authorized():
            return jsonify({"error": "The operator access key is invalid"}), 403
        action = request.form.get("action", "validate")
        if action not in {"validate", "publish"}:
            return jsonify({"error": "Unsupported plugin action"}), 400
        required = {
            "manifest": ("manifest.yaml", request.files.get("manifest")),
            "adapter": ("adapter.py", request.files.get("adapter")),
            "requirements": ("requirements.lock", request.files.get("requirements")),
        }
        if any(upload is None for _name, upload in required.values()):
            return jsonify({"error": "manifest, adapter.py and requirements.lock are required"}), 400
        try:
            with tempfile.TemporaryDirectory(prefix="plugin_upload_") as directory:
                package = Path(directory)
                budget = [0]
                for filename, upload in required.values():
                    _save_upload(upload, package / filename, budget)
                support_names: set[str] = set()
                for upload in request.files.getlist("support_files"):
                    filename = secure_filename(upload.filename or "")
                    if (
                        not filename
                        or filename in RESERVED_FILES
                        or filename in support_names
                        or not filename.endswith(".py")
                    ):
                        raise PluginRepositoryError(
                            "support files must be uniquely named top-level .py files"
                        )
                    support_names.add(filename)
                    _save_upload(upload, package / filename, budget)
                repository = PluginRepository(
                    os.getenv(
                        "DETECTOR_PLUGIN_REPOSITORY_ROOT",
                        "/opt/cloudsentinel/detectors",
                    )
                )
                validation = repository.validate_source(package)
                repository._validate_adapter_contract(package, validation)
                manifest = validation["manifest"]
                result = {
                    "status": "valid",
                    "detector_id": manifest["id"],
                    "detector_version": manifest["version"],
                    "package_sha256": validation["package_sha256"],
                }
                if action == "publish":
                    result = repository.install(package)
                    if request.form.get("activate") == "true":
                        result = repository.activate(
                            result["detector_id"], result["package_sha256"]
                        )
                    result["status"] = "published"
                return jsonify(result), 200
        except (PluginRepositoryError, PluginPackageError, ManifestValidationError) as exc:
            return jsonify({"error": _safe_error(exc)}), 400
        except OSError:
            return jsonify({"error": "The plugin package could not be staged"}), 503

    return blueprint
