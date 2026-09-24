"""HTTP API for detector manifest discovery."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify

from .registry import ManifestValidationError, discover_detectors


def create_detectors_blueprint(detector_dir: str | Path | None = None) -> Blueprint:
    blueprint = Blueprint("detectors", __name__)

    @blueprint.get("/detectors")
    def list_detectors():
        try:
            manifests = discover_detectors(detector_dir)
        except ManifestValidationError as exc:
            return jsonify({"error": "Detector manifest discovery failed", "details": str(exc)}), 500
        return jsonify({"detectors": manifests})

    return blueprint
