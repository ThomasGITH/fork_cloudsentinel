"""HTTP API for detector manifest discovery."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify

from .registry import scan_detectors


def create_detectors_blueprint(detector_dir: str | Path | None = None) -> Blueprint:
    blueprint = Blueprint("detectors", __name__)

    @blueprint.get("/detectors")
    def list_detectors():
        return jsonify(scan_detectors(detector_dir))

    return blueprint
