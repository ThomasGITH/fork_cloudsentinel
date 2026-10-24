"""Bounded Kubernetes workload experiments for research dataset capture."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hmac
import json
import os
from pathlib import Path
import tempfile
import threading
import uuid
from typing import Any

from flask import Blueprint, current_app, jsonify, request
try:
    from kubernetes import client, config
    from kubernetes.client.exceptions import ApiException
except ModuleNotFoundError:  # Lightweight unit-test environments do not install the cluster client.
    client = None
    config = None

    class ApiException(Exception):
        pass


PROFILES = {
    "elevated": {
        "label": "Elevated traffic",
        "description": "A moderate request increase suitable for observing latency, network and CPU changes.",
        "users": "40",
        "rate": "5",
        "workload_intensity": "Medium",
    },
    "burst": {
        "label": "Traffic burst",
        "description": "A short, bounded high-load interval intended to create a visible context change.",
        "users": "80",
        "rate": "10",
        "workload_intensity": "High",
    },
}
_LOCK = threading.RLock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    return (value or _now()).isoformat().replace("+00:00", "Z")


def _authorized() -> bool:
    expected = current_app.config.get("EXPERIMENT_OPERATOR_KEY", "")
    provided = request.headers.get("X-Experiment-Operator-Key", "")
    return bool(expected) and hmac.compare_digest(expected, provided)


def _storage_path() -> Path:
    return Path(current_app.config["WORKLOAD_EXPERIMENT_STORAGE_PATH"])


def _read_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        value = {}
    return value if isinstance(value, dict) else {}


def _write_state(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".workload-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _deployment_environment(configuration: dict[str, Any]) -> dict[str, str]:
    if client is None or config is None:
        raise RuntimeError("Kubernetes client is unavailable")
    config.load_incluster_config()
    deployment = client.AppsV1Api().read_namespaced_deployment(
        configuration["deployment"], configuration["namespace"]
    )
    for container in deployment.spec.template.spec.containers:
        if container.name == configuration["container"]:
            return {item.name: item.value for item in container.env or [] if item.value is not None}
    raise RuntimeError("Configured load-generator container was not found")


def _patch_load(configuration: dict[str, Any], values: dict[str, str]) -> None:
    if client is None or config is None:
        raise RuntimeError("Kubernetes client is unavailable")
    config.load_incluster_config()
    body = {
        "spec": {
            "template": {
                "metadata": {"annotations": {"cloudsentinel.io/experiment-update": _timestamp()}},
                "spec": {
                    "containers": [
                        {
                            "name": configuration["container"],
                            "env": [
                                {"name": configuration["users_variable"], "value": values["users"]},
                                {"name": configuration["rate_variable"], "value": values["rate"]},
                            ],
                        }
                    ]
                },
            }
        }
    }
    client.AppsV1Api().patch_namespaced_deployment(
        configuration["deployment"], configuration["namespace"], body
    )


def _public(state: dict[str, Any], configuration: dict[str, Any]) -> dict[str, Any]:
    return {
        "profiles": [
            {"id": profile_id, **deepcopy(profile)}
            for profile_id, profile in PROFILES.items()
        ],
        "target": {
            "namespace": configuration["namespace"],
            "deployment": configuration["deployment"],
            "container": configuration["container"],
        },
        "active": deepcopy(state.get("active")),
        "history": deepcopy(state.get("history", []))[-20:][::-1],
        "safety": {
            "minimum_duration_minutes": 2,
            "maximum_duration_minutes": 15,
            "automatic_restore": True,
        },
    }


def _configuration(app_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "namespace": app_config["WORKLOAD_EXPERIMENT_NAMESPACE"],
        "deployment": app_config["WORKLOAD_EXPERIMENT_DEPLOYMENT"],
        "container": app_config["WORKLOAD_EXPERIMENT_CONTAINER"],
        "users_variable": app_config["WORKLOAD_EXPERIMENT_USERS_VARIABLE"],
        "rate_variable": app_config["WORKLOAD_EXPERIMENT_RATE_VARIABLE"],
    }


def _restore(
    path: Path,
    configuration: dict[str, Any],
    reason: str,
    expected_experiment_id: str | None = None,
) -> None:
    with _LOCK:
        state = _read_state(path)
        active = state.get("active")
        if not active:
            return
        if expected_experiment_id and active.get("experiment_id") != expected_experiment_id:
            return
        _patch_load(configuration, active["baseline"])
        active["status"] = "completed"
        active["completed_at"] = _timestamp()
        active["completion_reason"] = reason
        state.setdefault("history", []).append(active)
        state["active"] = None
        _write_state(path, state)


def _schedule_restore(
    path: Path,
    configuration: dict[str, Any],
    seconds: int,
    experiment_id: str,
) -> None:
    timer = threading.Timer(
        seconds,
        _restore,
        args=(path, configuration, "automatic timeout", experiment_id),
    )
    timer.daemon = True
    timer.start()


def create_workload_experiments_blueprint() -> Blueprint:
    blueprint = Blueprint("workload_experiments", __name__, url_prefix="/experiments/workload")

    @blueprint.route("", methods=["GET", "POST"])
    def workload_experiments():
        if not current_app.config.get("EXPERIMENT_OPERATOR_KEY"):
            return jsonify({"error": "Workload experiments are not configured"}), 503
        if not _authorized():
            return jsonify({"error": "Workload experiment authorization failed"}), 403
        path = _storage_path()
        configuration = _configuration(current_app.config)
        try:
            with _LOCK:
                state = _read_state(path)
                active = state.get("active")
                if active and datetime.fromisoformat(active["planned_end_at"].replace("Z", "+00:00")) <= _now():
                    _restore(path, configuration, "automatic timeout")
                    state = _read_state(path)
                if request.method == "GET":
                    return jsonify(_public(state, configuration))

                payload = request.get_json(silent=True) or {}
                action = payload.get("action", "start")
                if action == "stop":
                    if not state.get("active"):
                        return jsonify({"error": "No workload experiment is active"}), 409
                    _restore(path, configuration, "stopped by operator")
                    return jsonify(_public(_read_state(path), configuration))
                if action != "start":
                    return jsonify({"error": "Unsupported experiment action"}), 400
                if state.get("active"):
                    return jsonify({"error": "A workload experiment is already active"}), 409
                profile_id = payload.get("profile")
                if profile_id not in PROFILES:
                    return jsonify({"error": "Select a supported workload profile"}), 400
                try:
                    duration = int(payload.get("duration_minutes", 5))
                except (TypeError, ValueError):
                    return jsonify({"error": "Duration must be a whole number of minutes"}), 400
                if duration < 2 or duration > 15:
                    return jsonify({"error": "Duration must be between 2 and 15 minutes"}), 400

                environment = _deployment_environment(configuration)
                baseline = {
                    "users": environment.get(configuration["users_variable"], "10"),
                    "rate": environment.get(configuration["rate_variable"], "1"),
                }
                profile = PROFILES[profile_id]
                applied = {"users": profile["users"], "rate": profile["rate"]}
                started = _now()
                _patch_load(configuration, applied)
                active = {
                    "experiment_id": f"workload-{uuid.uuid4().hex}",
                    "profile": profile_id,
                    "profile_label": profile["label"],
                    "status": "active",
                    "started_at": _timestamp(started),
                    "planned_end_at": _timestamp(started + timedelta(minutes=duration)),
                    "completed_at": None,
                    "duration_minutes": duration,
                    "baseline": baseline,
                    "applied": applied,
                    "workload_context": {
                        "workload_intensity": profile["workload_intensity"],
                        "dominant_workload_characteristic": "High traffic",
                        "anomaly_scenario": "Other",
                        "anomaly_scenario_detail": profile["label"],
                    },
                }
                state["active"] = active
                state.setdefault("history", [])
                _write_state(path, state)
                _schedule_restore(
                    path, configuration, duration * 60, active["experiment_id"]
                )
                return jsonify(_public(state, configuration)), 202
        except (ApiException, OSError, RuntimeError, ValueError, KeyError):
            return jsonify({"error": "The controlled workload could not be updated safely"}), 503

    return blueprint


def configure_workload_experiment_defaults(app: Any) -> None:
    app.config.setdefault("EXPERIMENT_OPERATOR_KEY", os.getenv("EXPERIMENT_OPERATOR_KEY", ""))
    app.config.setdefault(
        "WORKLOAD_EXPERIMENT_STORAGE_PATH",
        os.getenv("WORKLOAD_EXPERIMENT_STORAGE_PATH", "/app/storage/workload_experiments.json"),
    )
    app.config.setdefault("WORKLOAD_EXPERIMENT_NAMESPACE", os.getenv("WORKLOAD_EXPERIMENT_NAMESPACE", "default"))
    app.config.setdefault("WORKLOAD_EXPERIMENT_DEPLOYMENT", os.getenv("WORKLOAD_EXPERIMENT_DEPLOYMENT", "loadgenerator"))
    app.config.setdefault("WORKLOAD_EXPERIMENT_CONTAINER", os.getenv("WORKLOAD_EXPERIMENT_CONTAINER", "main"))
    app.config.setdefault("WORKLOAD_EXPERIMENT_USERS_VARIABLE", os.getenv("WORKLOAD_EXPERIMENT_USERS_VARIABLE", "USERS"))
    app.config.setdefault("WORKLOAD_EXPERIMENT_RATE_VARIABLE", os.getenv("WORKLOAD_EXPERIMENT_RATE_VARIABLE", "RATE"))
    path = Path(app.config["WORKLOAD_EXPERIMENT_STORAGE_PATH"])
    state = _read_state(path)
    active = state.get("active")
    if isinstance(active, dict) and active.get("experiment_id") and active.get("planned_end_at"):
        try:
            end = datetime.fromisoformat(active["planned_end_at"].replace("Z", "+00:00"))
            seconds = max(0, int((end - _now()).total_seconds()))
            _schedule_restore(
                path,
                _configuration(app.config),
                seconds,
                active["experiment_id"],
            )
        except (TypeError, ValueError):
            pass
