from flask import Flask, request, jsonify
import logging
import requests
import json
import traceback
import time
import uuid
import os
import re
import csv
from io import StringIO
from datetime import datetime, timezone
from celery import Celery
from kubernetes import client, config
from flask_cors import CORS
import redis

from data_collector import collect_crca_data
from config import set_initial_metric_config, get_config, set_config

try:
    from data_ingestion.catalogue import (
        configure_catalogue_defaults,
        create_catalogue_blueprint,
    )
    from data_ingestion.catalogue.fetching import execute_catalogue_fetch
except ModuleNotFoundError as exc:  # The service image copies catalogue beside app.py.
    if exc.name not in {"data_ingestion", "data_ingestion.catalogue"}:
        raise
    from catalogue import configure_catalogue_defaults, create_catalogue_blueprint
    from catalogue.fetching import execute_catalogue_fetch

try:
    from data_ingestion.live_monitoring import (
        LiveMonitoringError,
        collect_recipe_window,
        decode_session,
        encode_session,
        merge_bounded_buffer,
        safe_session_projection,
        stopped_session,
        validate_resolved_live_model,
    )
except ModuleNotFoundError:
    from live_monitoring import (
        LiveMonitoringError,
        collect_recipe_window,
        decode_session,
        encode_session,
        merge_bounded_buffer,
        safe_session_projection,
        stopped_session,
        validate_resolved_live_model,
    )

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
configure_catalogue_defaults(app)
app.register_blueprint(create_catalogue_blueprint())

# Configure and initialize Celery
app.config['broker_url'] = os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0')
app.config['result_backend'] = os.getenv(
    'CELERY_RESULT_BACKEND', 'redis://redis:6379/0'
)

# testing purposes
# app.config['broker_url'] = 'redis://localhost:6379/0'
# app.config['result_backend'] = 'redis://localhost:6379/0'

celery = Celery(
    app.name,
    broker=app.config['broker_url'],
    backend=app.config['result_backend'],
)
celery.conf.update(
    app.config,
    broker_connection_retry_on_startup=True,
    worker_cancel_long_running_tasks_on_connection_loss=True,
    task_acks_late=True,  # If you are using late acknowledgments
    worker_prefetch_multiplier=1,  # Example configuration to avoid over-fetching
)


@celery.task(bind=True, name="fetch_catalogue_dataset_task")
def fetch_catalogue_dataset_task(self, dataset_id, version, attempt_id):
    """Fetch one immutable catalogue version; this task never schedules itself."""

    def report_progress(phase, completed, total):
        self.update_state(
            state=phase.upper(),
            meta={"phase": phase, "completed": completed, "total": total},
        )

    return execute_catalogue_fetch(
        dataset_id,
        version,
        attempt_id,
        dict(app.config),
        progress_callback=report_progress,
    )


def _dispatch_catalogue_fetch(dataset_id, version, attempt_id, task_id):
    return fetch_catalogue_dataset_task.apply_async(
        args=[dataset_id, version, attempt_id], task_id=task_id
    )


app.extensions["catalogue_fetch_dispatch"] = _dispatch_catalogue_fetch

# Configure Redis
redis_client = redis.StrictRedis.from_url(app.config['broker_url'])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@app.route('/healthz', methods=['GET'])
def health_check():
    """Report process health without touching external dependencies."""
    return jsonify({"status": "healthy"}), 200

# Command to run the Celery worker:
# celery -A app.celery worker --loglevel=info


SESSION_PREFIX = "live_monitoring_session:"


def _session_key(session_id):
    return f"{SESSION_PREFIX}{session_id}"


def _stop_key(session_id):
    return f"{SESSION_PREFIX}stopped:{session_id}"


def _read_session(session_id):
    return decode_session(redis_client.get(_session_key(session_id)))


def _write_session(session, *, force=False):
    if not force and redis_client.get(_stop_key(session["session_id"])):
        return False
    session["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    redis_client.setex(
        _session_key(session["session_id"]),
        int(os.getenv("LIVE_MONITORING_SESSION_TTL_SECONDS", str(30 * 24 * 3600))),
        encode_session(session),
    )
    return True


def _safe_upstream_error(response, fallback):
    try:
        message = response.json().get("error")
    except (ValueError, AttributeError):
        message = None
    return str(message or fallback)[:500]


def _generic_detect(session, collected):
    base_url = os.getenv(
        "API_GENERIC_ANOMALY_DETECTION_URL",
        "http://generic-anomaly-detection-service.cloudsentinel.svc.cluster.local:80",
    ).rstrip("/")
    activate = requests.post(
        f"{base_url}/models/{session['model_id']}/activate", timeout=(5, 60)
    )
    if activate.status_code != 200:
        raise LiveMonitoringError(_safe_upstream_error(activate, "model activation failed"))
    stream = StringIO()
    csv.writer(stream).writerows(collected["rows"])
    metadata = {
        "model_id": session["model_id"],
        "feature_order": collected["feature_order"],
        "feature_order_sha256": collected["feature_order_sha256"],
        "timestamps": collected["timestamps"],
        "context": {
            "live_monitoring_session_id": session["session_id"],
            "recipe_sha256": session["recipe_sha256"],
        },
    }
    response = requests.post(
        f"{base_url}/detect",
        files={"matrix": ("matrix.csv", stream.getvalue(), "text/csv")},
        data={"metadata": json.dumps(metadata)},
        timeout=(5, 120),
    )
    if response.status_code != 200:
        raise LiveMonitoringError(_safe_upstream_error(response, "generic detection failed"))
    payload = response.json()
    return {
        key: payload.get(key)
        for key in (
            "status",
            "model_id",
            "input_observation_count",
            "warmup_observations",
            "prediction_count",
            "anomaly_count",
            "anomaly_percentage",
            "result_reference",
        )
        if key in payload
    }


@celery.task(bind=True)
def monitoring_task(self, session_id):
    """Run one recipe-pinned collection cycle and schedule the next cycle."""
    session = _read_session(session_id)
    if not session or session.get("status") == "stopped":
        return
    try:
        session["status"] = "collecting"
        session["current_task_id"] = self.request.id
        if not _write_session(session):
            return
        end_time = int(time.time())
        start_time = end_time - int(session["window_seconds"])
        collected = collect_recipe_window(
            session["recipe"], start_time, end_time, dict(app.config)
        )
        maximum = int(os.getenv("LIVE_MONITORING_MAX_BUFFER_OBSERVATIONS", "5000"))
        session["buffer"] = merge_bounded_buffer(
            session.get("buffer", []),
            collected["timestamps"],
            collected["rows"],
            maximum,
        )
        bounded = {
            **collected,
            "timestamps": [item["timestamp"] for item in session["buffer"]],
            "rows": [item["values"] for item in session["buffer"]],
        }
        result = _generic_detect(session, bounded)
        session.update(
            {
                "status": "warmup" if not result.get("prediction_count") else "active",
                "iteration": int(session.get("iteration", 0)) + 1,
                "buffered_observations": len(session["buffer"]),
                "missing_values": collected["missing_values"],
                "warnings": collected["warnings"][:20],
                "warmup_observations": result.get("warmup_observations", 0),
                "latest_result": result,
                "error": None,
            }
        )
        if not _write_session(session):
            return
        next_task = monitoring_task.apply_async(
            args=[session_id], countdown=int(session["poll_interval_seconds"])
        )
        session["next_task_id"] = next_task.id
        if not _write_session(session):
            celery.control.revoke(next_task.id, terminate=True)
    except Exception as exc:
        session["status"] = "error"
        session["error"] = str(exc)[:500] if isinstance(exc, LiveMonitoringError) else "live monitoring failed"
        _write_session(session)
        logger.error("Live monitoring session %s failed", session_id)


@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """
    Starts a new monitoring task.

    Returns:
        Response: JSON response with the status and task ID.
    """
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "a JSON request is required"}), 400
        model_id = payload.get("model_id")
        if (
            not isinstance(model_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", model_id) is None
        ):
            return jsonify({"error": "a valid model_id is required"}), 400
        window_seconds = int(payload.get("window_seconds", 600))
        poll_seconds = int(payload.get("poll_interval_seconds", 300))
        if not 10 <= window_seconds <= 86400 or not 5 <= poll_seconds <= 3600:
            return jsonify({"error": "monitoring timing is outside safe limits"}), 400
        learning_url = os.getenv(
            "API_LEARNING_ADAPTATION_URL",
            "http://learning-adaptation-service.cloudsentinel.svc.cluster.local:80",
        ).rstrip("/")
        response = requests.get(
            f"{learning_url}/internal/models/{model_id}/live-input-recipe",
            timeout=(5, 30),
        )
        if response.status_code != 200:
            return jsonify({"error": _safe_upstream_error(response, "model is not ready for live monitoring")}), 409
        try:
            resolved = validate_resolved_live_model(response.json(), model_id)
        except (ValueError, LiveMonitoringError) as exc:
            return jsonify({"error": str(exc)[:500]}), 409
        recipe = resolved["recipe"]
        session_id = f"monitor_{uuid.uuid4().hex}"
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        session = {
            "session_id": session_id,
            "model_id": model_id,
            "status": "queued",
            "created_at": now,
            "started_at": now,
            "artifact_manifest_sha256": resolved["artifact_manifest_sha256"],
            "recipe_sha256": recipe["recipe_sha256"],
            "feature_order_sha256": recipe["feature_order_sha256"],
            "sampling_interval_seconds": recipe["sampling_interval_seconds"],
            "window_seconds": window_seconds,
            "poll_interval_seconds": poll_seconds,
            "iteration": 0,
            "buffer": [],
            "buffered_observations": 0,
            "recipe": recipe,
        }
        _write_session(session)
        task = monitoring_task.apply_async(args=[session_id])
        session["current_task_id"] = task.id
        _write_session(session)
        return jsonify({"status": "monitoring_started", "session": safe_session_projection(session)}), 202
    except Exception as e:
        logger.error(f"Error starting monitoring task: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': 'live monitoring could not be started'}), 500


@app.route('/get_active_tasks', methods=['GET'])
def get_tasks():
    """
    Retrieves the list of scheduled tasks with their IDs.

    Returns:
        Response: JSON response containing the scheduled tasks with their IDs.
    """
    try:
        sessions = []
        for key in redis_client.scan_iter(f"{SESSION_PREFIX}*"):
            value = decode_session(redis_client.get(key))
            if value:
                sessions.append(safe_session_projection(value))
        sessions.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return jsonify(sessions[:100]), 200
    except Exception as e:
        logger.error(f"Error retrieving scheduled tasks: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/monitoring-sessions/<session_id>', methods=['GET'])
def monitoring_session_status(session_id):
    session = _read_session(session_id)
    if not session:
        return jsonify({"error": "unknown monitoring session"}), 404
    return jsonify(safe_session_projection(session)), 200


@app.route('/stop_monitoring/<session_id>', methods=['DELETE'])
def stop_monitoring(session_id):
    """
    Stops a running monitoring task and its scheduled instances.

    Args:
        task_id (str): The ID of the task to be stopped.

    Returns:
        Response: JSON response with the status of the operation.
    """
    try:
        session = _read_session(session_id)
        if not session:
            return jsonify({"error": "unknown monitoring session"}), 404
        redis_client.setex(
            _stop_key(session_id),
            int(os.getenv("LIVE_MONITORING_SESSION_TTL_SECONDS", str(30 * 24 * 3600))),
            "1",
        )
        session = stopped_session(
            session,
            lambda task_id: celery.control.revoke(task_id, terminate=True),
            stopped_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        _write_session(session, force=True)
        return jsonify({"status": "monitoring_stopped", "session": safe_session_projection(session)}), 200
    except Exception as e:
        logger.error(f"Error stopping session {session_id}: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': 'monitoring could not be stopped'}), 500



@app.route('/anomaly_rca', methods=['POST'])
def anomaly_rca():
    """
    Endpoint for anomaly root cause analysis (RCA).

    Returns:
        Response: JSON response with the RCA result.
    """
    try:
        data = request.form.get('crca_data')
        data = json.loads(data)
        response = collect_crca_data(data)
        logger.info("Anomaly RCA completed")
        return response
    except Exception as e:
        logger.error(f"Error performing anomaly RCA: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/get_pod_names', methods=['POST'])
def get_pod_names():
    """
    List the names of the pods in the specified namespace.

    Returns:
        Response: JSON response with the list of pod names.
    """
    try:
        namespace = request.json['namespace']
        v1 = client.CoreV1Api()
        ret = v1.list_namespaced_pod(namespace, watch=False)
        # ret = v1.list_pod_for_all_namespaces(watch=False)
        pod_names = [item.metadata.name for item in ret.items]
        logger.info(f"Retrieved pod names for namespace {namespace}")
        return jsonify(pod_names), 200
    except Exception as e:
        logger.error(f"Error retrieving pod names: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/load_kube_config', methods=['POST'])
def load_config():
    """
    Loads Kubernetes configuration.

    Returns:
        Response: JSON response with the status of the operation.
    """
    try:
        config.load_incluster_config()
        logger.info("Kubernetes configuration loaded successfully")
        return jsonify({"message": "Kubernetes configuration loaded successfully."}), 200
    except FileNotFoundError as e:
        logger.error(f"Kubernetes configuration file not found: {traceback.format_exc()}")
        return jsonify({"error": f"Kubernetes configuration file not found: {str(e)}"}), 400
    except config.ConfigException as e:
        logger.error(f"Error loading Kubernetes configuration: {traceback.format_exc()}")
        return jsonify({"error": f"Invalid Kubernetes configuration: {str(e)}"}), 400
    except ConnectionError as e:
        logger.error(f"Could not connect to Kubernetes cluster: {traceback.format_exc()}")
        return jsonify({"error": f"Could not connect to Kubernetes cluster: {str(e)}"}), 503
    except Exception as e:
        logger.error(f"Unexpected error: {traceback.format_exc()}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


@app.route('/get_metrics', methods=['GET'])
def get_config_route():
    """
    Retrieves the current metrics configuration.

    Returns:
        Response: JSON response with the metrics configuration.
    """
    try:
        config = get_config()
        logger.info("Metrics configuration retrieved")
        return jsonify(config), 200
    except Exception as e:
        logger.error(f"Error retrieving metrics configuration: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/update_config', methods=['POST'])
def update_config():
    """
    Updates the configuration for the monitoring application.

    Returns:
        Response: JSON response with the status of the operation.
    """
    new_config = request.json
    try:
        set_config(new_config)
        logger.info("Configuration updated successfully")
        return jsonify("success"), 200
    except Exception as e:
        logger.error(f"Error updating configuration: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    set_initial_metric_config()
    logger.info("Starting Flask app")
    app.run(debug=False, host='0.0.0.0', port=5001)
