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
import hashlib

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
        align_new_predictions,
        merge_buffer_with_watermark,
        merge_execution_projection,
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
        align_new_predictions,
        merge_buffer_with_watermark,
        merge_execution_projection,
        safe_session_projection,
        stopped_session,
        validate_resolved_live_model,
    )

try:
    from data_ingestion.monitoring_history import (
        MonitoringHistoryError,
        MonitoringHistoryStore,
        MonitoringSessionNotFound,
        MonitoringSessionStopped,
        utc_now,
    )
except ModuleNotFoundError:
    from monitoring_history import (
        MonitoringHistoryError,
        MonitoringHistoryStore,
        MonitoringSessionNotFound,
        MonitoringSessionStopped,
        utc_now,
    )

try:
    from data_ingestion.monitoring_rca import (
        MonitoringRcaError,
        assess_recipe_compatibility,
        canonical_sha256 as rca_sha256,
        decide_trigger,
        initial_rca_state,
        mark_triggered,
        select_analysis_input,
        stable_run_id,
        validate_rca_configuration,
    )
except ModuleNotFoundError:
    from monitoring_rca import (
        MonitoringRcaError,
        assess_recipe_compatibility,
        canonical_sha256 as rca_sha256,
        decide_trigger,
        initial_rca_state,
        mark_triggered,
        select_analysis_input,
        stable_run_id,
        validate_rca_configuration,
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


monitoring_history = MonitoringHistoryStore(
    os.getenv("MONITORING_HISTORY_STORAGE_ROOT", "/app/storage/monitoring_history"),
    max_observations=int(os.getenv("LIVE_MONITORING_MAX_BUFFER_OBSERVATIONS", "5000")),
    max_features=int(os.getenv("CATALOGUE_MAX_FEATURES", "1000")),
)


def _session_key(session_id):
    return f"{SESSION_PREFIX}{session_id}"


def _stop_key(session_id):
    return f"{SESSION_PREFIX}stopped:{session_id}"


def _read_session(session_id):
    try:
        return decode_session(redis_client.get(_session_key(session_id)))
    except redis.RedisError:
        return None


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
        f"{base_url}/internal/evaluate",
        files={"matrix": ("matrix.csv", stream.getvalue(), "text/csv")},
        data={"metadata": json.dumps(metadata)},
        timeout=(5, 120),
    )
    if response.status_code != 200:
        raise LiveMonitoringError(_safe_upstream_error(response, "generic detection failed"))
    payload = response.json()
    if payload.get("model_id") != session["model_id"] or payload.get(
        "artifact_manifest_sha256"
    ) != session["artifact_manifest_sha256"]:
        raise LiveMonitoringError("generic detection returned mismatched model provenance")
    return payload


def _cycle_id(session_id, query_start, query_end):
    identity = f"{session_id}|{query_start}|{query_end}".encode("utf-8")
    return f"cycle_{hashlib.sha256(identity).hexdigest()[:32]}"


def _public_latest_result(aligned, model_id):
    return {
        "status": "success",
        "model_id": model_id,
        "warmup_observations": aligned["warmup_observations"],
        "prediction_count": aligned["prediction_count"],
        "anomaly_count": aligned["anomaly_count"],
        "anomaly_percentage": aligned["anomaly_percentage"],
    }


def _crca_url():
    return os.getenv(
        "API_CRCA_ANOMALY_DETECTION_URL",
        "http://crca-anomaly-detection-service.cloudsentinel.svc.cluster.local:80",
    ).rstrip("/")


def _crca_capability():
    try:
        response = requests.get(f"{_crca_url()}/internal/monitoring-capability", timeout=(5, 15))
    except requests.RequestException as exc:
        raise MonitoringRcaError("CausalRCA runtime readiness could not be verified.") from exc
    if response.status_code != 200:
        raise MonitoringRcaError("CausalRCA runtime readiness could not be verified.")
    try:
        payload = response.json()
    except ValueError as exc:
        raise MonitoringRcaError("CausalRCA runtime returned invalid readiness metadata.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("config_sha256"), str):
        raise MonitoringRcaError("CausalRCA runtime returned invalid readiness metadata.")
    return payload


def _dispatch_rca_run(session_id, run_id):
    """Idempotently submit a durable intent. Failure never fails AD monitoring."""
    session = monitoring_history.read_session(session_id)
    if session.get("status") == "stopped":
        return False
    run = monitoring_history.read_rca_run(session_id, run_id)
    if run.get("status") != "queued" or run.get("dispatch_status") == "submitted":
        return False
    try:
        response = requests.post(
            f"{_crca_url()}/internal/monitoring-runs",
            json={"session_id": session_id, "run_id": run_id},
            timeout=(5, 15),
        )
        if response.status_code not in {200, 202}:
            raise MonitoringRcaError(_safe_upstream_error(response, "CausalRCA dispatch failed"))
        monitoring_history.update_rca_run(
            session_id,
            run_id,
            lambda record: record.update(dispatch_status="submitted", dispatch_error=None),
        )
        return True
    except (requests.RequestException, MonitoringRcaError) as exc:
        monitoring_history.update_rca_run(
            session_id,
            run_id,
            lambda record: record.update(
                dispatch_status="pending",
                dispatch_error={"type": "rca_dispatch_error", "message": "CausalRCA dispatch is temporarily unavailable."},
            ),
        )
        logger.warning("Durable CausalRCA intent %s remains pending", run_id)
        return False


def _reconcile_rca_dispatches(session_id):
    session = monitoring_history.read_session(session_id)
    if session.get("status") == "stopped":
        return
    for run in monitoring_history.list_rca_runs(session_id, page=1, page_size=100)["items"]:
        if run.get("status") == "queued" and run.get("dispatch_status") != "submitted":
            _dispatch_rca_run(session_id, run["run_id"])


def _reconcile_rca_history(session_id):
    for cycle_id in monitoring_history.pending_rca_cycles(session_id, limit=100):
        _process_cycle_rca(session_id, cycle_id)
    _reconcile_rca_dispatches(session_id)


def _process_cycle_rca(session_id, cycle_id):
    """Evaluate one committed cycle once and persist its trigger decision."""
    if monitoring_history.read_rca_event(session_id, cycle_id) is not None:
        return
    session = monitoring_history.read_session(session_id)
    state = session.get("rca")
    if not isinstance(state, dict):
        state = initial_rca_state(
            validate_rca_configuration({}), {"status": "incompatible", "reason": "Automatic RCA was not configured for this session."}, None
        )
    cycle = monitoring_history.read_cycle(session_id, cycle_id)
    updated, event = decide_trigger(state, cycle)
    if event["status"] != "trigger" or session.get("status") == "stopped":
        if session.get("status") == "stopped" and event["status"] == "trigger":
            event = {**event, "status": "suppressed_stopped", "reason": "Monitoring was stopped before RCA dispatch."}
        monitoring_history.record_rca_event(session_id, cycle_id, event, rca_state=updated)
        return
    active_runs = [
        item
        for item in monitoring_history.list_rca_runs(session_id, page=1, page_size=100)["items"]
        if item.get("status") in {"queued", "running"}
    ]
    if active_runs:
        updated["episode_state"] = "running_suppressed"
        updated["active_run_id"] = active_runs[0]["run_id"]
        monitoring_history.record_rca_event(
            session_id,
            cycle_id,
            {**event, "status": "suppressed_running", "reason": "Another CausalRCA run is already queued or running."},
            rca_state=updated,
        )
        return
    recipe = monitoring_history.read_recipe(session_id)
    readiness = assess_recipe_compatibility(recipe)
    if readiness.get("status") != "ready":
        updated["readiness"] = {"status": "incompatible", "reason": readiness.get("reason")}
        monitoring_history.record_rca_event(
            session_id,
            cycle_id,
            {**event, "status": "incompatible", "reason": readiness.get("reason")},
            rca_state=updated,
        )
        return
    cycle_input = monitoring_history.read_artifact(cycle["input_reference"])
    cycle_input["input_reference"] = cycle["input_reference"]
    cycle_input["input_sha256"] = cycle["input_sha256"]
    configuration_identity = {
        "threshold_percentage": updated["threshold_percentage"],
        "analysis_window_seconds": updated["analysis_window_seconds"],
        "cooldown_seconds": updated["cooldown_seconds"],
        "algorithm_id": updated.get("algorithm_id"),
        "algorithm_config_sha256": updated.get("algorithm_config_sha256"),
        "preprocessing_contract": updated.get("preprocessing_contract"),
    }
    detection_time = (cycle.get("prediction_window") or {}).get("end")
    base_run = {
        "status": "queued",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "trigger_percentage": cycle.get("anomaly_percentage"),
        "threshold_percentage": updated["threshold_percentage"],
        "trigger_window": cycle.get("prediction_window"),
        "context_window": cycle.get("inference_window"),
        "model_identity": {
            "model_id": session.get("model_id"),
            "artifact_id": session.get("artifact_id"),
            "artifact_manifest_sha256": session.get("artifact_manifest_sha256"),
        },
        "recipe_identity": {
            "contract": session.get("recipe_contract"),
            "recipe_sha256": session.get("recipe_sha256"),
            "feature_order_sha256": session.get("feature_order_sha256"),
        },
        "preprocessing": {
            "contract": updated.get("preprocessing_contract"),
            "configuration_sha256": rca_sha256({"feature_range": [0.0, 1.0]}),
        },
        "algorithm": {
            "id": updated.get("algorithm_id"),
            "configuration_sha256": updated.get("algorithm_config_sha256"),
        },
        "dispatch_status": "pending",
    }
    try:
        selected = select_analysis_input(
            cycle_input,
            end_timestamp=detection_time,
            window_seconds=updated["analysis_window_seconds"],
            feature_metadata=readiness["feature_metadata"],
            sampling_interval_seconds=recipe["sampling_interval_seconds"],
            minimum_observations=updated.get("minimum_observations", 2),
            minimum_features=updated.get("minimum_features", 2),
        )
        selected_sha = rca_sha256(selected)
        run_id = stable_run_id(session_id, cycle_id, selected_sha, configuration_identity)
        updated = mark_triggered(updated, run_id, cycle.get("completed_at") or cycle["committed_at"])
        run = {
            **base_run,
            "run_id": run_id,
            "episode_id": run_id,
            "cooldown_until": updated["cooldown_until"],
            "analysis_window": selected["analysis_window"],
            "observation_count": len(selected["matrix"]),
            "feature_count": len(selected["feature_order"]),
            "feature_identity": {
                "feature_order_sha256": selected.get("feature_order_sha256"),
                "feature_metadata_sha256": selected.get("feature_metadata_sha256"),
            },
        }
        monitoring_history.create_rca_run(
            session_id,
            cycle_id,
            run,
            input_payload=selected,
            event={**event, "status": "started", "episode_state": "triggered"},
            rca_state=updated,
        )
        _dispatch_rca_run(session_id, run_id)
    except MonitoringRcaError as exc:
        skipped_id = stable_run_id(session_id, cycle_id, cycle["input_sha256"], configuration_identity)
        updated = mark_triggered(updated, skipped_id, cycle.get("completed_at") or cycle["committed_at"])
        monitoring_history.create_rca_run(
            session_id,
            cycle_id,
            {**base_run, "run_id": skipped_id, "episode_id": skipped_id, "status": "skipped", "completed_at": utc_now(), "skip_reason": str(exc)[:500]},
            input_payload=None,
            event={**event, "status": "skipped", "reason": str(exc)[:500], "episode_state": "triggered"},
            rca_state=updated,
        )


@celery.task(bind=True)
def monitoring_task(self, session_id):
    """Run one recipe-pinned collection cycle and schedule the next cycle."""
    session = _read_session(session_id)
    if not session or session.get("status") == "stopped":
        return
    request_id = str(self.request.id or "unassigned")
    plan = session.get("inflight_cycle")
    if not isinstance(plan, dict) or plan.get("task_id") != request_id:
        end_time = int(time.time())
        start_time = end_time - int(session["window_seconds"])
        plan = {
            "task_id": request_id,
            "cycle_id": _cycle_id(session_id, start_time, end_time),
            "start_time": start_time,
            "end_time": end_time,
            "started_at": utc_now(),
        }
        session["inflight_cycle"] = plan
    cycle_started_at = plan["started_at"]
    start_time = int(plan["start_time"])
    end_time = int(plan["end_time"])
    cycle_id = plan["cycle_id"]
    collected = None
    bounded = None
    cycle_committed = False
    next_task = None
    try:
        session["status"] = "collecting"
        session["current_task_id"] = self.request.id
        if not _write_session(session):
            return
        collected = collect_recipe_window(
            session["recipe"], start_time, end_time, dict(app.config)
        )
        maximum = int(os.getenv("LIVE_MONITORING_MAX_BUFFER_OBSERVATIONS", "5000"))
        session["buffer"], late_count = merge_buffer_with_watermark(
            session.get("buffer", []),
            collected["timestamps"],
            collected["rows"],
            maximum,
            session.get("prediction_watermark"),
        )
        bounded = {
            **collected,
            "timestamps": [item["timestamp"] for item in session["buffer"]],
            "rows": [item["values"] for item in session["buffer"]],
        }
        runtime_result = _generic_detect(session, bounded)
        aligned = align_new_predictions(
            bounded["timestamps"], runtime_result, session.get("prediction_watermark")
        )
        warnings = list(collected["warnings"][:20])
        if late_count:
            warnings.append(
                f"Ignored {late_count} late or revised sample(s) at or before the committed prediction watermark."
            )
        prediction_count = aligned["prediction_count"]
        cycle_status = (
            "warmup"
            if not runtime_result.get("prediction_count")
            else "completed" if prediction_count else "no_new_data"
        )
        completed_at = utc_now()
        cycle = {
            "status": cycle_status,
            "artifact_identity": {
                "artifact_id": session["artifact_id"],
                "artifact_manifest_sha256": session["artifact_manifest_sha256"],
            },
            "query_window": {
                "start": datetime.fromtimestamp(start_time, timezone.utc).isoformat().replace("+00:00", "Z"),
                "end": datetime.fromtimestamp(end_time, timezone.utc).isoformat().replace("+00:00", "Z"),
            },
            "inference_window": {
                "start": bounded["timestamps"][0] if bounded["timestamps"] else None,
                "end": bounded["timestamps"][-1] if bounded["timestamps"] else None,
                "observation_count": len(bounded["timestamps"]),
            },
            "prediction_window": {
                "start": aligned["prediction_timestamps"][0] if prediction_count else None,
                "end": aligned["prediction_timestamps"][-1] if prediction_count else None,
            },
            "started_at": cycle_started_at,
            "completed_at": completed_at,
            "prediction_count": prediction_count,
            "anomaly_count": aligned["anomaly_count"],
            "anomaly_percentage": aligned["anomaly_percentage"],
            "warmup_observations": aligned["warmup_observations"],
            "warnings": warnings,
        }
        input_payload = {
            "schema_version": 1,
            "model_id": session["model_id"],
            "artifact_id": session["artifact_id"],
            "artifact_manifest_sha256": session["artifact_manifest_sha256"],
            "recipe_sha256": session["recipe_sha256"],
            "feature_order": bounded["feature_order"],
            "feature_order_sha256": bounded["feature_order_sha256"],
            "timestamps": bounded["timestamps"],
            "matrix": bounded["rows"],
        }
        result_payload = {
            "schema_version": 1,
            "model_id": session["model_id"],
            "artifact_id": session["artifact_id"],
            "artifact_manifest_sha256": session["artifact_manifest_sha256"],
            "warmup_observations": aligned["warmup_observations"],
            "prediction_timestamps": aligned["all_prediction_timestamps"],
            "binary_predictions": aligned["all_binary_predictions"],
            "anomaly_scores": aligned["all_anomaly_scores"],
            "new_prediction_timestamps": aligned["prediction_timestamps"],
            "new_binary_predictions": aligned["binary_predictions"],
            "new_anomaly_scores": aligned["anomaly_scores"],
        }
        persistent = monitoring_history.commit_cycle(
            session_id,
            cycle_id,
            input_payload=input_payload,
            result_payload=result_payload,
            cycle=cycle,
            next_watermark=aligned["next_watermark"],
            session_status="warmup" if cycle_status == "warmup" else "active",
        )
        cycle_committed = True
        monitoring_history.update_session(
            session_id,
            lambda record: record.update(
                buffered_observations=len(session["buffer"]),
                warmup_observations=aligned["warmup_observations"],
                warnings=warnings,
                error=None,
            ),
        )
        try:
            _reconcile_rca_history(session_id)
        except Exception:
            logger.exception("Automatic RCA processing failed for cycle %s without stopping monitoring", cycle_id)
        public_result = _public_latest_result(aligned, session["model_id"])
        session.update(
            {
                "status": "warmup" if cycle_status == "warmup" else "active",
                "iteration": int(session.get("iteration", 0)) + 1,
                "buffered_observations": len(session["buffer"]),
                "missing_values": collected["missing_values"],
                "warnings": warnings,
                "warmup_observations": aligned["warmup_observations"],
                "prediction_watermark": aligned["next_watermark"],
                "latest_result": public_result,
                "latest_cycle": monitoring_history.public_cycle(persistent),
                "last_successful_update": persistent["committed_at"],
                "error": None,
            }
        )
        session.pop("inflight_cycle", None)
        if not _write_session(session):
            return
        next_task = monitoring_task.apply_async(
            args=[session_id], countdown=int(session["poll_interval_seconds"])
        )
        session["next_task_id"] = next_task.id
        if not _write_session(session):
            celery.control.revoke(next_task.id, terminate=True)
    except MonitoringSessionStopped:
        return
    except Exception as exc:
        if next_task is not None:
            try:
                celery.control.revoke(next_task.id, terminate=True)
            except Exception:
                logger.warning("Could not revoke follow-up task for %s", session_id)
        session["status"] = "error"
        session["error"] = str(exc)[:500] if isinstance(exc, LiveMonitoringError) else "live monitoring failed"
        if bounded is not None and not cycle_committed:
            try:
                completed_at = utc_now()
                monitoring_history.commit_cycle(
                    session_id,
                    cycle_id,
                    input_payload={
                        "schema_version": 1,
                        "model_id": session["model_id"],
                        "artifact_id": session["artifact_id"],
                        "artifact_manifest_sha256": session["artifact_manifest_sha256"],
                        "recipe_sha256": session["recipe_sha256"],
                        "feature_order": bounded["feature_order"],
                        "feature_order_sha256": bounded["feature_order_sha256"],
                        "timestamps": bounded["timestamps"],
                        "matrix": bounded["rows"],
                    },
                    result_payload={
                        "schema_version": 1,
                        "model_id": session["model_id"],
                        "artifact_id": session["artifact_id"],
                        "artifact_manifest_sha256": session["artifact_manifest_sha256"],
                        "status": "failed",
                        "safe_error": {
                            "type": "monitoring_error",
                            "message": session["error"],
                        },
                    },
                    cycle={
                        "status": "failed",
                        "artifact_identity": {
                            "artifact_id": session["artifact_id"],
                            "artifact_manifest_sha256": session["artifact_manifest_sha256"],
                        },
                        "query_window": {
                            "start": datetime.fromtimestamp(start_time, timezone.utc).isoformat().replace("+00:00", "Z"),
                            "end": datetime.fromtimestamp(end_time, timezone.utc).isoformat().replace("+00:00", "Z"),
                        },
                        "inference_window": {
                            "start": bounded["timestamps"][0] if bounded["timestamps"] else None,
                            "end": bounded["timestamps"][-1] if bounded["timestamps"] else None,
                            "observation_count": len(bounded["timestamps"]),
                        },
                        "started_at": cycle_started_at,
                        "completed_at": completed_at,
                        "prediction_count": 0,
                        "anomaly_count": 0,
                        "anomaly_percentage": None,
                        "warmup_observations": 0,
                        "warnings": list(collected.get("warnings", []))[:20] if collected else [],
                        "safe_error": {
                            "type": "monitoring_error",
                            "message": session["error"],
                        },
                    },
                    next_watermark=session.get("prediction_watermark"),
                    session_status="failed",
                )
            except (MonitoringHistoryError, MonitoringSessionStopped):
                logger.error("Failed monitoring cycle could not be committed for %s", session_id)
        try:
            _write_session(session)
        except redis.RedisError:
            logger.warning("Redis execution state unavailable while failing %s", session_id)
        try:
            monitoring_history.fail_session(session_id, session["error"])
        except MonitoringHistoryError:
            logger.error("Durable monitoring failure state could not be written for %s", session_id)
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
        display_name = payload.get("display_name")
        if display_name is not None:
            if not isinstance(display_name, str) or len(display_name.strip()) > 120:
                return jsonify({"error": "display_name is invalid"}), 400
            display_name = display_name.strip() or None
        try:
            rca_configuration = validate_rca_configuration(payload.get("automatic_rca"))
        except MonitoringRcaError as exc:
            return jsonify({"error": str(exc)}), 400
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
        rca_readiness = assess_recipe_compatibility(recipe)
        rca_capability = None
        if rca_configuration["enabled"]:
            if rca_readiness.get("status") != "ready":
                return jsonify({"error": rca_readiness.get("reason") or "This model input is not compatible with CausalRCA."}), 409
            try:
                rca_capability = _crca_capability()
            except MonitoringRcaError as exc:
                return jsonify({"error": str(exc)}), 503
        session_id = f"monitor_{uuid.uuid4().hex}"
        now = utc_now()
        session = {
            "session_id": session_id,
            "display_name": display_name or f"{resolved.get('detector_id', 'Model')} monitoring",
            "model_id": model_id,
            "detector_id": resolved.get("detector_id"),
            "detector_version": resolved.get("detector_version"),
            "status": "queued",
            "created_at": now,
            "started_at": now,
            "artifact_manifest_sha256": resolved["artifact_manifest_sha256"],
            "artifact_id": resolved["artifact_id"],
            "inference_contract": resolved["inference_contract"],
            "artifact_format": resolved["artifact_format"],
            "recipe_contract": recipe["contract"],
            "recipe_sha256": recipe["recipe_sha256"],
            "feature_order_sha256": recipe["feature_order_sha256"],
            "sampling_interval_seconds": recipe["sampling_interval_seconds"],
            "window_seconds": window_seconds,
            "poll_interval_seconds": poll_seconds,
            "rca": initial_rca_state(rca_configuration, rca_readiness, rca_capability),
            "iteration": 0,
            "buffer": [],
            "buffered_observations": 0,
            "recipe": recipe,
        }
        monitoring_history.create_session(
            {key: value for key, value in session.items() if key not in {"recipe", "buffer"}},
            recipe,
        )
        _write_session(session)
        try:
            task = monitoring_task.apply_async(args=[session_id])
        except Exception:
            monitoring_history.fail_session(
                session_id, "monitoring task could not be dispatched", status="dispatch_failed"
            )
            raise
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
        history = monitoring_history.list_sessions(page=1, page_size=100)
        sessions = []
        for stored in history["items"]:
            live = _read_session(stored["session_id"])
            sessions.append(merge_execution_projection(stored, live))
        return jsonify(sessions), 200
    except Exception as e:
        logger.error(f"Error retrieving scheduled tasks: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': 'Monitoring History is temporarily unavailable.'}), 500


@app.route('/monitoring-sessions/<session_id>', methods=['GET'])
def monitoring_session_status(session_id):
    try:
        cycle_page = request.args.get("cycle_page", 1)
        cycle_page_size = request.args.get("cycle_page_size", 50)
        rca_page = request.args.get("rca_page", 1)
        rca_page_size = request.args.get("rca_page_size", 20)
        detail = monitoring_history.session_detail(
            session_id, cycle_page=cycle_page, cycle_page_size=cycle_page_size
        )
        detail["rca_runs"] = monitoring_history.list_rca_runs(
            session_id, page=rca_page, page_size=rca_page_size
        )
        live = _read_session(session_id)
        detail["session"] = merge_execution_projection(detail["session"], live)
        return jsonify(detail), 200
    except MonitoringSessionNotFound:
        return jsonify({"error": "unknown monitoring session"}), 404
    except MonitoringHistoryError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route('/monitoring-sessions/<session_id>/rca-runs/<run_id>', methods=['GET'])
def monitoring_rca_run_detail(session_id, run_id):
    try:
        return jsonify(monitoring_history.rca_run_detail(session_id, run_id)), 200
    except MonitoringSessionNotFound:
        return jsonify({"error": "unknown monitoring session"}), 404
    except MonitoringHistoryError as exc:
        status = 404 if "unknown" in str(exc) else 400
        return jsonify({"error": str(exc)}), status


@app.route('/monitoring-sessions/<session_id>/rca-reconcile', methods=['POST'])
def monitoring_rca_reconcile(session_id):
    try:
        session = monitoring_history.read_session(session_id)
        if session.get("status") == "stopped":
            return jsonify({"error": "monitoring session is stopped"}), 409
        _reconcile_rca_history(session_id)
        return jsonify({"status": "reconciled"}), 200
    except MonitoringSessionNotFound:
        return jsonify({"error": "unknown monitoring session"}), 404
    except MonitoringHistoryError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route('/monitoring-rca/readiness/<model_id>', methods=['GET'])
def monitoring_rca_readiness(model_id):
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", model_id or "") is None:
        return jsonify({"error": "a valid model_id is required"}), 400
    learning_url = os.getenv(
        "API_LEARNING_ADAPTATION_URL",
        "http://learning-adaptation-service.cloudsentinel.svc.cluster.local:80",
    ).rstrip("/")
    try:
        response = requests.get(
            f"{learning_url}/internal/models/{model_id}/live-input-recipe", timeout=(5, 30)
        )
        if response.status_code != 200:
            return jsonify({"status": "incompatible", "reason": "A validated live input recipe is unavailable."}), 200
        resolved = validate_resolved_live_model(response.json(), model_id)
        readiness = assess_recipe_compatibility(resolved["recipe"])
        return jsonify({key: readiness.get(key) for key in ("status", "reason", "input_contract")}), 200
    except (requests.RequestException, ValueError, LiveMonitoringError):
        return jsonify({"status": "unavailable", "reason": "RCA readiness could not be verified."}), 200


@app.route('/monitoring-sessions', methods=['GET'])
def monitoring_sessions():
    try:
        result = monitoring_history.list_sessions(
            page=request.args.get("page", 1),
            page_size=request.args.get("page_size", 20),
            status=request.args.get("status") or None,
            search=request.args.get("search") or None,
        )
        for item in result["items"]:
            live = _read_session(item["session_id"])
            item.update(merge_execution_projection(item, live))
        return jsonify(result), 200
    except MonitoringHistoryError as exc:
        return jsonify({"error": str(exc)}), 400


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
        try:
            persistent = monitoring_history.read_session(session_id)
        except MonitoringSessionNotFound:
            return jsonify({"error": "unknown monitoring session"}), 404
        if not session:
            session = persistent
        stopped_at = utc_now()
        monitoring_history.stop_session(session_id, stopped_at)
        try:
            redis_client.setex(
                _stop_key(session_id),
                int(os.getenv("LIVE_MONITORING_SESSION_TTL_SECONDS", str(30 * 24 * 3600))),
                "1",
            )
        except redis.RedisError:
            logger.warning("Redis stop marker unavailable for %s; durable stop remains authoritative", session_id)
        session = stopped_session(
            session,
            lambda task_id: celery.control.revoke(task_id, terminate=True),
            stopped_at=stopped_at,
        )
        try:
            _write_session(session, force=True)
        except redis.RedisError:
            pass
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
