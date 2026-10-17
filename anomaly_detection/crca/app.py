from flask import Flask, request, jsonify
import logging
import json
import os
import traceback
import shutil
import hashlib
import re
import pandas as pd
from flask_cors import CORS
from celery import Celery
try:
    from .tasks import run_crca_task, run_monitoring_rca_task
    from .config import set_config, get_config, set_initial_config
    from .baro_adapter import capability as baro_capability
except ImportError:  # Flat module layout inside the service image.
    from tasks import run_crca_task, run_monitoring_rca_task
    from config import set_config, get_config, set_initial_config
    from baro_adapter import capability as baro_capability

# Set multiprocessing start method to 'spawn' for Celery compatibility
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Initialize Flask application
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Set the environment variable
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# Configure and initialize Celery instance
app.config['broker_url'] = os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/1')
app.config['result_backend'] = os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/1')

# testing purposes
# app.config['broker_url'] = 'redis://localhost:6379/1'
# app.config['result_backend'] = 'redis://localhost:6379/1'

celery = Celery(app.import_name, backend=app.config['result_backend'], broker=app.config['broker_url'])
celery.conf.update(
    app.config,
    broker_connection_retry_on_startup=True,
    worker_cancel_long_running_tasks_on_connection_loss=True,
    task_acks_late=True,  # If you are using late acknowledgments
    worker_prefetch_multiplier=1,  # Example configuration to avoid over-fetching
)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RCA_METHOD_CAUSAL_LEGACY = "cloudsentinel.causal-rca/legacy-v1"
RCA_METHOD_BARO = "cloudsentinel.baro-robust-scorer/v1"
DEFAULT_RCA_METHOD = RCA_METHOD_CAUSAL_LEGACY
RCA_METHOD_TASKS = {
    RCA_METHOD_CAUSAL_LEGACY: run_monitoring_rca_task,
    RCA_METHOD_BARO: run_monitoring_rca_task,
}


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "healthy"}), 200


@app.route('/internal/monitoring-capability', methods=['GET'])
def monitoring_capability():
    config_sha256 = _canonical_sha256(get_config())
    baro = baro_capability()
    return jsonify(
        {
            "contract": "cloudsentinel.monitoring-rca/v1",
            "input_contract": "cloudsentinel.crca-input/v1",
            "algorithm_id": RCA_METHOD_CAUSAL_LEGACY,
            "preprocessing_contract": "cloudsentinel.crca-minmax/v1",
            "minimum_observations": 2,
            "minimum_features": 2,
            "config_sha256": config_sha256,
            "default_method": DEFAULT_RCA_METHOD,
            "methods": [
                {
                    "id": RCA_METHOD_CAUSAL_LEGACY,
                    "display_name": "CausalRCA",
                    "description": "Dependency-oriented analysis that ranks features and can produce a causal dependency graph.",
                    "input_type": "A finite, time-aligned metrics matrix with explicit feature identities.",
                    "output_type": "A ranked suspect list and, when available, a dependency graph.",
                    "status": "available",
                    "input_contract": "cloudsentinel.crca-input/v1",
                    "preprocessing_contract": "cloudsentinel.crca-minmax/v1",
                    "minimum_requirements": {"observations": 2, "features": 2},
                    "config_sha256": config_sha256,
                },
                baro,
            ],
        }
    ), 200


@app.route('/internal/monitoring-runs', methods=['POST'])
def submit_monitoring_run():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "a JSON request is required"}), 400
    session_id = payload.get("session_id")
    run_id = payload.get("run_id")
    method = payload.get("method", DEFAULT_RCA_METHOD)
    safe = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    if not isinstance(session_id, str) or not safe.fullmatch(session_id):
        return jsonify({"error": "session_id is invalid"}), 400
    if not isinstance(run_id, str) or not safe.fullmatch(run_id):
        return jsonify({"error": "run_id is invalid"}), 400
    if method not in {RCA_METHOD_CAUSAL_LEGACY, RCA_METHOD_BARO}:
        return jsonify({"error": "RCA method is unknown"}), 400
    task_runner = RCA_METHOD_TASKS.get(method)
    if task_runner is None:
        return jsonify({"error": "RCA method is unavailable"}), 409
    if method == RCA_METHOD_BARO and baro_capability().get("status") != "available":
        return jsonify({"error": "RCA method is unavailable"}), 409
    try:
        task_runner.apply_async(
            args=[session_id, run_id, method], task_id=f"monitoring-rca-{run_id}"
        )
    except Exception:
        logger.error("Monitoring RCA dispatch failed")
        return jsonify({"error": "RCA dispatch failed"}), 503
    return jsonify({"status": "submitted", "run_id": run_id}), 202

# Command to run the Celery worker:
# celery -A app.celery worker --loglevel=info -P gevent


def save_crca(response_data, task_id):
    """
    Save CRCA results to a JSON file.

    Args:
        response_data (dict): CRCA results to be saved.
        task_id (str): Unique identifier of the task.

    Returns:
        None
    """
    try:
        # Save response data to a JSON file
        path = 'results/' + task_id
        if not os.path.isdir(path):
            os.mkdir(path)
        with open(path + '/crca_results.json', 'w') as f:
            json.dump(response_data, f, indent=2)
        logger.info("JSON object created and saved for task %s", task_id)
    except Exception as e:
        logger.error("Error in save_crca: %s", traceback.format_exc())
        raise e


@app.route('/crca', methods=['POST'])
def crca():
    """
    Endpoint to initiate CRCA (Change Root Cause Analysis) task.

    Expects:
    - 'crca_file': CSV file containing CRCA data.
    - 'crca_info': JSON string containing additional CRCA information.

    Returns:
    - JSON response with 'task_id' upon successful initiation.
    - Error response if an exception occurs.
    """
    try:
        logger.info("Received request to initiate CRCA task")
        crca_file = request.files['crca_file']
        crca_data = pd.read_csv(crca_file, header=None)
        crca_data_json = crca_data.to_json(orient='split')
        crca_info_json = request.form.get('crca_info')
        crca_info = json.loads(crca_info_json)

        # Initiate CRCA task asynchronously
        task = run_crca_task.apply_async(args=[crca_data_json, crca_info])
        logger.info(f"CRCA task initiated with task_id: {task.id}")

        return jsonify({"task_id": task.id}), 202

    except Exception as e:
        logger.error(f"Error initiating CRCA task: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/get_status/<task_id>', methods=['GET'])
def get_status(task_id):
    """
    Endpoint to retrieve the status of a specific task identified by task_id.

    Expects:
    - task_id: Unique identifier of the task.

    Returns:
    - JSON response with task state and result if available.
    - Error response if the task_id does not exist or an exception occurs.
    """
    try:
        logger.info(f"Received request to get status for task_id: {task_id}")
        task = run_crca_task.AsyncResult(task_id)

        if task.state == 'PENDING':
            response = {
                'state': task.state,
                'status': 'Pending...'
            }
        elif task.state != 'FAILURE':
            response = {
                'state': task.state,
                'result': task.result
            }
        else:
            response = {
                'state': task.state,
                'status': str(task.info)
            }
        logger.info(f"Status for task_id {task_id}: {response}")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error retrieving status for task_id {task_id}: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/results/<task_id>', methods=['GET'])
def get_results(task_id):
    """
    Endpoint to retrieve results of a specific CRCA task identified by task_id.

    Expects:
    - task_id: Unique identifier of the task whose results are to be retrieved.

    Returns:
    - JSON response with CRCA results.
    - Error response if the task_id does not exist or an exception occurs.
    """
    try:
        logger.info(f"Received request to get results for task_id: {task_id}")
        task = run_crca_task.AsyncResult(task_id)

        logger.info(f"Task state for task_id {task_id}: {task.state}")

        if task.state == 'SUCCESS':
            logger.info(f"Task result for task_id {task_id}: {task.result}")
            save_crca(task.result, task_id)
            data = task.result
            logger.info(f"Results retrieved and saved for task_id: {task_id}")
        else:
            data = {"error": "Results not available yet or task failed"}
            logger.info(f"No results found for task_id: {task_id}")

        return jsonify(data), 200

    except Exception as e:
        logger.error(f"Error retrieving results for task_id {task_id}: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# @app.route('/results/<task_id>', methods=['GET'])
# def get_results(task_id):
#     """
#     Endpoint to retrieve results of a specific CRCA task identified by task_id.

#     Expects:
#     - task_id: Unique identifier of the task whose results are to be retrieved.

#     Returns:
#     - JSON response with CRCA results.
#     - Error response if the task_id does not exist or an exception occurs.
#     """
#     try:
#         logger.info(f"Received request to get results for task_id: {task_id}")
#         path = 'results/'+str(task_id)
#         if os.path.exists(path):
#             with open(path + '/crca_results.json', 'r') as f:
#                 data = json.load(f)
#             logger.info(f"Results retrieved for task_id: {task_id}")
#         else:
#             data = {"error": "Results not found"}
#             logger.info(f"No results found for task_id: {task_id}")
#         return jsonify(data), 200

#     except Exception as e:
#         logger.error(f"Error retrieving results for task_id {task_id}: {traceback.format_exc()}")
#         return jsonify({"error": str(e)}), 500

@app.route('/delete_results/<task_id>', methods=['DELETE'])
def delete_results(task_id):
    """
    Endpoint to delete results associated with a specific CRCA task identified by task_id.

    Expects:
    - task_id: Unique identifier of the task whose results are to be deleted.

    Returns:
    - JSON response indicating success upon successful deletion.
    - Error response if the task_id does not exist or an exception occurs.
    """
    try:
        logger.info(f"Received request to delete results for task_id: {task_id}")
        if os.path.exists('results/'+task_id):
            shutil.rmtree('results/'+task_id)
            logger.info(f"Results deleted for task_id: {task_id}")
        else:
            logger.info(f"No results found for task_id: {task_id}")
        return jsonify({"success": True}), 200

    except Exception as e:
        logger.error(f"Error deleting results for task_id {task_id}: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/get_config', methods=['GET'])
def get_config_route():
    """
    Endpoint to retrieve the current configuration settings.

    Returns:
    - JSON response with current configuration settings.
    - Error response if an exception occurs while retrieving the configuration.
    """
    try:
        logger.info("Received request to get configuration")
        config = get_config()
        logger.info("Configuration retrieved successfully")
        return jsonify(config), 200

    except Exception as e:
        logger.error(f"Error retrieving configuration: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/update_config', methods=['POST'])
def update_config():
    """
    Endpoint to update the configuration settings.

    Expects:
    - JSON payload containing new configuration settings.

    Returns:
    - JSON response indicating success upon successful update.
    - Error response if an exception occurs while updating the configuration.
    """
    new_config = request.json
    try:
        logger.info("Received request to update configuration")
        set_config(new_config)
        logger.info("Configuration updated successfully")
        return jsonify({"status": "success", "message": "Configuration updated successfully"}), 200

    except Exception as e:
        logger.error(f"Error updating configuration: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    logger.info("Initializing configuration")
    set_initial_config()
    logger.info("Starting Flask app")
    app.run(debug=False, host='0.0.0.0', port=5023)
