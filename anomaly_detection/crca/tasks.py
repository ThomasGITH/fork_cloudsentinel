# tasks.py
from celery import Celery
import os
import pandas as pd
import logging
import traceback
from sklearn.preprocessing import MinMaxScaler

try:
    from .crca import run_crca
    from .config import get_config
    from .monitoring_result import safe_monitoring_result
except ImportError:  # Flat module layout inside the service image.
    from crca import run_crca
    from config import get_config
    from monitoring_result import safe_monitoring_result

try:
    from monitoring_history import MonitoringHistoryStore, MonitoringHistoryError, canonical_sha256
except ModuleNotFoundError:  # Repository-root test imports.
    from data_ingestion.monitoring_history import MonitoringHistoryStore, MonitoringHistoryError, canonical_sha256

# Configure and initialize Celery
celery = Celery(
    __name__,
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/1'),
    broker=os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/1'),
)

# testing purposes
# celery = Celery(__name__, backend='redis://localhost:6379/1', broker='redis://localhost:6379/1')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@celery.task(bind=True)
def run_crca_task(self, crca_data_json, crca_info):
    """
    Celery task to perform CRCA.

    Args:
        self (Task): The Celery task instance.
        crca_data_json (dict): Data to perform CRCA on.
        crca_info (dict): Information required for CRCA.

    Returns:
        dict: CRCA results
    """
    try:
        crca_data = pd.read_json(crca_data_json, orient='split')
        task_id = self.request.id

        response_data = run_crca(crca_data, crca_info, task_id)
        logger.info(response_data)
        return response_data
    except Exception as e:
        logger.error(traceback.format_exc())
        raise self.retry(exc=e, countdown=60)


def _history_store():
    return MonitoringHistoryStore(
        os.getenv("MONITORING_HISTORY_STORAGE_ROOT", "/app/storage/monitoring_history")
    )


@celery.task(bind=True, name="run_monitoring_rca_task")
def run_monitoring_rca_task(self, session_id, run_id):
    """Execute one durable monitoring RCA intent and persist before returning."""
    store = _history_store()
    claimed = store.claim_rca_run(session_id, run_id)
    if claimed is None:
        return {"run_id": run_id, "status": store.read_rca_run(session_id, run_id).get("status")}
    try:
        input_payload = store.read_artifact(claimed["input_reference"])
        expected_config = claimed.get("algorithm", {}).get("configuration_sha256")
        actual_config = canonical_sha256(get_config())
        if expected_config != actual_config:
            raise ValueError("CausalRCA configuration no longer matches the pinned run")
        matrix = input_payload.get("matrix")
        metadata = input_payload.get("feature_metadata")
        if not isinstance(matrix, list) or not isinstance(metadata, list):
            raise ValueError("CausalRCA input is invalid")
        frame = pd.DataFrame(matrix, columns=[item["feature_name"] for item in metadata])
        if frame.shape[0] < 2 or frame.shape[1] < 2:
            raise ValueError("CausalRCA input has insufficient dimensions")
        scaled = MinMaxScaler().fit_transform(frame)
        scaled_frame = pd.DataFrame(scaled, columns=frame.columns)
        info = {
            "data": {
                "feature_names": list(frame.columns),
                "feature_metadata": metadata,
                "crca_pods": sorted({item["entity_id"] for item in metadata}),
                "metrics": sorted({item["metric_name"] for item in metadata}),
            }
        }
        response = run_crca(scaled_frame, info, run_id)
        result = safe_monitoring_result(response, metadata)
        result.update(
            run_id=run_id,
            session_id=session_id,
            input_sha256=claimed.get("input_sha256"),
            preprocessing={
                "contract": "cloudsentinel.crca-minmax/v1",
                "configuration": {"feature_range": [0.0, 1.0]},
                "configuration_sha256": canonical_sha256({"feature_range": [0.0, 1.0]}),
            },
            algorithm={
                "id": "cloudsentinel.causal-rca/legacy-v1",
                "configuration": get_config(),
                "configuration_sha256": actual_config,
            },
        )
        store.complete_rca_run(session_id, run_id, result)
        return {"run_id": run_id, "status": "completed"}
    except Exception as exc:
        try:
            store.fail_rca_run(session_id, run_id, "CausalRCA execution failed.")
        except MonitoringHistoryError:
            logger.error("Could not persist failure for monitoring RCA run %s", run_id)
        logger.error("Monitoring CausalRCA run %s failed", run_id)
        return {"run_id": run_id, "status": "failed"}
