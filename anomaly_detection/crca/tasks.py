# tasks.py
from celery import Celery
import math
import os
import pandas as pd
import logging
import traceback
from sklearn.preprocessing import MinMaxScaler

try:
    from .crca import run_crca
    from .config import get_config
    from .monitoring_result import (
        normalize_feature_metadata,
        safe_monitoring_result,
        validate_pinned_method_contract,
    )
    from .baro_adapter import (
        BARO_METHOD_ID,
        BARO_PREPROCESSING_CONTRACT,
        BaroRobustScorerAdapter,
        configuration as baro_configuration,
        configuration_sha256 as baro_configuration_sha256,
    )
except ImportError:  # Flat module layout inside the service image.
    from crca import run_crca
    from config import get_config
    from monitoring_result import (
        normalize_feature_metadata,
        safe_monitoring_result,
        validate_pinned_method_contract,
    )
    from baro_adapter import (
        BARO_METHOD_ID,
        BARO_PREPROCESSING_CONTRACT,
        BaroRobustScorerAdapter,
        configuration as baro_configuration,
        configuration_sha256 as baro_configuration_sha256,
    )

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

RCA_METHOD_CAUSAL_LEGACY = "cloudsentinel.causal-rca/legacy-v1"
RCA_METHOD_BARO = "cloudsentinel.baro-robust-scorer/v1"
DEFAULT_RCA_METHOD = RCA_METHOD_CAUSAL_LEGACY
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


def _run_causal_legacy(store, claimed, session_id, run_id):
    input_payload = store.read_artifact(claimed["input_reference"])
    validate_pinned_method_contract(
        claimed, input_payload, RCA_METHOD_CAUSAL_LEGACY
    )
    expected_config = claimed.get("algorithm", {}).get("configuration_sha256")
    actual_config = canonical_sha256(get_config())
    if expected_config != actual_config:
        raise ValueError("CausalRCA configuration no longer matches the pinned run")
    matrix = input_payload.get("matrix")
    metadata = input_payload.get("feature_metadata")
    if not isinstance(matrix, list) or not isinstance(metadata, list):
        raise ValueError("CausalRCA input is invalid")
    metadata = normalize_feature_metadata(
        metadata, len(matrix[0]) if matrix and isinstance(matrix[0], list) else None
    )
    frame = pd.DataFrame(matrix, columns=[item["feature_name"] for item in metadata])
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ValueError("CausalRCA input has insufficient dimensions")
    if any(isinstance(value, bool) for row in matrix for value in row):
        raise ValueError("CausalRCA input must contain finite numeric values")
    try:
        numeric_frame = frame.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("CausalRCA input must contain finite numeric values") from exc
    if not numeric_frame.map(math.isfinite).values.all():
        raise ValueError("CausalRCA input must contain finite numeric values")
    scaled = MinMaxScaler().fit_transform(numeric_frame)
    scaled_frame = pd.DataFrame(scaled, columns=frame.columns)
    info = {
        "data": {
            "feature_names": list(frame.columns),
            "feature_metadata": metadata,
            "crca_pods": sorted(
                {
                    item["entity_id"]
                    for item in metadata
                    if item.get("entity_type") == "pod" and item.get("entity_id")
                }
            ),
            "metrics": sorted({item["metric_name"] for item in metadata}),
        }
    }
    response = run_crca(scaled_frame, info, run_id)
    result = safe_monitoring_result(response, metadata)
    result.update(
        run_id=run_id,
        session_id=session_id,
        method=RCA_METHOD_CAUSAL_LEGACY,
        input_sha256=claimed.get("input_sha256"),
        preprocessing={
            "contract": "cloudsentinel.crca-minmax/v1",
            "configuration": {"feature_range": [0.0, 1.0]},
            "configuration_sha256": canonical_sha256({"feature_range": [0.0, 1.0]}),
        },
        algorithm={
            "id": RCA_METHOD_CAUSAL_LEGACY,
            "configuration": get_config(),
            "configuration_sha256": actual_config,
        },
    )
    store.complete_rca_run(session_id, run_id, result)


def _run_baro_robust_scorer(store, claimed, session_id, run_id):
    expected_config = claimed.get("algorithm", {}).get("configuration_sha256")
    actual_config = baro_configuration_sha256()
    if expected_config != actual_config:
        raise ValueError("BARO configuration no longer matches the pinned run")
    input_payload = store.read_artifact(claimed["input_reference"])
    validate_pinned_method_contract(claimed, input_payload, RCA_METHOD_BARO)
    result = BaroRobustScorerAdapter().run(input_payload)
    result.update(
        run_id=run_id,
        session_id=session_id,
        input_sha256=claimed.get("input_sha256"),
        preprocessing={
            "contract": BARO_PREPROCESSING_CONTRACT,
            "configuration": {
                "input": "raw_finite_numeric",
                "baseline_normalization": "fse-baro RobustScorer",
            },
            "configuration_sha256": canonical_sha256({
                "input": "raw_finite_numeric",
                "baseline_normalization": "fse-baro RobustScorer",
            }),
        },
        algorithm={
            "id": BARO_METHOD_ID,
            "configuration": baro_configuration(),
            "configuration_sha256": actual_config,
        },
    )
    store.complete_rca_run(session_id, run_id, result)


RCA_METHOD_RUNNERS = {
    RCA_METHOD_CAUSAL_LEGACY: _run_causal_legacy,
    RCA_METHOD_BARO: _run_baro_robust_scorer,
}


def _resolve_method(record, requested_method=None):
    method = record.get("method") if isinstance(record, dict) else None
    algorithm = record.get("algorithm") if isinstance(record, dict) else None
    algorithm_id = algorithm.get("id") if isinstance(algorithm, dict) else None
    if isinstance(method, str) and isinstance(algorithm_id, str) and method != algorithm_id:
        raise ValueError("RCA method does not match algorithm provenance")
    # A durable legacy record without either field is always CausalRCA. The
    # transient dispatch argument may verify that identity, never redefine it.
    method = method or algorithm_id or DEFAULT_RCA_METHOD
    if requested_method is not None and method != requested_method:
        raise ValueError("RCA method does not match the pinned run")
    return method


@celery.task(bind=True, name="run_monitoring_rca_task")
def run_monitoring_rca_task(self, session_id, run_id, method=None):
    """Execute one durable method-pinned RCA intent and persist before returning."""
    store = _history_store()
    claimed = store.claim_rca_run(session_id, run_id)
    if claimed is None:
        return {"run_id": run_id, "status": store.read_rca_run(session_id, run_id).get("status")}
    try:
        method_id = _resolve_method(claimed, method)
        runner = RCA_METHOD_RUNNERS.get(method_id)
        if runner is None:
            raise ValueError("The selected RCA method is unavailable")
        runner(store, claimed, session_id, run_id)
        return {"run_id": run_id, "status": "completed"}
    except Exception:
        try:
            store.fail_rca_run(session_id, run_id, "RCA execution failed.")
        except MonitoringHistoryError:
            logger.error("Could not persist failure for monitoring RCA run %s", run_id)
        logger.error("Monitoring RCA run %s failed", run_id)
        return {"run_id": run_id, "status": "failed"}
