# tasks.py

import os
import json
import logging
import shutil
import traceback
from pathlib import Path
import numpy as np
from celery import Celery, Task
from detectors import registry
from detectors.isolation_forest.training_request import validate_model_id
try:
    from learning_adaptation.isolation_forest_promotion import (
        IsolationForestPromotionError,
        promote_isolation_forest_model,
    )
except ModuleNotFoundError as exc:  # The service image copies this module beside tasks.py.
    if exc.name not in {
        "learning_adaptation",
        "learning_adaptation.isolation_forest_promotion",
    }:
        raise
    from isolation_forest_promotion import (
        IsolationForestPromotionError,
        promote_isolation_forest_model,
    )

# Configure and initialize Celery
celery = Celery(
    __name__,
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/2'),
    broker=os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/2'),
)

# testing purposes
# celery = Celery(__name__, backend='redis://localhost:6379/2', broker='redis://localhost:6379/2')


class CustomTask(Task):
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 1, 'countdown': 60}
    # OPTIONAL: Set time limits for the task
    # time_limit = 10800
    # soft_time_limit = 10000


celery.Task = CustomTask

celery.conf.update(
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    broker_heartbeat=0,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRAINED_MODELS_TEMP_ROOT = Path(
    os.getenv("TRAINED_MODELS_TEMP_ROOT", "trained_models_temp")
)
ISOLATION_FOREST_ARTIFACT_ROOT = TRAINED_MODELS_TEMP_ROOT / "isolation-forest"
ISOLATION_FOREST_EVALUATION_KEYS = (
    "precision",
    "recall",
    "f1",
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
    "test_observations",
    "actual_anomalies",
    "predicted_anomalies",
)


@celery.task(bind=True)
def train_and_evaluate_task(self, train_array, test_array, anomaly_label_array, train_info):
    """
    Celery task to train and evaluate a CGNN model.

    Args:
        self (Task): The Celery task instance.
        train_array (list): Training data array.
        test_array (list): Testing data array.
        anomaly_label_array (list): Anomaly label array.
        train_info (dict): Training information and settings.

    Returns:
        str: Success message if training is completed successfully.

    Raises:
        Exception: Retries the task if an exception occurs.
    """
    try:
        logger.info("Starting the training process.")

        adapter = registry.get_adapter("cgnn")

        # Update task state to EVALUATING
        self.update_state(state='INITIATING', meta='Initiating training')

        def progress_callback(progress_state, message, outer=None, total_outer=None, inner=None, total_inner=None):
            if message == '':
                self.update_state(state=progress_state, meta={
                    'outer': outer + 1,
                    'total_outer': total_outer,
                    'inner': inner + 1,
                    'total_inner': total_inner
                })
            else:
                self.update_state(state=progress_state, meta=message)

        # Train the model
        model_config, feature_importance = adapter.train(
            train_info['data'],
            np.array(train_array, dtype=np.float32),
            np.array(test_array, dtype=np.float32),
            np.array(anomaly_label_array, dtype=np.float32),
            progress_callback=progress_callback
        )

        logger.info("Training completed.")

        # Update task state to EVALUATING
        self.update_state(state='EVALUATING', meta='Evaluating the model')

        logger.info("Starting the evaluation process.")

        # Evaluate the model
        adapter.evaluate(
            model_config,
            np.array(train_array, dtype=np.float32),
            np.array(test_array, dtype=np.float32),
            np.array(anomaly_label_array, dtype=np.float32),
            progress_callback=progress_callback
        )

        logger.info("Evaluation completed.")
        if model_config['feature_importance']:
            # Generate feature names based on the given container and metric names
            feature_names = {}
            index = 0
            for container in train_info['data']["containers"]:
                for metric in train_info['data']["metrics"]:
                    feature_names[str(index)] = f"{container}_{metric}"
                    index += 1
            print(feature_importance)
            # Create a ranked list of features based on importance
            ranked_features = {name: feature_importance[int(idx)] for idx, name in feature_names.items()}
            ranked_features = dict(sorted(ranked_features.items(), key=lambda item: item[1], reverse=True))

            train_info['data']["ranked_features"] = ranked_features
        print(train_info['data'])

        # Save model parameters
        model_dir = TRAINED_MODELS_TEMP_ROOT / (
            f"{model_config['dataset']}_{model_config['id']}"
        )
        os.makedirs(model_dir, exist_ok=True)
        with open(model_dir / "model_params.json", "w") as f:
            json.dump(train_info['data'], f, indent=2)

        return "Training Successful"
    except Exception as e:
        logger.error(traceback.format_exc())
        raise self.retry(exc=e, countdown=60)


@celery.task(
    bind=True,
    dont_autoretry_for=(IsolationForestPromotionError,),
)
def train_and_evaluate_isolation_forest_task(
    self,
    train_array,
    test_array,
    anomaly_label_array,
    training_parameters,
    model_id,
):
    """Train and evaluate one Isolation Forest model through its plugin adapter."""
    model_id = validate_model_id(model_id)
    artifact_dir = ISOLATION_FOREST_ARTIFACT_ROOT / model_id
    adapter = registry.get_adapter("isolation-forest")

    self.update_state(state='INITIATING', meta='Initiating Isolation Forest training')
    self.update_state(state='TRAINING', meta='Training the Isolation Forest model')
    metadata = adapter.train(
        train_array,
        artifact_dir,
        training_parameters=training_parameters,
        model_id=model_id,
    )

    self.update_state(state='EVALUATING', meta='Evaluating the Isolation Forest model')
    evaluation = adapter.evaluate(test_array, anomaly_label_array, artifact_dir)
    compact_evaluation = {
        key: evaluation[key]
        for key in ISOLATION_FOREST_EVALUATION_KEYS
    }

    self.update_state(
        state='PROMOTING', meta='Promoting the Isolation Forest model'
    )
    promotion = promote_isolation_forest_model(artifact_dir, model_id)

    try:
        shutil.rmtree(artifact_dir)
    except OSError as exc:
        logger.warning(
            "Isolation Forest model %s was promoted, but local artifact cleanup failed: %s",
            model_id,
            exc,
        )

    self.update_state(
        state='COMPLETED', meta='Isolation Forest training and promotion completed'
    )

    return {
        "status": "completed",
        "detector_id": "isolation-forest",
        "model_id": model_id,
        "artifact_dir": f"isolation-forest/{model_id}",
        "training_parameters": metadata["training_parameters"],
        "n_features": metadata["n_features"],
        "evaluation": compact_evaluation,
        "promotion": promotion,
    }
