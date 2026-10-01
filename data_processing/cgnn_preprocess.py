import numpy as np
import pandas as pd
import requests
import json
import os
import re
from flask import Response
from ast import literal_eval
from sklearn.preprocessing import MinMaxScaler


SAFE_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
GENERIC_DETECTION_URL_ENV = "API_GENERIC_ANOMALY_DETECTION_URL"
LEARNING_API_URL_ENV = "API_LEARNING_ADAPTATION_URL"
DEFAULT_GENERIC_DETECTION_URL = (
    "http://generic-anomaly-detection-service.cloudsentinel.svc.cluster.local:80"
)
DEFAULT_LEARNING_API_URL = (
    "http://learning-adaptation-service.cloudsentinel.svc.cluster.local:80"
)
HTTP_TIMEOUT = (5.0, 120.0)


class GenericDetectionCallerError(RuntimeError):
    """Safe failure raised by the server-configured generic detection caller."""


def _configured_url(environment_name, default):
    value = os.getenv(environment_name, default).strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        raise GenericDetectionCallerError(
            f"{environment_name} is not configured with an HTTP service URL"
        )
    return value


def _model_identity(test_info):
    data = test_info.get("data") if isinstance(test_info, dict) else None
    model_id = (
        data.get("model_id") or data.get("model")
        if isinstance(data, dict)
        else None
    )
    if not isinstance(model_id, str) or not SAFE_MODEL_ID.fullmatch(model_id):
        raise GenericDetectionCallerError("a safe model_id is required for detection")
    return model_id


def _model_feature_identity(model_id):
    learning_url = _configured_url(LEARNING_API_URL_ENV, DEFAULT_LEARNING_API_URL)
    try:
        response = requests.get(
            f"{learning_url}/models/{model_id}", timeout=HTTP_TIMEOUT
        )
    except requests.RequestException as exc:
        raise GenericDetectionCallerError(
            "model metadata service is temporarily unavailable"
        ) from exc
    if response.status_code == 404:
        raise GenericDetectionCallerError("unknown model_id")
    if response.status_code >= 400:
        raise GenericDetectionCallerError("model metadata could not be retrieved")
    try:
        model = response.json()
    except ValueError as exc:
        raise GenericDetectionCallerError("model metadata response is invalid") from exc
    if not isinstance(model, dict) or model.get("model_id") != model_id:
        raise GenericDetectionCallerError("model metadata identity mismatch")
    if (model.get("inference") or {}).get("status") != "ready":
        raise GenericDetectionCallerError("selected model is not ready for generic inference")
    identity = model.get("feature_identity")
    if not isinstance(identity, dict):
        raise GenericDetectionCallerError("selected model has no feature identity")
    order = identity.get("feature_order") or []
    digest = identity.get("feature_order_sha256")
    if not isinstance(order, list) or any(not isinstance(item, str) or not item for item in order):
        raise GenericDetectionCallerError("selected model feature order is invalid")
    if not order and not isinstance(digest, str):
        raise GenericDetectionCallerError("selected model feature identity is incomplete")
    return {"feature_order": order, "feature_order_sha256": digest}


def _safe_detection_context(test_info):
    data = test_info.get("data", {}) if isinstance(test_info, dict) else {}
    allowed = {
        "task_id",
        "iteration",
        "start_time",
        "end_time",
        "containers",
        "metrics",
        "data_interval",
        "crca_threshold",
        "crca_pods",
    }
    return {key: data[key] for key in allowed if key in data}


def handle_cgnn_request(test_data, test_info):
    """
    Handles the request for CGNN anomaly detection.

    Args:
        test_data (FileStorage): The test data file uploaded by the user.
        test_info (dict): Information about the test including settings.

    Returns:
        Response: Flask response object containing the result from the anomaly detection API.
    """
    model_id = _model_identity(test_info)
    feature_identity = _model_feature_identity(model_id)
    test_array, _, _ = load_data(test_data)
    if not np.isfinite(test_array).all():
        raise GenericDetectionCallerError("detection matrix must contain finite values")
    if feature_identity["feature_order"] and len(feature_identity["feature_order"]) != test_array.shape[1]:
        raise GenericDetectionCallerError(
            "detection matrix feature count does not match the selected model"
        )

    runtime_url = _configured_url(
        GENERIC_DETECTION_URL_ENV, DEFAULT_GENERIC_DETECTION_URL
    )
    try:
        activation = requests.post(
            f"{runtime_url}/models/{model_id}/activate", timeout=HTTP_TIMEOUT
        )
    except requests.RequestException as exc:
        raise GenericDetectionCallerError(
            "generic detection runtime is temporarily unavailable"
        ) from exc
    if activation.status_code >= 400:
        return Response(
            json.dumps({"status": "error", "error": "selected model could not be activated"}),
            status=activation.status_code,
            content_type="application/json",
        )

    metadata = {
        "model_id": model_id,
        **feature_identity,
        "context": _safe_detection_context(test_info),
    }
    matrix_csv = pd.DataFrame(test_array).to_csv(index=False, header=False)
    try:
        response = requests.post(
            f"{runtime_url}/detect",
            files={"matrix": ("matrix.csv", matrix_csv, "text/csv")},
            data={"metadata": json.dumps(metadata)},
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise GenericDetectionCallerError(
            "generic detection runtime is temporarily unavailable"
        ) from exc

    flask_response = Response(
        response.content,
        status=response.status_code,
        content_type=response.headers.get('Content-Type', 'application/json')
    )
    return flask_response


def handle_cgnn_train_request(cgnn_data, train_info):
    """
    Handles the request for CGNN training.

    Args:
        cgnn_data (dict): A dictionary containing the training, test, and anomaly label files.
        train_info (dict): Information about the training including settings and data specifics.

    Returns:
        Response: Flask response object containing the result from the training API.
    """
    test_array, train_array, anomaly_label_array = load_data(cgnn_data['test_array'],
                                                             cgnn_data['train_array'],
                                                             cgnn_data['anomaly_label_array'],
                                                             train_info['data']['anomaly_sequence'])
    train_data, test_data, anomaly_data = get_data(test_array, test_array.shape[1],
                                                   train_array, anomaly_label_array)
    train_files = {
        'train_array': train_data,
        'test_array': test_data,
        'anomaly_label_array': anomaly_data
    }
    train_info_json = json.dumps(train_info)
    response = requests.post(f"{train_info['settings']['API_LEARNING_ADAPTATION_URL']}/cgnn_train_model",
                             files=train_files, data={'train_info': train_info_json})

    flask_response = Response(
        response.content,
        status=response.status_code,
        content_type=response.headers['Content-Type']
    )
    return flask_response


def load_data(test_data, train_data=None, anomaly_label=None, anomaly_sequence="False"):
    """
    Loads and processes the data files.

    Args:
        test_data (FileStorage): The test data file.
        train_data (FileStorage, optional): The training data file. Defaults to None.
        anomaly_label (FileStorage, optional): The anomaly label file. Defaults to None.
        anomaly_sequence (str, optional): Indicates if the anomaly sequence is in string format. Defaults to "False".

    Returns:
        tuple: Processed test array, train array, and anomaly label array.
    """
    test_df = pd.read_csv(test_data, header=None)
    test_array = test_df.to_numpy(dtype=np.float32)
    if train_data is not None:
        train_df = pd.read_csv(train_data, header=None)
        train_array = train_df.to_numpy(dtype=np.float32)
    else:
        train_array = None
    if anomaly_label is not None:
        if anomaly_sequence == "True":
            anomalies = literal_eval(anomaly_label, header=None)
            length = len(test_array)
            label = np.zeros([length], dtype=np.float32)
            for anomaly in anomalies:
                label[anomaly[0]: anomaly[1] + 1] = 1.0
            anomaly_label_array = np.asarray(label)
        else:
            anomaly_label_df = pd.read_csv(anomaly_label, header=None)
            anomaly_label_array = anomaly_label_df.to_numpy(dtype=np.float32)
    else:
        anomaly_label_array = None
    return test_array, train_array, anomaly_label_array


def get_data(test_array, data_dim, train_array=None, anomaly_label_array=None):
    """
    Processes and normalizes the data arrays.

    Args:
        test_array (np.ndarray): The test data array.
        data_dim (int): The dimension of the data.
        train_array (np.ndarray, optional): The training data array. Defaults to None.
        anomaly_label_array (np.ndarray, optional): The anomaly label array. Defaults to None.

    Returns:
        tuple: CSV formatted strings of the processed and normalized data arrays.
    """
    test_data = test_array.reshape((-1, data_dim))
    if train_array is not None:
        train_data = train_array.reshape((-1, data_dim))
    if anomaly_label_array is not None:
        anomaly_data = anomaly_label_array.reshape((-1))

    if anomaly_label_array is not None:
        train_data, scaler = normalize_data(train_data, scaler=None)
        test_data, _ = normalize_data(test_data, scaler=scaler)
        print("train set shape: ", train_data.shape)
        print("test set shape: ", test_data.shape)
        print("test set label shape: ", None if anomaly_data is None else anomaly_data.shape)
        return (pd.DataFrame(train_data).to_csv(index=False, header=False),
                pd.DataFrame(test_data).to_csv(index=False, header=False),
                pd.DataFrame(anomaly_data).to_csv(index=False, header=False))
    else:
        test_data, _ = normalize_data(test_data, scaler=None)
        print("test set shape: ", test_data.shape)
        return pd.DataFrame(test_data).to_csv(index=False, header=False)


def normalize_data(data, scaler=None):
    """
    Normalizes the data using MinMaxScaler.

    Args:
        data (np.ndarray): The data array to be normalized.
        scaler (MinMaxScaler, optional): Pre-fitted scaler to be used for normalization. Defaults to None.

    Returns:
        tuple: The normalized data array and the scaler used.
    """
    data = np.asarray(data, dtype=np.float32)

    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")

    return data, scaler
