"""Thin, lazy bridge to the existing CGNN runtime implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any


class CGNNAdapter:
    """Delegate plugin operations without changing legacy CGNN behaviour."""

    @staticmethod
    def train(
        dataset_config: dict[str, Any],
        train_array: Any,
        test_array: Any,
        anomaly_label_array: Any,
        progress_callback: Any = None,
    ) -> Any:
        train = import_module("cgnn.train").train
        return train(
            dataset_config,
            train_array,
            test_array,
            anomaly_label_array,
            progress_callback=progress_callback,
        )

    @staticmethod
    def evaluate(
        model_config: dict[str, Any],
        train_array: Any,
        test_array: Any,
        anomaly_label_array: Any,
        progress_callback: Any = None,
        save_output: bool = True,
    ) -> Any:
        predict_and_evaluate = import_module("cgnn.evaluate_prediction").predict_and_evaluate
        return predict_and_evaluate(
            model_config,
            train_array,
            test_array,
            anomaly_label_array,
            progress_callback=progress_callback,
            save_output=save_output,
        )

    @staticmethod
    def predict(test_array: Any, dataset: str, save_output: bool = True) -> Any:
        load_model_and_predict = import_module("predict").load_model_and_predict
        return load_model_and_predict(test_array, dataset, save_output=save_output)
