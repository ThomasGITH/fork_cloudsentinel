"""Thin lazy adapter for the Isolation Forest plugin implementation."""

from __future__ import annotations

from importlib import import_module
from os import PathLike
from typing import Any


class IsolationForestAdapter:
    """Load the Isolation Forest runtime only when an operation is invoked."""

    @staticmethod
    def train(
        train_array: Any,
        artifact_dir: str | PathLike[str],
        *,
        training_parameters: dict[str, Any] | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        implementation = import_module("detectors.isolation_forest.implementation")
        return implementation.train_model(
            train_array,
            artifact_dir,
            training_parameters=training_parameters,
            model_id=model_id,
        )

    @staticmethod
    def evaluate(
        test_array: Any,
        anomaly_label_array: Any,
        artifact_dir: str | PathLike[str],
    ) -> dict[str, Any]:
        implementation = import_module("detectors.isolation_forest.implementation")
        return implementation.evaluate_model(test_array, anomaly_label_array, artifact_dir)

    @staticmethod
    def predict(
        test_array: Any,
        artifact_dir: str | PathLike[str],
    ) -> dict[str, Any]:
        implementation = import_module("detectors.isolation_forest.implementation")
        return implementation.predict_with_model(test_array, artifact_dir)

