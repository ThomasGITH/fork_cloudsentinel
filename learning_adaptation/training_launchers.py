"""Detector-specific preparation and dispatch for generic training runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from detectors.isolation_forest.training_request import (
    IsolationForestRequestError,
    validate_training_data,
    validate_training_parameters,
)

try:
    from learning_adaptation.training_run_storage import TrainingRunValidationError
except ModuleNotFoundError:  # The service image copies modules into /app.
    from training_run_storage import TrainingRunValidationError


@dataclass(frozen=True)
class PreparedTraining:
    task: Any
    args: list[Any]


def _load_csv(path: Path, name: str, dtype: Any = None) -> np.ndarray:
    try:
        return pd.read_csv(path, header=None, skip_blank_lines=False).to_numpy(dtype=dtype)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise TrainingRunValidationError(f"invalid {name} CSV: {exc}") from exc


class CGNNTrainingLauncher:
    detector_id = "cgnn"

    def __init__(self, task: Any):
        self.task = task

    def prepare(
        self,
        snapshot: dict[str, Any],
        child_input_dir: Path,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> PreparedTraining:
        train = _load_csv(snapshot["directory"] / "train.csv", "train", np.float32)
        test = _load_csv(snapshot["directory"] / "test.csv", "test", np.float32)
        labels = _load_csv(snapshot["directory"] / "labels.csv", "labels", np.float32)
        if train.ndim != 2 or test.ndim != 2 or labels.ndim not in (1, 2):
            raise TrainingRunValidationError("CGNN dataset matrices have invalid dimensions")
        if train.shape[1] != test.shape[1]:
            raise TrainingRunValidationError(
                "CGNN train and test data must have the same number of features"
            )
        labels_flat = labels.reshape(-1)
        if len(labels_flat) != len(test):
            raise TrainingRunValidationError(
                "CGNN labels must have the same number of observations as test data"
            )
        if snapshot.get("source", {}).get("type") == "catalogue":
            lookback = parameters["lookback"]
            if len(train) <= lookback or len(test) <= lookback:
                raise TrainingRunValidationError(
                    "CGNN catalogue partition needs more train and test observations "
                    f"than lookback={lookback}"
                )
            if parameters.get("feature_importance"):
                raise TrainingRunValidationError(
                    "CGNN feature importance is unavailable for arbitrary catalogue features"
                )

        # This deliberately mirrors data_processing.cgnn_preprocess.normalize_data.
        from sklearn.preprocessing import MinMaxScaler

        if np.any(sum(np.isnan(train))):
            train = np.nan_to_num(train)
        if np.any(sum(np.isnan(test))):
            test = np.nan_to_num(test)
        scaler = MinMaxScaler()
        scaler.fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)

        np.savetxt(child_input_dir / "train.csv", train, delimiter=",")
        np.savetxt(child_input_dir / "test.csv", test, delimiter=",")
        np.savetxt(child_input_dir / "labels.csv", labels_flat, delimiter=",")

        details = dict(snapshot["dataset_details"])
        details["dataset"] = context["dataset_id"]
        details.update(parameters)
        details["training_run_id"] = context["run_id"]
        details["orchestration_model_id"] = context["model_id"]
        train_info = {"data": details}
        return PreparedTraining(
            self.task,
            # Preserve the legacy /cgnn_train_model one-column label matrix.
            [train.tolist(), test.tolist(), labels.reshape(-1, 1).tolist(), train_info],
        )

    @staticmethod
    def dispatch(prepared: PreparedTraining) -> str:
        return prepared.task.apply_async(args=prepared.args).id


class IsolationForestTrainingLauncher:
    detector_id = "isolation-forest"

    def __init__(self, task: Any):
        self.task = task

    def prepare(
        self,
        snapshot: dict[str, Any],
        child_input_dir: Path,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> PreparedTraining:
        train = _load_csv(snapshot["directory"] / "train.csv", "train")
        test = _load_csv(snapshot["directory"] / "test.csv", "test")
        labels = _load_csv(snapshot["directory"] / "labels.csv", "labels")
        try:
            train, test, labels = validate_training_data(train, test, labels)
            parameters = validate_training_parameters(parameters)
        except IsolationForestRequestError as exc:
            raise TrainingRunValidationError(str(exc)) from exc

        for filename in ("train.csv", "test.csv", "labels.csv"):
            shutil.copyfile(snapshot["directory"] / filename, child_input_dir / filename)
        (child_input_dir / "parameters.json").write_text(
            json.dumps(parameters, indent=2, sort_keys=True), encoding="utf-8"
        )
        return PreparedTraining(
            self.task,
            [
                train.tolist(),
                test.tolist(),
                labels.tolist(),
                parameters,
                context["model_id"],
            ],
        )

    @staticmethod
    def dispatch(prepared: PreparedTraining) -> str:
        return prepared.task.apply_async(args=prepared.args).id


def build_training_launchers(cgnn_task: Any, isolation_forest_task: Any) -> dict[str, Any]:
    """Return the explicit launcher registry; it is separate from adapter discovery."""
    return {
        "cgnn": CGNNTrainingLauncher(cgnn_task),
        "isolation-forest": IsolationForestTrainingLauncher(isolation_forest_task),
    }
