"""Thin, lazy bridge to the existing CGNN runtime implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from detectors.contracts import (
    ArtifactResult,
    DetectorCompatibilityError,
    DetectorExecutionError,
    ProgressReporter,
    PromotionResult,
    TrainingContext,
    TrainingResult,
)


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

    @staticmethod
    def _load_training_data(context: TrainingContext) -> tuple[Any, Any, Any]:
        try:
            np = import_module("numpy")
            pd = import_module("pandas")
            train = pd.read_csv(
                context.data.train_path, header=None, skip_blank_lines=False
            ).to_numpy(dtype=np.float32)
            test = pd.read_csv(
                context.data.test_path, header=None, skip_blank_lines=False
            ).to_numpy(dtype=np.float32)
            if context.data.labels_path is None:
                raise DetectorCompatibilityError(
                    "CGNN requires labels for its current training-time evaluation"
                )
            labels = pd.read_csv(
                context.data.labels_path, header=None, skip_blank_lines=False
            ).to_numpy(dtype=np.float32)
        except DetectorCompatibilityError:
            raise
        except Exception as exc:
            raise DetectorCompatibilityError(f"CGNN input CSV is invalid: {exc}") from exc
        return train, test, labels

    def validate_training(self, context: TrainingContext) -> None:
        train, test, labels = self._load_training_data(context)
        if train.ndim != 2 or test.ndim != 2 or labels.ndim not in (1, 2):
            raise DetectorCompatibilityError("CGNN dataset matrices have invalid dimensions")
        if train.shape[1] != test.shape[1]:
            raise DetectorCompatibilityError(
                "CGNN train and test data must have the same number of features"
            )
        if labels.reshape(-1).shape[0] != test.shape[0]:
            raise DetectorCompatibilityError(
                "CGNN labels must have the same number of observations as test data"
            )
        if context.dataset.get("source") == "catalogue":
            lookback = context.parameters["lookback"]
            if train.shape[0] <= lookback or test.shape[0] <= lookback:
                raise DetectorCompatibilityError(
                    "CGNN needs more train and test observations than "
                    f"lookback={lookback}"
                )
            if context.parameters.get("feature_importance"):
                raise DetectorCompatibilityError(
                    "CGNN feature importance is unavailable for arbitrary catalogue features"
                )

    def run_training(
        self, context: TrainingContext, progress: ProgressReporter
    ) -> TrainingResult:
        try:
            np = import_module("numpy")
            scaler_class = import_module("sklearn.preprocessing").MinMaxScaler
            train, test, labels = self._load_training_data(context)

            # Preserve the preprocessing performed by the former CGNN launcher.
            if np.any(sum(np.isnan(train))):
                train = np.nan_to_num(train)
            if np.any(sum(np.isnan(test))):
                test = np.nan_to_num(test)
            scaler = scaler_class()
            scaler.fit(train)
            train = scaler.transform(train)
            test = scaler.transform(test)

            dataset_config = dict(context.snapshot.get("dataset_details", {}))
            dataset_config["dataset"] = context.dataset["dataset_id"]
            dataset_config.update(dict(context.parameters))
            dataset_config["training_run_id"] = context.run_id
            dataset_config["orchestration_model_id"] = context.model_id
            dataset_config["orchestration_context"] = dict(context.model_record_context)

            active_phase = "TRAINING"

            def legacy_progress(
                state: str, message: str, *values: Any, **named_values: Any
            ) -> None:
                detail: Any = message or {
                    "values": [value for value in values if value is not None][:8],
                    **{
                        key: value
                        for key, value in named_values.items()
                        if value is not None
                    },
                }
                progress.report(
                    state if state in progress.ALLOWED_STATES else active_phase,
                    detail,
                )

            progress.report("TRAINING", "Training the CGNN model")
            model_config, feature_importance = self.train(
                dataset_config,
                train,
                test,
                labels,
                progress_callback=legacy_progress,
            )
            active_phase = "EVALUATING"
            progress.report("EVALUATING", "Evaluating the CGNN model")
            evaluation_result = self.evaluate(
                model_config,
                train,
                test,
                labels,
                progress_callback=legacy_progress,
            )

            artifact_dir = context.artifact_workspace / (
                f"{model_config['dataset']}_{model_config['id']}"
            )
            ranked_features = None
            if model_config.get("feature_importance") and feature_importance is not None:
                containers = dataset_config.get("containers", [])
                metrics = dataset_config.get("metrics", [])
                names = [f"{container}_{metric}" for container in containers for metric in metrics]
                ranked_features = dict(
                    sorted(
                        zip(names, feature_importance),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )
                dataset_config["ranked_features"] = ranked_features

            json_module = import_module("json")
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "model_params.json").write_text(
                json_module.dumps(dataset_config, indent=2), encoding="utf-8"
            )
            evaluation = evaluation_result if isinstance(evaluation_result, dict) else {}
            evaluation_path = artifact_dir / "model_evaluation.json"
            if evaluation_path.is_file():
                stored = json_module.loads(evaluation_path.read_text(encoding="utf-8"))
                if isinstance(stored, dict):
                    evaluation = stored
            return TrainingResult(
                status="completed",
                artifact=ArtifactResult(
                    directory=artifact_dir,
                    required_files=(
                        "model.pt",
                        "model_config.json",
                        "model_evaluation.json",
                        "model_params.json",
                    ),
                    format="torch-state-dict",
                    safe_reference=context.model_id,
                ),
                evaluation=evaluation,
                promotion=PromotionResult(status="not_promoted"),
                model_metadata={
                    "n_features": int(train.shape[1]),
                    "feature_importance_available": ranked_features is not None,
                },
                cleanup="keep",
            )
        except DetectorCompatibilityError:
            raise
        except Exception as exc:
            raise DetectorExecutionError(f"CGNN training failed: {exc}") from exc
