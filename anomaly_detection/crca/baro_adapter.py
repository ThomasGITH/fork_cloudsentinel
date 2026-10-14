"""Isolated CloudSentinel adapter for fse-baro's RobustScorer component."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from typing import Any, Callable

import pandas as pd
from sklearn.preprocessing import RobustScaler

try:
    from .monitoring_result import normalize_feature_metadata
except ImportError:  # Flat module layout inside the service image.
    from monitoring_result import normalize_feature_metadata


BARO_PACKAGE = "fse-baro"
BARO_PACKAGE_VERSION = "0.2.2"
BARO_METHOD_ID = "cloudsentinel.baro-robust-scorer/v1"
BARO_INPUT_CONTRACT = "cloudsentinel.baro-input/v1"
BARO_PREPROCESSING_CONTRACT = "cloudsentinel.baro-robust-scorer-raw/v1"
BARO_ADAPTER_CONTRACT = "cloudsentinel.baro-adapter/v1"
BARO_MINIMUM_BASELINE = 2
BARO_MINIMUM_POST_CHANGE = 2
BARO_MINIMUM_FEATURES = 2


class BaroAdapterError(ValueError):
    """Safe BARO adapter contract error."""


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def configuration() -> dict[str, Any]:
    return {
        "adapter_contract": BARO_ADAPTER_CONTRACT,
        "method": BARO_METHOD_ID,
        "package": BARO_PACKAGE,
        "package_version": BARO_PACKAGE_VERSION,
        "component": "RobustScorer",
        "bocpd": False,
        "minimum_baseline_observations": BARO_MINIMUM_BASELINE,
        "minimum_post_change_observations": BARO_MINIMUM_POST_CHANGE,
    }


def configuration_sha256() -> str:
    return _sha256(configuration())


def _load_official_scorer() -> Callable[..., dict[str, Any]]:
    installed = importlib_metadata.version(BARO_PACKAGE)
    if installed != BARO_PACKAGE_VERSION:
        raise BaroAdapterError("BARO runtime dependency version is incompatible")
    try:
        from baro.root_cause_analysis import robust_scorer
    except (ImportError, AttributeError) as exc:
        raise BaroAdapterError("BARO RobustScorer is unavailable") from exc
    if not callable(robust_scorer):
        raise BaroAdapterError("BARO RobustScorer is unavailable")
    return robust_scorer


def capability() -> dict[str, Any]:
    try:
        _load_official_scorer()
    except (BaroAdapterError, importlib_metadata.PackageNotFoundError):
        return {
            "id": BARO_METHOD_ID,
            "display_name": "BARO RobustScorer",
            "status": "unavailable",
            "reason_code": "dependency_unavailable",
            "reason": "BARO RobustScorer is unavailable in this runtime.",
            "input_contract": BARO_INPUT_CONTRACT,
            "preprocessing_contract": BARO_PREPROCESSING_CONTRACT,
            "minimum_requirements": {
                "observations": BARO_MINIMUM_BASELINE + BARO_MINIMUM_POST_CHANGE,
                "features": BARO_MINIMUM_FEATURES,
                "baseline_observations": BARO_MINIMUM_BASELINE,
                "post_change_observations": BARO_MINIMUM_POST_CHANGE,
            },
        }
    return {
        "id": BARO_METHOD_ID,
        "display_name": "BARO RobustScorer",
        "status": "available",
        "input_contract": BARO_INPUT_CONTRACT,
        "preprocessing_contract": BARO_PREPROCESSING_CONTRACT,
        "minimum_requirements": {
            "observations": BARO_MINIMUM_BASELINE + BARO_MINIMUM_POST_CHANGE,
            "features": BARO_MINIMUM_FEATURES,
            "baseline_observations": BARO_MINIMUM_BASELINE,
            "post_change_observations": BARO_MINIMUM_POST_CHANGE,
        },
        "config_sha256": configuration_sha256(),
    }


class BaroRobustScorerAdapter:
    def __init__(
        self,
        scorer: Callable[..., dict[str, Any]] | None = None,
        *,
        package_version: str = BARO_PACKAGE_VERSION,
    ):
        self._scorer = scorer or _load_official_scorer()
        self.package_version = package_version

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        if (
            not isinstance(payload, dict)
            or (payload.get("input_contract") or payload.get("contract")) != BARO_INPUT_CONTRACT
        ):
            raise BaroAdapterError("BARO input contract is invalid")
        matrix = payload.get("matrix")
        feature_metadata = payload.get("feature_metadata")
        if not isinstance(matrix, list) or not matrix:
            raise BaroAdapterError("BARO input matrix is invalid")
        width = len(matrix[0]) if isinstance(matrix[0], list) else 0
        metadata = normalize_feature_metadata(feature_metadata, width)
        if width < BARO_MINIMUM_FEATURES or any(
            not isinstance(row, list) or len(row) != width for row in matrix
        ):
            raise BaroAdapterError("BARO input matrix dimensions are invalid")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for row in matrix
            for value in row
        ):
            raise BaroAdapterError("BARO input must contain finite numeric values")

        change_point = payload.get("change_point")
        if not isinstance(change_point, dict):
            raise BaroAdapterError("BARO change-point evidence is missing")
        index = change_point.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise BaroAdapterError("BARO change-point index is invalid")
        if index < BARO_MINIMUM_BASELINE:
            raise BaroAdapterError("BARO input has insufficient baseline observations")
        if len(matrix) - index < BARO_MINIMUM_POST_CHANGE:
            raise BaroAdapterError("BARO input has insufficient post-change observations")
        if (
            change_point.get("baseline_observation_count") != index
            or change_point.get("anomalous_observation_count") != len(matrix) - index
        ):
            raise BaroAdapterError("BARO change-point counts are invalid")
        timestamps = payload.get("timestamps")
        digest = change_point.get("cycle_result_sha256")
        if (
            not isinstance(timestamps, list)
            or len(timestamps) != len(matrix)
            or not isinstance(change_point.get("detected_change_point"), str)
            or change_point["detected_change_point"] != timestamps[index]
            or change_point.get("source") != "stored_monitoring_cycle_result"
            or not isinstance(change_point.get("cycle_id"), str)
            or not change_point["cycle_id"]
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise BaroAdapterError("BARO change-point provenance is invalid")

        aliases = [f"cs_feature_{position:04d}" for position in range(width)]
        frame = pd.DataFrame(matrix, columns=aliases, dtype=float)
        response = self._scorer(frame, anomalies=[index], dk_select_useful=False)
        ranks = response.get("ranks") if isinstance(response, dict) else None
        if not isinstance(ranks, list) or not ranks:
            raise BaroAdapterError("BARO RobustScorer returned no ranking")
        if len(set(ranks)) != len(ranks) or any(alias not in aliases for alias in ranks):
            raise BaroAdapterError("BARO RobustScorer returned an invalid ranking")

        scores = {}
        baseline = frame.iloc[:index]
        anomalous = frame.iloc[index:]
        for alias in ranks:
            scaler = RobustScaler().fit(baseline[[alias]])
            values = scaler.transform(anomalous[[alias]])[:, 0]
            score = float(max(values))
            if not math.isfinite(score):
                raise BaroAdapterError("BARO RobustScorer returned a non-finite score")
            scores[alias] = score
        expected_order = sorted(ranks, key=lambda alias: scores[alias], reverse=True)
        if ranks != expected_order:
            raise BaroAdapterError("BARO RobustScorer ranking and scores are inconsistent")

        ranking = []
        for rank, alias in enumerate(ranks, start=1):
            column_index = aliases.index(alias)
            item = metadata[column_index]
            ranking.append({
                "rank": rank,
                "column_index": column_index,
                "feature_id": item["feature_id"],
                "feature_name": item["feature_name"],
                "entity": item["entity"],
                "entity_type": item["entity_type"],
                "entity_id": item["entity_id"],
                "metric_name": item["metric_name"],
                "score": scores[alias],
            })
        evidence = {
            key: change_point.get(key)
            for key in (
                "source",
                "cycle_id",
                "cycle_result_sha256",
                "detected_change_point",
                "index",
                "baseline_observation_count",
                "anomalous_observation_count",
            )
        }
        return {
            "schema_version": 1,
            "method": BARO_METHOD_ID,
            "ranking": ranking,
            "graphs": [],
            "change_point_evidence": evidence,
        }
