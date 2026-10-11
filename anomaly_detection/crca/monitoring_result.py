"""Validation of legacy CausalRCA output for the durable monitoring contract."""

from __future__ import annotations

import base64
import csv
from io import StringIO
import math
from typing import Any


def safe_monitoring_result(
    response: dict[str, Any], feature_metadata: list[dict[str, Any]]
) -> dict[str, Any]:
    ranking_csv = response.get("ranking")
    if not isinstance(ranking_csv, str):
        raise ValueError("CausalRCA did not return a ranking")
    allowed = {item["feature_name"]: item for item in feature_metadata}
    ranking = []
    for row in csv.DictReader(StringIO(ranking_csv)):
        label = row.get("metric")
        if label not in allowed:
            raise ValueError("CausalRCA returned an unknown feature")
        try:
            index = int(row.get("index"))
            score = float(row.get("score"))
        except (TypeError, ValueError) as exc:
            raise ValueError("CausalRCA returned an invalid ranking") from exc
        metadata = allowed[label]
        if index != metadata["column_index"] or not math.isfinite(score):
            raise ValueError("CausalRCA returned an invalid ranking")
        ranking.append(
            {
                "rank": len(ranking) + 1,
                "column_index": index,
                "feature_name": label,
                "entity_type": metadata["entity_type"],
                "entity_id": metadata["entity_id"],
                "metric_name": metadata["metric_name"],
                "score": score,
            }
        )
    graphs = []
    for encoded in response.get("graph_image", [])[:3]:
        if not isinstance(encoded, str) or len(encoded) > 3_000_000:
            raise ValueError("CausalRCA returned an invalid graph")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("CausalRCA returned an invalid graph") from exc
        if not decoded.startswith(b"\x89PNG\r\n\x1a\n") or len(decoded) > 2_000_000:
            raise ValueError("CausalRCA returned an invalid graph")
        graphs.append({"media_type": "image/png", "base64": encoded})
    return {"schema_version": 1, "ranking": ranking, "graphs": graphs}
