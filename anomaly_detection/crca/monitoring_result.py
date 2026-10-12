"""Validation of legacy CausalRCA output for the durable monitoring contract."""

from __future__ import annotations

import base64
import csv
from io import StringIO
import math
from typing import Any


def normalize_feature_metadata(
    feature_metadata: Any, expected_columns: int | None = None
) -> list[dict[str, Any]]:
    """Validate the exact matrix-column identity and normalize legacy flat entities."""
    if not isinstance(feature_metadata, list) or len(feature_metadata) < 2:
        raise ValueError("CausalRCA feature metadata is invalid")
    if expected_columns is not None and len(feature_metadata) != expected_columns:
        raise ValueError("CausalRCA feature metadata does not match the matrix")
    normalized = []
    seen_ids = set()
    seen_names = set()
    for expected_index, item in enumerate(feature_metadata):
        if not isinstance(item, dict) or item.get("column_index") != expected_index:
            raise ValueError("CausalRCA feature metadata column order is invalid")
        feature_id = item.get("feature_id")
        feature_name = item.get("feature_name")
        metric_name = item.get("metric_name")
        if (
            not isinstance(feature_id, str)
            or not feature_id
            or not isinstance(feature_name, str)
            or not feature_name
            or feature_name != feature_id
            or not isinstance(metric_name, str)
            or not metric_name
            or feature_id in seen_ids
            or feature_name in seen_names
        ):
            raise ValueError("CausalRCA feature metadata identity is invalid")
        seen_ids.add(feature_id)
        seen_names.add(feature_name)

        entity = item.get("entity")
        flat_type = item.get("entity_type")
        flat_id = item.get("entity_id")
        if entity is not None:
            if (
                not isinstance(entity, dict)
                or not isinstance(entity.get("type"), str)
                or not entity["type"]
                or not isinstance(entity.get("id"), str)
                or not entity["id"]
                or not isinstance(entity.get("source"), str)
                or not entity["source"]
            ):
                raise ValueError("CausalRCA feature entity metadata is invalid")
            if (flat_type not in (None, entity["type"])) or (
                flat_id not in (None, entity["id"])
            ):
                raise ValueError("CausalRCA feature entity metadata conflicts")
            safe_entity = {
                "type": entity["type"],
                "id": entity["id"],
                "source": entity["source"],
            }
        elif flat_type is not None or flat_id is not None:
            # Backward compatibility for already queued v1 RCA inputs.
            if (
                not isinstance(flat_type, str)
                or not flat_type
                or not isinstance(flat_id, str)
                or not flat_id
            ):
                raise ValueError("CausalRCA legacy feature entity metadata is invalid")
            safe_entity = {"type": flat_type, "id": flat_id, "source": "legacy_explicit"}
        else:
            safe_entity = None
        normalized.append(
            {
                "column_index": expected_index,
                "feature_id": feature_id,
                "feature_name": feature_name,
                "metric_name": metric_name,
                "entity": safe_entity,
                "entity_type": safe_entity["type"] if safe_entity else None,
                "entity_id": safe_entity["id"] if safe_entity else None,
            }
        )
    return normalized


def safe_monitoring_result(
    response: dict[str, Any], feature_metadata: list[dict[str, Any]]
) -> dict[str, Any]:
    ranking_csv = response.get("ranking")
    if not isinstance(ranking_csv, str):
        raise ValueError("CausalRCA did not return a ranking")
    normalized_metadata = normalize_feature_metadata(feature_metadata)
    allowed = {item["feature_name"]: item for item in normalized_metadata}
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
                "feature_id": metadata["feature_id"],
                "entity": metadata["entity"],
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
