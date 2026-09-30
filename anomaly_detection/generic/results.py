"""Compact atomic detection-result storage."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid


class DetectionResultStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def write(self, payload: dict[str, Any]) -> str:
        result_id = f"detection_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)
        final = self.root / f"{result_id}.json"
        descriptor, temporary_name = tempfile.mkstemp(prefix=".detection.", dir=self.root)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as target:
                json.dump(payload, target, sort_keys=True)
                target.write("\n")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_name, final)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return result_id
