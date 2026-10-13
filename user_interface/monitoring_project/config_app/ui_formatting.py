"""Shared, presentation-only formatting for CloudSentinel UI values."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def format_ui_datetime(value: Any, fallback: str = "—") -> str:
    """Render an ISO timestamp in a stable, explicit UTC representation."""
    if value is None or value == "":
        return fallback
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return value
    else:
        return str(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
