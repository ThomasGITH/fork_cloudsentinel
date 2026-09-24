"""Shared interface for anomaly detector adapters."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DetectorAdapter(Protocol):
    """Minimal operations required by the current detector integration."""

    def train(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        ...
