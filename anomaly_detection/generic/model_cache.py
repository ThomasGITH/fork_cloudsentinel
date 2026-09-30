"""Small synchronized LRU cache for opaque loaded detector models."""

from __future__ import annotations

from collections import OrderedDict
import threading
from typing import Any, Callable


class ModelCache:
    def __init__(self, maximum_entries: int = 4):
        if maximum_entries < 1 or maximum_entries > 64:
            raise ValueError("model cache size must be between 1 and 64")
        self.maximum_entries = maximum_entries
        self._items: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: tuple[str, str]) -> Any | None:
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def load(self, key: tuple[str, str], loader: Callable[[], Any]) -> Any:
        with self._lock:
            existing = self._items.get(key)
            if existing is not None:
                self._items.move_to_end(key)
                return existing
            value = loader()
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.maximum_entries:
                self._items.popitem(last=False)
            return value

    def contains(self, key: tuple[str, str]) -> bool:
        with self._lock:
            return key in self._items
