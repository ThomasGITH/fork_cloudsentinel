"""Discovery and adapter resolution for CloudSentinel detector plugins."""

from .contracts import DetectorAdapter
from .registry import (
    AdapterContractError,
    AdapterImportError,
    InvalidEntryPointError,
    ManifestValidationError,
    UnknownDetectorError,
    discover_detectors,
    get_adapter,
    load_manifest,
)

__all__ = [
    "AdapterContractError",
    "AdapterImportError",
    "DetectorAdapter",
    "InvalidEntryPointError",
    "ManifestValidationError",
    "UnknownDetectorError",
    "discover_detectors",
    "get_adapter",
    "load_manifest",
]
