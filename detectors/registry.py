"""Discover and validate detector manifests without loading detector code."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import re
from typing import Any

import yaml

from .contracts import DetectorAdapter


SUPPORTED_SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 1_000_000
REQUIRED_FIELDS = {
    "schema_version",
    "id",
    "name",
    "version",
    "description",
    "supported_modalities",
    "input_requirements",
    "training_parameters",
    "entry_point",
}
PARAMETER_TYPES = {"boolean", "integer", "number", "string"}
ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")
ENTRY_POINT_PATTERN = re.compile(
    r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$"
)


class ManifestValidationError(ValueError):
    """Raised when a detector manifest is unsafe or violates the schema."""


class UnknownDetectorError(LookupError):
    """Raised when an adapter is requested for an unknown detector ID."""


class InvalidEntryPointError(ManifestValidationError):
    """Raised when a manifest entry point cannot identify an adapter object."""


class AdapterImportError(ImportError):
    """Raised when an adapter entry-point module cannot be imported."""


class AdapterContractError(TypeError):
    """Raised when a resolved adapter does not implement the required operations."""


def _fail(path: Path, message: str) -> ManifestValidationError:
    return ManifestValidationError(f"{path}: {message}")


def _require_non_empty_string(manifest: dict[str, Any], field: str, path: Path) -> None:
    value = manifest[field]
    if not isinstance(value, str) or not value.strip():
        raise _fail(path, f"'{field}' must be a non-empty string")


def _validate_default(parameter_name: str, metadata: dict[str, Any], path: Path) -> None:
    parameter_type = metadata["type"]
    default = metadata["default"]
    nullable = metadata.get("nullable", False)

    if default is None:
        if not nullable:
            raise _fail(path, f"training parameter '{parameter_name}' has a null default but is not nullable")
        return

    valid_type = {
        "boolean": lambda value: isinstance(value, bool),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
        "string": lambda value: isinstance(value, str),
    }[parameter_type]
    if not valid_type(default):
        raise _fail(
            path,
            f"training parameter '{parameter_name}' default does not match type '{parameter_type}'",
        )


def validate_manifest(manifest: Any, path: Path) -> dict[str, Any]:
    """Validate a decoded manifest and return it unchanged."""
    if not isinstance(manifest, dict):
        raise _fail(path, "manifest root must be a mapping")

    missing = sorted(REQUIRED_FIELDS.difference(manifest))
    if missing:
        raise _fail(path, f"missing required fields: {', '.join(missing)}")

    if manifest["schema_version"] != SUPPORTED_SCHEMA_VERSION:
        raise _fail(
            path,
            f"unsupported schema_version {manifest['schema_version']!r}; expected {SUPPORTED_SCHEMA_VERSION}",
        )

    for field in ("id", "name", "version", "description", "entry_point"):
        _require_non_empty_string(manifest, field, path)

    if not ID_PATTERN.fullmatch(manifest["id"]):
        raise _fail(path, "'id' must use lowercase letters, digits, hyphens, or underscores")
    if not ENTRY_POINT_PATTERN.fullmatch(manifest["entry_point"]):
        raise InvalidEntryPointError(
            f"{path}: 'entry_point' must have the form 'python.module:attribute'"
        )

    modalities = manifest["supported_modalities"]
    if (
        not isinstance(modalities, list)
        or not modalities
        or any(not isinstance(item, str) or not item.strip() for item in modalities)
    ):
        raise _fail(path, "'supported_modalities' must be a non-empty list of strings")
    if len(set(modalities)) != len(modalities):
        raise _fail(path, "'supported_modalities' must not contain duplicates")

    if not isinstance(manifest["input_requirements"], dict):
        raise _fail(path, "'input_requirements' must be a mapping")

    parameters = manifest["training_parameters"]
    if not isinstance(parameters, dict):
        raise _fail(path, "'training_parameters' must be a mapping")
    for parameter_name, metadata in parameters.items():
        if not isinstance(parameter_name, str) or not parameter_name:
            raise _fail(path, "training parameter names must be non-empty strings")
        if not isinstance(metadata, dict):
            raise _fail(path, f"training parameter '{parameter_name}' metadata must be a mapping")
        missing_metadata = {"type", "default", "description"}.difference(metadata)
        if missing_metadata:
            raise _fail(
                path,
                f"training parameter '{parameter_name}' is missing: {', '.join(sorted(missing_metadata))}",
            )
        if metadata["type"] not in PARAMETER_TYPES:
            raise _fail(path, f"training parameter '{parameter_name}' has an unsupported type")
        if not isinstance(metadata["description"], str) or not metadata["description"].strip():
            raise _fail(path, f"training parameter '{parameter_name}' needs a description")
        if "nullable" in metadata and not isinstance(metadata["nullable"], bool):
            raise _fail(path, f"training parameter '{parameter_name}' nullable must be boolean")
        _validate_default(parameter_name, metadata, path)

    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read one YAML manifest with safe parsing and bounded input size."""
    manifest_path = Path(path)
    try:
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise _fail(manifest_path, f"manifest exceeds {MAX_MANIFEST_BYTES} bytes")
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = yaml.safe_load(manifest_file)
    except ManifestValidationError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise _fail(manifest_path, f"cannot read YAML: {exc}") from exc
    return validate_manifest(manifest, manifest_path)


def discover_detectors(detector_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Return validated manifests without importing their entry points."""
    root = Path(detector_dir) if detector_dir is not None else Path(__file__).parent
    root = root.resolve()
    if not root.is_dir():
        raise ManifestValidationError(f"{root}: detector directory does not exist")

    manifests: list[dict[str, Any]] = []
    seen_ids: dict[str, Path] = {}
    for path in sorted(root.glob("*/manifest.yaml")):
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(root):
            raise _fail(path, "manifest resolves outside the detector directory")
        manifest = load_manifest(resolved_path)
        detector_id = manifest["id"]
        if detector_id in seen_ids:
            raise _fail(
                path,
                f"duplicate detector id '{detector_id}' (already defined in {seen_ids[detector_id]})",
            )
        seen_ids[detector_id] = path
        manifests.append(manifest)
    return manifests


def get_adapter(
    detector_id: str,
    detector_dir: str | Path | None = None,
) -> DetectorAdapter:
    """Resolve one detector adapter from its manifest without eager plugin imports."""
    manifest = next(
        (
            candidate
            for candidate in discover_detectors(detector_dir)
            if candidate["id"] == detector_id
        ),
        None,
    )
    if manifest is None:
        raise UnknownDetectorError(f"Unknown detector id: {detector_id!r}")

    entry_point = manifest["entry_point"]
    if not ENTRY_POINT_PATTERN.fullmatch(entry_point):
        raise InvalidEntryPointError(
            f"Detector {detector_id!r} has invalid entry point {entry_point!r}"
        )
    module_name, attribute_name = entry_point.split(":", 1)

    try:
        module = import_module(module_name)
    except Exception as exc:
        raise AdapterImportError(
            f"Could not import adapter module {module_name!r} for detector {detector_id!r}: {exc}"
        ) from exc

    try:
        adapter_class = getattr(module, attribute_name)
    except AttributeError as exc:
        raise InvalidEntryPointError(
            f"Adapter entry point {entry_point!r} for detector {detector_id!r} does not exist"
        ) from exc

    if not isinstance(adapter_class, type):
        raise InvalidEntryPointError(
            f"Adapter entry point {entry_point!r} for detector {detector_id!r} must resolve to a class"
        )

    try:
        adapter = adapter_class()
    except Exception as exc:
        raise AdapterContractError(
            f"Adapter {entry_point!r} for detector {detector_id!r} could not be instantiated: {exc}"
        ) from exc

    if not isinstance(adapter, DetectorAdapter):
        missing_operations = [
            operation
            for operation in ("train", "evaluate", "predict")
            if not callable(getattr(adapter, operation, None))
        ]
        details = ", ".join(missing_operations) or "invalid operation definitions"
        raise AdapterContractError(
            f"Adapter {entry_point!r} for detector {detector_id!r} does not satisfy "
            f"DetectorAdapter: {details}"
        )

    return adapter
