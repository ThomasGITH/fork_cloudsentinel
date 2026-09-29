"""Discover and validate detector manifests without loading detector code."""

from __future__ import annotations

from importlib import import_module
import json
import os
from pathlib import Path
import re
from typing import Any

import yaml

from .contracts import DetectorAdapter, TrainableDetectorAdapter


SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_TRAINING_PROTOCOLS = {"cloudsentinel.training/v1"}
SUPPORTED_INPUT_PROFILES = {"metrics-partition-v1"}
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


class UnsupportedTrainingProtocolError(ManifestValidationError):
    """Raised when a detector requests an unsupported training protocol."""


class ParameterValidationError(ValueError):
    """Raised when submitted values violate manifest parameter metadata."""


def _fail(path: Path, message: str) -> ManifestValidationError:
    return ManifestValidationError(f"{path}: {message}")


def _detector_root(detector_dir: str | Path | None) -> Path:
    if detector_dir is not None:
        return Path(detector_dir)
    configured = os.getenv("DETECTOR_PLUGIN_ROOT")
    return Path(configured) if configured else Path(__file__).parent


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


def _validate_parameter_constraints(
    parameter_name: str, metadata: dict[str, Any], path: Path
) -> None:
    parameter_type = metadata["type"]
    if "advanced" in metadata and not isinstance(metadata["advanced"], bool):
        raise _fail(path, f"training parameter '{parameter_name}' advanced must be boolean")

    numeric = parameter_type in {"integer", "number"}
    bounds = ("minimum", "exclusive_minimum", "maximum", "exclusive_maximum")
    for field in bounds:
        if field not in metadata:
            continue
        value = metadata[field]
        if not numeric or not isinstance(value, (int, float)) or isinstance(value, bool):
            raise _fail(
                path,
                f"training parameter '{parameter_name}' {field} requires a numeric parameter",
            )
    if "minimum" in metadata and "exclusive_minimum" in metadata:
        raise _fail(path, f"training parameter '{parameter_name}' has two lower bounds")
    if "maximum" in metadata and "exclusive_maximum" in metadata:
        raise _fail(path, f"training parameter '{parameter_name}' has two upper bounds")
    lower = metadata.get("minimum", metadata.get("exclusive_minimum"))
    upper = metadata.get("maximum", metadata.get("exclusive_maximum"))
    if lower is not None and upper is not None and (
        lower > upper
        or (
            lower == upper
            and ("exclusive_minimum" in metadata or "exclusive_maximum" in metadata)
        )
    ):
        raise _fail(path, f"training parameter '{parameter_name}' has inconsistent bounds")

    allowed = metadata.get("allowed_values")
    if allowed is not None:
        if not isinstance(allowed, list) or not allowed:
            raise _fail(
                path, f"training parameter '{parameter_name}' allowed_values must be a non-empty list"
            )
        if len({json.dumps(item, sort_keys=True) for item in allowed}) != len(allowed):
            raise _fail(path, f"training parameter '{parameter_name}' allowed_values has duplicates")
        probe = {**metadata, "nullable": metadata.get("nullable", False)}
        for item in allowed:
            probe["default"] = item
            _validate_default(parameter_name, probe, path)
        if metadata["default"] not in allowed:
            raise _fail(
                path, f"training parameter '{parameter_name}' default is not in allowed_values"
            )

    excluded = metadata.get("excluded_values")
    if excluded is not None:
        if not isinstance(excluded, list) or not excluded:
            raise _fail(
                path,
                f"training parameter '{parameter_name}' excluded_values must be a non-empty list",
            )
        if len({json.dumps(item, sort_keys=True) for item in excluded}) != len(excluded):
            raise _fail(path, f"training parameter '{parameter_name}' excluded_values has duplicates")
        probe = {**metadata, "nullable": metadata.get("nullable", False)}
        for item in excluded:
            probe["default"] = item
            _validate_default(parameter_name, probe, path)
        if metadata["default"] in excluded:
            raise _fail(
                path, f"training parameter '{parameter_name}' default is excluded"
            )

    candidates = [("default", metadata["default"])]
    if allowed is not None:
        candidates.extend(("allowed value", item) for item in allowed)
    for candidate_name, candidate in candidates:
        if candidate is None or not numeric:
            continue
        if "minimum" in metadata and candidate < metadata["minimum"]:
            raise _fail(path, f"training parameter '{parameter_name}' {candidate_name} is below minimum")
        if "exclusive_minimum" in metadata and candidate <= metadata["exclusive_minimum"]:
            raise _fail(path, f"training parameter '{parameter_name}' {candidate_name} is below exclusive_minimum")
        if "maximum" in metadata and candidate > metadata["maximum"]:
            raise _fail(path, f"training parameter '{parameter_name}' {candidate_name} is above maximum")
        if "exclusive_maximum" in metadata and candidate >= metadata["exclusive_maximum"]:
            raise _fail(path, f"training parameter '{parameter_name}' {candidate_name} is above exclusive_maximum")


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

    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, dict):
        raise _fail(path, "'capabilities' must be a mapping")
    training = capabilities.get("training")
    if not isinstance(training, dict) or not isinstance(training.get("enabled"), bool):
        raise _fail(path, "'capabilities.training.enabled' must be boolean")
    if training["enabled"]:
        protocol = training.get("protocol")
        profile = training.get("input_profile")
        if not isinstance(protocol, str) or not protocol:
            raise _fail(path, "enabled training capability needs a protocol")
        if not isinstance(profile, str) or not profile:
            raise _fail(path, "enabled training capability needs an input_profile")

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
        _validate_parameter_constraints(parameter_name, metadata, path)

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


def _safe_discovery_error(path: Path, exc: Exception, code: str) -> dict[str, str]:
    message = str(exc).replace(str(path), path.name).replace(str(path.parent), path.parent.name)
    return {
        "plugin": path.parent.name[:128],
        "code": code,
        "message": message.replace("\n", " ").replace("\r", " ")[:1000],
    }


def scan_detectors(detector_dir: str | Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Discover manifests independently, quarantining invalid and duplicate plugins."""
    root = _detector_root(detector_dir)
    root = root.resolve()
    if not root.is_dir():
        return {
            "detectors": [],
            "discovery_errors": [
                {
                    "plugin": "detectors",
                    "code": "directory_unavailable",
                    "message": "detector directory does not exist",
                }
            ],
        }

    candidates: list[tuple[Path, dict[str, Any]]] = []
    errors: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/manifest.yaml")):
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise _fail(path, "manifest resolves outside the detector directory")
            candidates.append((path, load_manifest(resolved)))
        except (ManifestValidationError, OSError, UnicodeError) as exc:
            errors.append(_safe_discovery_error(path, exc, "invalid_manifest"))

    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, manifest in candidates:
        grouped.setdefault(manifest["id"], []).append((path, manifest))

    detectors = []
    for detector_id, entries in grouped.items():
        if len(entries) > 1:
            for path, _manifest in entries:
                errors.append(
                    {
                        "plugin": path.parent.name[:128],
                        "code": "duplicate_detector_id",
                        "message": f"detector id {detector_id!r} is declared by multiple plugins",
                    }
                )
            continue
        detectors.append(entries[0][1])
    detectors.sort(key=lambda item: item["id"])
    errors.sort(key=lambda item: (item["plugin"], item["code"]))
    return {"detectors": detectors, "discovery_errors": errors}


def discover_detectors(detector_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Return validated manifests without importing their entry points."""
    root = _detector_root(detector_dir)
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


def get_manifest(
    detector_id: str, detector_dir: str | Path | None = None
) -> dict[str, Any]:
    report = scan_detectors(detector_dir)
    manifest = next(
        (item for item in report["detectors"] if item["id"] == detector_id), None
    )
    if manifest is None:
        # Preserve a targeted schema error when the requested ID belongs to one
        # invalid manifest. Duplicate IDs remain quarantined and unresolved.
        root = _detector_root(detector_dir).resolve()
        matching_paths = []
        for path in sorted(root.glob("*/manifest.yaml")) if root.is_dir() else ():
            try:
                if path.stat().st_size > MAX_MANIFEST_BYTES:
                    continue
                candidate = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, yaml.YAMLError):
                continue
            if isinstance(candidate, dict) and candidate.get("id") == detector_id:
                matching_paths.append(path)
        if len(matching_paths) == 1:
            return load_manifest(matching_paths[0])
        raise UnknownDetectorError(f"Unknown or invalid detector id: {detector_id!r}")
    return manifest


def validate_training_parameters(
    detector_id: str,
    submitted: Any,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if submitted is None:
        submitted = {}
    if not isinstance(submitted, dict):
        raise ParameterValidationError(
            f"parameters for detector {detector_id!r} must be an object"
        )
    definitions = manifest["training_parameters"]
    unknown = sorted(set(submitted).difference(definitions))
    if unknown:
        raise ParameterValidationError(
            f"unknown training parameter(s) for {detector_id!r}: {', '.join(unknown)}"
        )
    values = {name: metadata["default"] for name, metadata in definitions.items()}
    values.update(submitted)
    for name, value in values.items():
        metadata = definitions[name]
        if value is None and metadata.get("nullable", False):
            continue
        expected = metadata["type"]
        valid = {
            "boolean": isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "string": isinstance(value, str),
        }[expected]
        if not valid:
            raise ParameterValidationError(
                f"parameter {name!r} for {detector_id!r} must be {expected}"
            )
        if "minimum" in metadata and value < metadata["minimum"]:
            raise ParameterValidationError(f"parameter {name!r} is below its minimum")
        if "exclusive_minimum" in metadata and value <= metadata["exclusive_minimum"]:
            raise ParameterValidationError(f"parameter {name!r} must exceed its lower bound")
        if "maximum" in metadata and value > metadata["maximum"]:
            raise ParameterValidationError(f"parameter {name!r} exceeds its maximum")
        if "exclusive_maximum" in metadata and value >= metadata["exclusive_maximum"]:
            raise ParameterValidationError(f"parameter {name!r} must be below its upper bound")
        if "allowed_values" in metadata and value not in metadata["allowed_values"]:
            raise ParameterValidationError(
                f"parameter {name!r} must be one of {metadata['allowed_values']}"
            )
        if value in metadata.get("excluded_values", []):
            raise ParameterValidationError(f"parameter {name!r} uses an excluded value")
    return values


def get_adapter(
    detector_id: str,
    detector_dir: str | Path | None = None,
) -> DetectorAdapter:
    """Resolve one detector adapter from its manifest without eager plugin imports."""
    try:
        manifest = get_manifest(detector_id, detector_dir)
    except UnknownDetectorError as exc:
        raise UnknownDetectorError(f"Unknown detector id: {detector_id!r}") from exc

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


def get_training_adapter(
    detector_id: str,
    detector_dir: str | Path | None = None,
) -> TrainableDetectorAdapter:
    """Resolve a v1 training adapter only when a worker starts its child task."""
    manifest = get_manifest(detector_id, detector_dir)
    training = manifest["capabilities"]["training"]
    if not training.get("enabled"):
        raise UnsupportedTrainingProtocolError(
            f"Detector {detector_id!r} does not enable training"
        )
    protocol = training.get("protocol")
    if protocol not in SUPPORTED_TRAINING_PROTOCOLS:
        raise UnsupportedTrainingProtocolError(
            f"Detector {detector_id!r} uses unsupported training protocol {protocol!r}"
        )
    profile = training.get("input_profile")
    if profile not in SUPPORTED_INPUT_PROFILES:
        raise UnsupportedTrainingProtocolError(
            f"Detector {detector_id!r} uses unsupported input profile {profile!r}"
        )

    entry_point = manifest["entry_point"]
    module_name, attribute_name = entry_point.split(":", 1)
    try:
        module = import_module(module_name)
    except Exception as exc:
        raise AdapterImportError(
            f"Could not import training adapter module for detector {detector_id!r}: {exc}"
        ) from exc
    try:
        adapter_class = getattr(module, attribute_name)
    except AttributeError as exc:
        raise InvalidEntryPointError(
            f"Training adapter entry point for detector {detector_id!r} does not exist"
        ) from exc
    if not isinstance(adapter_class, type):
        raise InvalidEntryPointError(
            f"Training adapter entry point for detector {detector_id!r} must resolve to a class"
        )
    try:
        adapter = adapter_class()
    except Exception as exc:
        raise AdapterContractError(
            f"Training adapter for detector {detector_id!r} could not be instantiated: {exc}"
        ) from exc
    if not isinstance(adapter, TrainableDetectorAdapter):
        missing = [
            name
            for name in ("validate_training", "run_training")
            if not callable(getattr(adapter, name, None))
        ]
        raise AdapterContractError(
            f"Training adapter for detector {detector_id!r} does not satisfy "
            f"TrainableDetectorAdapter: {', '.join(missing) or 'invalid methods'}"
        )
    return adapter
