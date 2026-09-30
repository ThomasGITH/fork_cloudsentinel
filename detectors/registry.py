"""Discover and validate detector manifests without loading detector code."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib import util as importlib_util
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any

import yaml

from .contracts import DetectorAdapter, PluginReference, TrainableDetectorAdapter
from .plugin_integrity import (
    PluginPackageError,
    package_sha256,
    pinned_requirements_compatibility,
    sha256_file,
)


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
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_RUNTIME_PROFILE = "python-ml-cpu/v1"


@dataclass(frozen=True)
class PluginDescriptor:
    """Private location and integrity data for one discovered plugin release."""

    detector_id: str
    detector_version: str
    source_class: str
    manifest_path: Path
    package_root: Path
    package_sha256: str
    manifest_sha256: str
    entry_point: str
    runtime_profile: str
    manifest: dict[str, Any]
    runtime_status: dict[str, str]
    repository_root: Path | None = None

    def reference(self) -> PluginReference:
        return PluginReference(
            source="builtin" if self.source_class == "builtin" else "external",
            detector_id=self.detector_id,
            detector_version=self.detector_version,
            package_sha256=self.package_sha256,
            manifest_sha256=self.manifest_sha256,
            runtime_profile=self.runtime_profile,
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


def _configured_roots(
    detector_dir: str | Path | None,
) -> list[tuple[Path, str, bool]]:
    """Return (root, source class, strict availability) in precedence order."""
    if detector_dir is not None:
        return [(Path(detector_dir), "legacy", True)]
    legacy = os.getenv("DETECTOR_PLUGIN_ROOT")
    if legacy:
        return [(Path(legacy), "legacy", True)]

    roots: list[tuple[Path, str, bool]] = [(Path(__file__).parent, "builtin", True)]
    configured = os.getenv("DETECTOR_PLUGIN_ROOTS", "")
    seen = {Path(__file__).parent.resolve()}
    accepted: list[Path] = []
    for raw in configured.split(os.pathsep):
        if not raw.strip():
            continue
        candidate = Path(raw.strip())
        if not candidate.is_absolute():
            # Invalid additive roots are isolated and surfaced during scanning.
            roots.append((candidate, "external", False))
            continue
        resolved = candidate.resolve()
        if resolved in seen or any(
            resolved.is_relative_to(other) or other.is_relative_to(resolved)
            for other in accepted
        ):
            roots.append((candidate, "external_duplicate_root", False))
            continue
        seen.add(resolved)
        accepted.append(resolved)
        roots.append((candidate, "external", False))
    return roots


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
    message = re.sub(
        r"(?<![A-Za-z0-9_.-])/(?:[^\s/:]+/)+[^\s:]+",
        "[path redacted]",
        message,
    )
    return {
        "plugin": path.parent.name[:128],
        "code": code,
        "message": message.replace("\n", " ").replace("\r", " ")[:1000],
    }


def _runtime_status(package_root: Path, source_class: str, digest: str) -> dict[str, str]:
    manifest = load_manifest(package_root / "manifest.yaml")
    inference = manifest.get("capabilities", {}).get("inference") or manifest.get(
        "capabilities", {}
    ).get("prediction")
    if source_class in {"builtin", "legacy"}:
        return {
            "training_runtime_status": "ready",
            "inference_runtime_status": "ready" if inference else "not_supported",
        }
    receipt_path = package_root / "install_receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {
            "training_runtime_status": "unvalidated",
            "inference_runtime_status": "unvalidated" if inference else "not_supported",
        }
    try:
        requirements_compatible = bool(
            pinned_requirements_compatibility(package_root / "requirements.lock")[
                "compatible"
            ]
        )
    except PluginPackageError:
        requirements_compatible = False
    if (
        not isinstance(receipt, dict)
        or receipt.get("package_sha256") != digest
        or not requirements_compatible
    ):
        return {
            "training_runtime_status": "incompatible",
            "inference_runtime_status": "incompatible" if inference else "not_supported",
        }
    training_status = receipt.get("training_runtime_status", "unvalidated")
    inference_status = receipt.get(
        "inference_runtime_status", "unvalidated" if inference else "not_supported"
    )
    if training_status not in {"ready", "unvalidated", "incompatible", "import_failed"}:
        training_status = "incompatible"
    if inference_status not in {
        "ready",
        "unvalidated",
        "incompatible",
        "import_failed",
        "not_supported",
    }:
        inference_status = "incompatible"
    return {
        "training_runtime_status": training_status,
        "inference_runtime_status": inference_status,
    }


def _descriptor(path: Path, root: Path, source_class: str) -> PluginDescriptor:
    resolved_manifest = path.resolve()
    package_root = resolved_manifest.parent.resolve()
    allowed_root = root.resolve()
    repository_root: Path | None = None
    if source_class == "external":
        repository_root = allowed_root.parent.resolve()
        releases_root = (repository_root / "releases").resolve()
        if not package_root.is_relative_to(releases_root):
            raise _fail(path, "active plugin does not reference an immutable repository release")
    elif not package_root.is_relative_to(allowed_root):
        raise _fail(path, "manifest resolves outside the detector directory")

    manifest = load_manifest(resolved_manifest)
    if source_class == "external" and "." in manifest["entry_point"].split(":", 1)[0]:
        # Nested plugin-relative modules are allowed; installed/global package names are not
        # distinguishable syntactically. The loader always resolves them below package_root.
        pass
    digest = package_sha256(package_root)
    manifest_digest = sha256_file(resolved_manifest)
    runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}
    runtime_profile = runtime.get("profile", DEFAULT_RUNTIME_PROFILE)
    if not isinstance(runtime_profile, str) or not runtime_profile.strip():
        raise _fail(path, "runtime.profile must be a non-empty string")
    return PluginDescriptor(
        detector_id=manifest["id"],
        detector_version=manifest["version"],
        source_class=source_class,
        manifest_path=resolved_manifest,
        package_root=package_root,
        package_sha256=digest,
        manifest_sha256=manifest_digest,
        entry_point=manifest["entry_point"],
        runtime_profile=runtime_profile,
        manifest=manifest,
        runtime_status=_runtime_status(package_root, source_class, digest),
        repository_root=repository_root,
    )


def _scan_descriptors(
    detector_dir: str | Path | None = None,
) -> tuple[list[PluginDescriptor], list[dict[str, str]]]:
    candidates: list[PluginDescriptor] = []
    errors: list[dict[str, str]] = []
    for configured_root, source_class, strict in _configured_roots(detector_dir):
        if not configured_root.is_absolute() and detector_dir is None and source_class.startswith("external"):
            errors.append({
                "plugin": "external-root",
                "source": "external",
                "code": "invalid_plugin_root",
                "message": "external detector roots must be absolute paths",
            })
            continue
        root = configured_root.resolve()
        if source_class == "external_duplicate_root":
            errors.append({
                "plugin": "external-root",
                "source": "external",
                "code": "duplicate_plugin_root",
                "message": "duplicate or nested external detector root was ignored",
            })
            continue
        if not root.is_dir():
            errors.append({
                "plugin": "detectors" if strict else "external-root",
                "source": "builtin" if source_class == "builtin" else "external",
                "code": "directory_unavailable",
                "message": "detector directory does not exist",
            })
            continue
        effective_source = "external" if source_class == "external" else source_class
        for path in sorted(root.glob("*/manifest.yaml")):
            try:
                candidates.append(_descriptor(path, root, effective_source))
            except (ManifestValidationError, PluginPackageError, OSError, UnicodeError) as exc:
                error = _safe_discovery_error(path, exc, "invalid_manifest")
                error["source"] = "builtin" if source_class == "builtin" else "external"
                errors.append(error)

    builtins: dict[str, list[PluginDescriptor]] = {}
    externals: dict[str, list[PluginDescriptor]] = {}
    for item in candidates:
        target = builtins if item.source_class == "builtin" else externals
        target.setdefault(item.detector_id, []).append(item)

    selected: list[PluginDescriptor] = []
    for detector_id, entries in builtins.items():
        if len(entries) != 1:
            for item in entries:
                errors.append({
                    "plugin": item.package_root.name[:128],
                    "source": "builtin",
                    "code": "duplicate_detector_id",
                    "message": f"detector id {detector_id!r} is declared by multiple built-in plugins",
                })
            continue
        selected.append(entries[0])

    for detector_id, entries in externals.items():
        if detector_id in builtins:
            for item in entries:
                errors.append({
                    "plugin": item.package_root.name[:128],
                    "source": "external",
                    "code": "reserved_detector_id",
                    "message": f"external detector id {detector_id!r} conflicts with a built-in plugin",
                })
            continue
        if len(entries) != 1:
            for item in entries:
                errors.append({
                    "plugin": item.package_root.name[:128],
                    "source": "external",
                    "code": "duplicate_detector_id",
                    "message": f"detector id {detector_id!r} is declared by multiple external plugins",
                })
            continue
        selected.append(entries[0])

    selected.sort(key=lambda item: item.detector_id)
    errors.sort(key=lambda item: (item.get("plugin", ""), item.get("code", "")))
    return selected, errors


def _public_manifest(descriptor: PluginDescriptor) -> dict[str, Any]:
    public = {key: value for key, value in descriptor.manifest.items() if key != "entry_point"}
    public["source"] = "builtin" if descriptor.source_class == "builtin" else "external"
    public.update(descriptor.runtime_status)
    return public


def scan_detectors(detector_dir: str | Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Discover all configured roots without importing plugin adapter code."""
    descriptors, errors = _scan_descriptors(detector_dir)
    return {
        "detectors": [_public_manifest(item) for item in descriptors],
        "discovery_errors": errors,
    }


def discover_detectors(detector_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Return full validated manifests for compatibility with internal callers."""
    descriptors, errors = _scan_descriptors(detector_dir)
    if errors:
        duplicate = next(
            (item for item in errors if item["code"] == "duplicate_detector_id"), None
        )
        if duplicate is not None and detector_dir is not None:
            match = re.search(r"detector id '([^']+)'", duplicate["message"])
            detector_id = match.group(1) if match else "unknown"
            raise ManifestValidationError(f"duplicate detector id '{detector_id}'")
        if detector_dir is not None:
            raise ManifestValidationError(errors[0]["message"])
    return [dict(item.manifest) for item in descriptors]


def get_descriptor(
    detector_id: str,
    detector_dir: str | Path | None = None,
    plugin_reference: dict[str, Any] | PluginReference | None = None,
) -> PluginDescriptor:
    descriptors, _errors = _scan_descriptors(detector_dir)
    reference = (
        plugin_reference.to_dict()
        if isinstance(plugin_reference, PluginReference)
        else plugin_reference
    )
    if reference is None:
        match = next((item for item in descriptors if item.detector_id == detector_id), None)
        if match is not None:
            return match
    else:
        required = {
            "source",
            "detector_id",
            "detector_version",
            "package_sha256",
            "manifest_sha256",
            "runtime_profile",
        }
        if not isinstance(reference, dict) or not required.issubset(reference):
            raise ManifestValidationError("plugin reference is incomplete")
        if reference["detector_id"] != detector_id:
            raise ManifestValidationError("plugin reference detector identity mismatch")
        for field in ("package_sha256", "manifest_sha256"):
            if not isinstance(reference[field], str) or not SHA256_PATTERN.fullmatch(reference[field]):
                raise ManifestValidationError(f"plugin reference {field} is invalid")
        match = next(
            (
                item
                for item in descriptors
                if item.detector_id == detector_id
                and item.detector_version == reference["detector_version"]
                and item.package_sha256 == reference["package_sha256"]
                and item.manifest_sha256 == reference["manifest_sha256"]
                and item.runtime_profile == reference["runtime_profile"]
                and ("builtin" if item.source_class == "builtin" else "external")
                == reference["source"]
            ),
            None,
        )
        if match is not None:
            return match

        # Active may have changed. Exact external releases remain addressable by digest.
        if reference.get("source") == "external":
            for configured_root, source_class, _strict in _configured_roots(detector_dir):
                if source_class != "external" or not configured_root.is_absolute():
                    continue
                repository = configured_root.resolve().parent
                release = (
                    repository
                    / "releases"
                    / detector_id
                    / str(reference["detector_version"])
                    / f"sha256_{reference['package_sha256']}"
                )
                manifest_path = release / "manifest.yaml"
                if manifest_path.is_file():
                    candidate = _descriptor(manifest_path, configured_root.resolve(), "external")
                    if candidate.reference().to_dict() == reference:
                        return candidate
        raise UnknownDetectorError(
            f"Unknown or changed plugin release for detector id: {detector_id!r}"
        )

    # Preserve targeted errors for one invalid manifest in an explicit/legacy root.
    for configured_root, _source, _strict in _configured_roots(detector_dir):
        root = configured_root.resolve()
        for path in sorted(root.glob("*/manifest.yaml")) if root.is_dir() else ():
            try:
                candidate = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, yaml.YAMLError):
                continue
            if isinstance(candidate, dict) and candidate.get("id") == detector_id:
                load_manifest(path)
    raise UnknownDetectorError(f"Unknown or invalid detector id: {detector_id!r}")


def get_manifest(
    detector_id: str,
    detector_dir: str | Path | None = None,
    plugin_reference: dict[str, Any] | PluginReference | None = None,
) -> dict[str, Any]:
    return dict(get_descriptor(detector_id, detector_dir, plugin_reference).manifest)


def get_plugin_reference(
    detector_id: str, detector_dir: str | Path | None = None
) -> dict[str, str]:
    return get_descriptor(detector_id, detector_dir).reference().to_dict()


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


def _load_external_module(descriptor: PluginDescriptor, module_name: str) -> ModuleType:
    """Load one external module below an integrity-pinned private namespace."""
    if module_name.startswith(".") or any(
        part in {"", ".", ".."} for part in module_name.split(".")
    ):
        raise InvalidEntryPointError("external adapter module path is unsafe")
    # Revalidate after discovery so a modified release cannot be imported.
    try:
        actual_package = package_sha256(descriptor.package_root)
        actual_manifest = sha256_file(descriptor.manifest_path)
    except (PluginPackageError, OSError) as exc:
        raise AdapterImportError(
            f"Could not verify external adapter for detector {descriptor.detector_id!r}"
        ) from exc
    if actual_package != descriptor.package_sha256 or actual_manifest != descriptor.manifest_sha256:
        raise AdapterImportError(
            f"External plugin integrity check failed for detector {descriptor.detector_id!r}"
        )

    safe_id = re.sub(r"[^A-Za-z0-9_]", "_", descriptor.detector_id)
    package_name = (
        f"_cloudsentinel_external_plugins.{safe_id}_{descriptor.package_sha256}"
    )
    base_name = "_cloudsentinel_external_plugins"
    if base_name not in sys.modules:
        base = ModuleType(base_name)
        base.__path__ = []  # type: ignore[attr-defined]
        sys.modules[base_name] = base
    if package_name not in sys.modules:
        initializer = descriptor.package_root / "__init__.py"
        if initializer.is_file() and not initializer.is_symlink():
            specification = importlib_util.spec_from_file_location(
                package_name,
                initializer,
                submodule_search_locations=[str(descriptor.package_root)],
            )
            if specification is None or specification.loader is None:
                raise AdapterImportError("Could not construct external plugin package")
            package = importlib_util.module_from_spec(specification)
            sys.modules[package_name] = package
            previous_bytecode_setting = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            try:
                specification.loader.exec_module(package)
            finally:
                sys.dont_write_bytecode = previous_bytecode_setting
        else:
            package = ModuleType(package_name)
            package.__path__ = [str(descriptor.package_root)]  # type: ignore[attr-defined]
            package.__package__ = package_name
            sys.modules[package_name] = package

    relative = Path(*module_name.split("."))
    module_file = (descriptor.package_root / relative).with_suffix(".py")
    package_file = descriptor.package_root / relative / "__init__.py"
    if module_file.is_file() and not module_file.is_symlink():
        target = module_file.resolve()
        submodule_locations = None
    elif package_file.is_file() and not package_file.is_symlink():
        target = package_file.resolve()
        submodule_locations = [str(package_file.parent)]
    else:
        raise AdapterImportError(
            f"Could not import external adapter module for detector {descriptor.detector_id!r}"
        )
    if not target.is_relative_to(descriptor.package_root):
        raise AdapterImportError("External adapter module resolves outside its plugin release")
    qualified_name = f"{package_name}.{module_name}"
    if qualified_name in sys.modules:
        return sys.modules[qualified_name]
    specification = importlib_util.spec_from_file_location(
        qualified_name, target, submodule_search_locations=submodule_locations
    )
    if specification is None or specification.loader is None:
        raise AdapterImportError("Could not construct external adapter module")
    module = importlib_util.module_from_spec(specification)
    sys.modules[qualified_name] = module
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    except Exception as exc:
        for loaded_name in tuple(sys.modules):
            if loaded_name == qualified_name or loaded_name.startswith(qualified_name + "."):
                sys.modules.pop(loaded_name, None)
        raise AdapterImportError(
            f"Could not import external adapter module for detector {descriptor.detector_id!r}: {exc}"
        ) from exc
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    return module


def _adapter_class(
    detector_id: str,
    detector_dir: str | Path | None,
    plugin_reference: dict[str, Any] | PluginReference | None,
) -> tuple[type, PluginDescriptor]:
    descriptor = get_descriptor(detector_id, detector_dir, plugin_reference)
    entry_point = descriptor.entry_point
    if not ENTRY_POINT_PATTERN.fullmatch(entry_point):
        raise InvalidEntryPointError(
            f"Detector {detector_id!r} has an invalid adapter entry point"
        )
    module_name, attribute_name = entry_point.split(":", 1)
    try:
        module = (
            import_module(module_name)
            if descriptor.source_class in {"builtin", "legacy"}
            else _load_external_module(descriptor, module_name)
        )
    except AdapterImportError:
        raise
    except Exception as exc:
        raise AdapterImportError(
            f"Could not import adapter module for detector {detector_id!r}: {exc}"
        ) from exc
    try:
        adapter_class = getattr(module, attribute_name)
    except AttributeError as exc:
        raise InvalidEntryPointError(
            f"Adapter entry point for detector {detector_id!r} does not exist"
        ) from exc
    if not isinstance(adapter_class, type):
        raise InvalidEntryPointError(
            f"Adapter entry point for detector {detector_id!r} must resolve to a class"
        )
    return adapter_class, descriptor


def get_adapter(
    detector_id: str,
    detector_dir: str | Path | None = None,
    plugin_reference: dict[str, Any] | PluginReference | None = None,
) -> DetectorAdapter:
    """Resolve one detector adapter from its manifest without eager plugin imports."""
    try:
        adapter_class, _descriptor_value = _adapter_class(
            detector_id, detector_dir, plugin_reference
        )
    except UnknownDetectorError as exc:
        raise UnknownDetectorError(f"Unknown detector id: {detector_id!r}") from exc

    try:
        adapter = adapter_class()
    except Exception as exc:
        raise AdapterContractError(
            f"Adapter for detector {detector_id!r} could not be instantiated: {exc}"
        ) from exc

    if not isinstance(adapter, DetectorAdapter):
        missing_operations = [
            operation
            for operation in ("train", "evaluate", "predict")
            if not callable(getattr(adapter, operation, None))
        ]
        details = ", ".join(missing_operations) or "invalid operation definitions"
        raise AdapterContractError(
            f"Adapter for detector {detector_id!r} does not satisfy "
            f"DetectorAdapter: {details}"
        )

    return adapter


def get_training_adapter(
    detector_id: str,
    detector_dir: str | Path | None = None,
    plugin_reference: dict[str, Any] | PluginReference | None = None,
) -> TrainableDetectorAdapter:
    """Resolve a v1 training adapter only when a worker starts its child task."""
    descriptor = get_descriptor(detector_id, detector_dir, plugin_reference)
    manifest = descriptor.manifest
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

    if descriptor.source_class == "external" and descriptor.runtime_status.get(
        "training_runtime_status"
    ) != "ready":
        raise AdapterImportError(
            f"External detector {detector_id!r} is not ready in this training runtime"
        )

    adapter_class, _descriptor_value = _adapter_class(
        detector_id, detector_dir, plugin_reference
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
