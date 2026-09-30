"""Operator-only installer for the file-backed external detector repository."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

from .contracts import InferenceDetectorAdapter, TrainableDetectorAdapter
from .plugin_integrity import (
    PluginPackageError,
    package_files,
    package_sha256,
    pinned_requirements_compatibility,
    runtime_fingerprint,
    sha256_file,
)
from . import registry


SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class PluginRepositoryError(ValueError):
    """Raised when an operator repository action cannot be completed safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def validate_requirements_lock(path: Path) -> dict[str, Any]:
    try:
        return dict(pinned_requirements_compatibility(path))
    except PluginPackageError as exc:
        raise PluginRepositoryError(str(exc)) from exc


class PluginRepository:
    def __init__(self, root: str | Path):
        root_path = Path(root)
        if not root_path.is_absolute():
            raise PluginRepositoryError("plugin repository root must be absolute")
        self.root = root_path.resolve()
        self.staging = self.root / "staging"
        self.releases = self.root / "releases"
        self.active = self.root / "active"
        self.index = self.root / "repository.json"

    def initialize(self) -> None:
        for directory in (self.staging, self.releases, self.active):
            directory.mkdir(parents=True, exist_ok=True)

    def validate_source(self, source: str | Path) -> dict[str, Any]:
        source_root = Path(source).resolve()
        package_files(source_root)
        manifest_path = source_root / "manifest.yaml"
        manifest = registry.load_manifest(manifest_path)
        if not SAFE_VERSION.fullmatch(manifest["version"]):
            raise PluginRepositoryError("plugin version is unsafe for repository storage")
        module_name = manifest["entry_point"].split(":", 1)[0]
        module_path = Path(*module_name.split("."))
        module_file = (source_root / module_path).with_suffix(".py")
        module_package = source_root / module_path / "__init__.py"
        if not (
            module_file.is_file()
            and not module_file.is_symlink()
            or module_package.is_file()
            and not module_package.is_symlink()
        ):
            raise PluginRepositoryError(
                "external entry_point must resolve to a module inside the plugin package"
            )
        requirements = validate_requirements_lock(source_root / "requirements.lock")
        if not requirements["compatible"]:
            names = ", ".join(requirements["incompatible"])
            raise PluginRepositoryError(
                f"plugin requirements are unavailable or incompatible: {names}"
            )
        return {
            "manifest": manifest,
            "package_sha256": package_sha256(source_root),
            "manifest_sha256": sha256_file(manifest_path),
            "requirements": requirements,
        }

    def _validate_adapter_contract(
        self, package_root: Path, validation: dict[str, Any]
    ) -> None:
        manifest = validation["manifest"]
        runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}
        descriptor = registry.PluginDescriptor(
            detector_id=manifest["id"],
            detector_version=manifest["version"],
            source_class="external",
            manifest_path=package_root / "manifest.yaml",
            package_root=package_root,
            package_sha256=validation["package_sha256"],
            manifest_sha256=validation["manifest_sha256"],
            entry_point=manifest["entry_point"],
            runtime_profile=runtime.get("profile", registry.DEFAULT_RUNTIME_PROFILE),
            manifest=manifest,
            runtime_status={
                "training_runtime_status": "ready",
                "inference_runtime_status": "not_supported",
            },
            repository_root=self.root,
        )
        module_name, attribute_name = descriptor.entry_point.split(":", 1)
        try:
            module = registry._load_external_module(descriptor, module_name)
        except (
            registry.AdapterImportError,
            registry.InvalidEntryPointError,
            registry.AdapterContractError,
        ) as exc:
            raise PluginRepositoryError(
                "external adapter failed controlled runtime validation"
            ) from exc
        adapter_class = getattr(module, attribute_name, None)
        if not isinstance(adapter_class, type):
            raise PluginRepositoryError("external adapter entry point is not a class")
        try:
            adapter = adapter_class()
        except Exception as exc:
            raise PluginRepositoryError("external adapter could not be instantiated") from exc
        training = manifest.get("capabilities", {}).get("training", {})
        if training.get("enabled") and not isinstance(adapter, TrainableDetectorAdapter):
            raise PluginRepositoryError(
                "external adapter does not satisfy TrainableDetectorAdapter"
            )
        inference = manifest.get("capabilities", {}).get("inference", {})
        if inference.get("enabled") and not isinstance(adapter, InferenceDetectorAdapter):
            raise PluginRepositoryError(
                "external adapter does not satisfy InferenceDetectorAdapter"
            )

    @staticmethod
    def _make_release_read_only(package_root: Path) -> None:
        for path in package_files(package_root):
            path.chmod(0o444)
        (package_root / "install_receipt.json").chmod(0o444)
        directories = [
            item for item in package_root.rglob("*") if item.is_dir() and not item.is_symlink()
        ]
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            directory.chmod(0o555)
        package_root.chmod(0o555)

    def install(self, source: str | Path) -> dict[str, Any]:
        self.initialize()
        source_root = Path(source).resolve()
        validation = self.validate_source(source_root)
        manifest = validation["manifest"]
        detector_id = manifest["id"]
        builtins, _errors = registry._scan_descriptors(Path(registry.__file__).parent)
        if any(item.detector_id == detector_id for item in builtins):
            raise PluginRepositoryError(
                f"detector id {detector_id!r} is reserved by a built-in plugin"
            )

        stage = Path(tempfile.mkdtemp(prefix="install_", dir=self.staging))
        staged_package = stage / "package"
        try:
            shutil.copytree(source_root, staged_package)
            staged_validation = self.validate_source(staged_package)
            if (
                staged_validation["package_sha256"] != validation["package_sha256"]
                or staged_validation["manifest_sha256"] != validation["manifest_sha256"]
            ):
                raise PluginRepositoryError("plugin changed while it was staged")
            self._validate_adapter_contract(staged_package, staged_validation)
            receipt = {
                "schema_version": 1,
                "detector_id": detector_id,
                "detector_version": manifest["version"],
                "package_sha256": validation["package_sha256"],
                "manifest_sha256": validation["manifest_sha256"],
                "runtime_profile": (
                    manifest.get("runtime", {}).get("profile", registry.DEFAULT_RUNTIME_PROFILE)
                    if isinstance(manifest.get("runtime", {}), dict)
                    else registry.DEFAULT_RUNTIME_PROFILE
                ),
                "runtime_fingerprint": runtime_fingerprint(),
                "training_runtime_status": "ready",
                "inference_runtime_status": (
                    "ready"
                    if manifest.get("capabilities", {}).get("inference", {}).get("enabled")
                    else "not_supported"
                ),
                "installed_at": _utc_now(),
                "requirements_compatible": True,
            }
            _atomic_json(staged_package / "install_receipt.json", receipt)
            release = (
                self.releases
                / detector_id
                / manifest["version"]
                / f"sha256_{validation['package_sha256']}"
            )
            release.parent.mkdir(parents=True, exist_ok=True)
            if release.exists():
                if package_sha256(release) != validation["package_sha256"]:
                    raise PluginRepositoryError("existing immutable release failed integrity validation")
                return self._safe_release_summary(release)
            os.replace(staged_package, release)
            self._make_release_read_only(release)
            return self._safe_release_summary(release)
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def _release_candidates(self, detector_id: str, selector: str) -> list[Path]:
        if not registry.ID_PATTERN.fullmatch(detector_id):
            raise PluginRepositoryError("detector id is invalid")
        detector_root = (self.releases / detector_id).resolve()
        if not detector_root.is_relative_to(self.releases.resolve()):
            raise PluginRepositoryError("detector id resolves outside repository")
        candidates: list[Path] = []
        normalized_digest = selector.removeprefix("sha256_")
        if registry.SHA256_PATTERN.fullmatch(normalized_digest):
            candidates = list(detector_root.glob(f"*/sha256_{normalized_digest}"))
        elif SAFE_VERSION.fullmatch(selector):
            candidates = list((detector_root / selector).glob("sha256_*"))
        else:
            raise PluginRepositoryError("release selector must be a safe version or SHA-256")
        return [item.resolve() for item in candidates if item.is_dir()]

    def activate(self, detector_id: str, selector: str) -> dict[str, Any]:
        self.initialize()
        candidates = self._release_candidates(detector_id, selector)
        if len(candidates) != 1:
            raise PluginRepositoryError("release selector must identify exactly one installed release")
        release = candidates[0]
        if not release.is_relative_to(self.releases.resolve()):
            raise PluginRepositoryError("release resolves outside repository")
        receipt = json.loads((release / "install_receipt.json").read_text(encoding="utf-8"))
        if receipt.get("training_runtime_status") != "ready":
            raise PluginRepositoryError("release is not ready for the training runtime")
        if package_sha256(release) != receipt.get("package_sha256"):
            raise PluginRepositoryError("release package checksum mismatch")
        if sha256_file(release / "manifest.yaml") != receipt.get("manifest_sha256"):
            raise PluginRepositoryError("release manifest checksum mismatch")

        link = self.active / detector_id
        temporary = self.active / f".{detector_id}.{os.getpid()}.tmp"
        try:
            temporary.unlink(missing_ok=True)
            target = os.path.relpath(release, self.active)
            temporary.symlink_to(target, target_is_directory=True)
            if not temporary.resolve().is_relative_to(self.releases.resolve()):
                raise PluginRepositoryError("active reference resolves outside releases")
            os.replace(temporary, link)
        finally:
            temporary.unlink(missing_ok=True)
        self._write_index()
        return self._safe_release_summary(release, active=True)

    def rollback(self, detector_id: str, selector: str) -> dict[str, Any]:
        return self.activate(detector_id, selector)

    def _safe_release_summary(self, release: Path, active: bool = False) -> dict[str, Any]:
        receipt = json.loads((release / "install_receipt.json").read_text(encoding="utf-8"))
        return {
            "detector_id": receipt["detector_id"],
            "detector_version": receipt["detector_version"],
            "package_sha256": receipt["package_sha256"],
            "training_runtime_status": receipt["training_runtime_status"],
            "inference_runtime_status": receipt["inference_runtime_status"],
            "active": active,
        }

    def list(self) -> list[dict[str, Any]]:
        self.initialize()
        active_targets = {
            item.name: item.resolve()
            for item in self.active.iterdir()
            if item.is_symlink() and item.resolve().is_relative_to(self.releases.resolve())
        }
        releases = []
        for receipt_path in sorted(self.releases.glob("*/*/sha256_*/install_receipt.json")):
            release = receipt_path.parent.resolve()
            summary = self._safe_release_summary(
                release,
                active=active_targets.get(receipt_path.parents[2].name) == release,
            )
            releases.append(summary)
        return releases

    def _write_index(self) -> None:
        _atomic_json(
            self.index,
            {"schema_version": 1, "updated_at": _utc_now(), "releases": self.list()},
        )


def _repository_from_args(args: argparse.Namespace) -> PluginRepository:
    configured = args.repository_root or os.getenv(
        "DETECTOR_PLUGIN_REPOSITORY_ROOT", "/opt/cloudsentinel/detectors"
    )
    return PluginRepository(configured)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cloudsentinel-plugin")
    parser.add_argument("--repository-root")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "install"):
        command = commands.add_parser(name)
        command.add_argument("source_dir")
    for name in ("activate", "rollback"):
        command = commands.add_parser(name)
        command.add_argument("detector_id")
        command.add_argument("selector")
    commands.add_parser("list")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = _repository_from_args(args)
    try:
        if args.command == "validate":
            result = repository.validate_source(args.source_dir)
            repository._validate_adapter_contract(Path(args.source_dir).resolve(), result)
            output: Any = {
                "status": "valid",
                "detector_id": result["manifest"]["id"],
                "detector_version": result["manifest"]["version"],
                "training_runtime_status": "ready",
            }
        elif args.command == "install":
            output = repository.install(args.source_dir)
        elif args.command == "activate":
            output = repository.activate(args.detector_id, args.selector)
        elif args.command == "rollback":
            output = repository.rollback(args.detector_id, args.selector)
        else:
            output = repository.list()
    except (PluginRepositoryError, PluginPackageError, registry.ManifestValidationError) as exc:
        message = re.sub(
            r"(?<![A-Za-z0-9_.-])/(?:[^\s/:]+/)+[^\s:]+",
            "[path redacted]",
            str(exc).replace("\n", " ").replace("\r", " "),
        )[:1000]
        print(json.dumps({"status": "error", "message": message}), file=sys.stderr)
        return 1
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
