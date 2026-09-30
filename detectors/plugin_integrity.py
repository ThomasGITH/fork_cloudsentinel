"""Deterministic integrity helpers for trusted external detector packages."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
import re
import stat
import sys


MAX_PACKAGE_FILES = 200
MAX_PACKAGE_BYTES = 25 * 1024 * 1024
MAX_PACKAGE_FILE_BYTES = 5 * 1024 * 1024
IGNORED_RECEIPT_FILES = {"install_receipt.json"}
FORBIDDEN_PARTS = {
    ".git",
    ".hg",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "env",
    "node_modules",
    "venv",
}
PINNED_REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)$")


class PluginPackageError(ValueError):
    """Raised when a plugin package is unsafe or cannot be verified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_files(root: str | Path, *, strict: bool = True) -> list[Path]:
    """Return safe regular package files in deterministic relative-path order."""
    package_root = Path(root).resolve()
    if not package_root.is_dir() or package_root.is_symlink():
        raise PluginPackageError("plugin source must be a regular directory")

    files: list[Path] = []
    total_size = 0
    for current_root, directory_names, file_names in os.walk(
        package_root, topdown=True, followlinks=False
    ):
        current = Path(current_root)
        for name in list(directory_names):
            candidate = current / name
            relative = candidate.relative_to(package_root)
            if name in FORBIDDEN_PARTS:
                if strict:
                    raise PluginPackageError(
                        f"plugin package contains forbidden directory {relative.as_posix()!r}"
                    )
                directory_names.remove(name)
                continue
            if candidate.is_symlink():
                raise PluginPackageError("plugin package must not contain symlinks")

        for name in file_names:
            candidate = current / name
            relative = candidate.relative_to(package_root)
            if name in IGNORED_RECEIPT_FILES:
                continue
            if name.endswith((".pyc", ".pyo")) or any(
                part in FORBIDDEN_PARTS for part in relative.parts
            ):
                if strict:
                    raise PluginPackageError(
                        f"plugin package contains forbidden file {relative.as_posix()!r}"
                    )
                continue
            try:
                metadata = candidate.lstat()
            except OSError as exc:
                raise PluginPackageError("plugin package cannot be inspected") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise PluginPackageError("plugin package must not contain symlinks")
            if not stat.S_ISREG(metadata.st_mode):
                raise PluginPackageError(
                    "plugin package may contain regular files only"
                )
            resolved = candidate.resolve()
            if not resolved.is_relative_to(package_root):
                raise PluginPackageError("plugin package path escapes its root")
            if metadata.st_size > MAX_PACKAGE_FILE_BYTES:
                raise PluginPackageError("plugin package contains an oversized file")
            total_size += metadata.st_size
            if total_size > MAX_PACKAGE_BYTES:
                raise PluginPackageError("plugin package exceeds its total size limit")
            files.append(candidate)
            if len(files) > MAX_PACKAGE_FILES:
                raise PluginPackageError("plugin package contains too many files")

    files.sort(key=lambda item: item.relative_to(package_root).as_posix())
    return files


def package_sha256(root: str | Path) -> str:
    """Hash relative names, sizes, and bytes while excluding installer receipts."""
    package_root = Path(root).resolve()
    digest = hashlib.sha256()
    for path in package_files(package_root, strict=False):
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        size = path.stat().st_size
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def runtime_fingerprint() -> str:
    """Identify the Python runtime and installed distributions without imports."""
    distributions = []
    for distribution in importlib_metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            normalized = name.lower().replace("_", "-").replace(".", "-")
            distributions.append(f"{normalized}=={distribution.version}")
    payload = "\n".join(
        [f"python=={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"]
        + sorted(set(distributions))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pinned_requirements_compatibility(path: str | Path) -> dict[str, object]:
    """Compare a simple pinned lockfile with installed distributions."""
    lock_path = Path(path)
    if not lock_path.is_file() or lock_path.is_symlink():
        raise PluginPackageError("external plugin requires requirements.lock")
    try:
        lines = lock_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PluginPackageError("requirements.lock cannot be read") from exc
    required: list[tuple[str, str]] = []
    for number, line in enumerate(lines, 1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        match = PINNED_REQUIREMENT.fullmatch(value)
        if not match:
            raise PluginPackageError(
                f"requirements.lock line {number} must use name==version"
            )
        required.append((match.group(1), match.group(2)))

    installed: dict[str, str] = {}
    for distribution in importlib_metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed[re.sub(r"[-_.]+", "-", name).lower()] = distribution.version
    incompatible = sorted(
        name
        for name, expected in required
        if installed.get(re.sub(r"[-_.]+", "-", name).lower()) != expected
    )
    return {
        "compatible": not incompatible,
        "requirement_count": len(required),
        "incompatible": incompatible,
    }
