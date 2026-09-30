import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import yaml

from detectors import registry
from detectors.plugin_integrity import PluginPackageError, package_sha256, sha256_file
from detectors.plugin_repository import PluginRepository, PluginRepositoryError, main


class ExternalPluginRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository_root = self.root / "repository"
        self.repository = PluginRepository(self.repository_root)
        self.source_root = self.root / "sources"
        self.source_root.mkdir()

    def tearDown(self):
        for name in tuple(__import__("sys").modules):
            if name.startswith("_cloudsentinel_external_plugins"):
                __import__("sys").modules.pop(name, None)
        for current_root, directory_names, file_names in os.walk(
            self.root, topdown=False, followlinks=False
        ):
            for name in file_names:
                path = Path(current_root) / name
                if not path.is_symlink():
                    path.chmod(0o600)
            for name in directory_names:
                path = Path(current_root) / name
                if not path.is_symlink():
                    path.chmod(0o700)
        self.temporary.cleanup()

    def write_plugin(
        self,
        folder: str,
        *,
        detector_id: str = "external-density",
        version: str = "1.0.0",
        marker: str = "v1",
        requirement: str = "scikit-learn==1.5.0",
        adapter_import: str = (
            "from sklearn.neighbors import LocalOutlierFactor\n"
            "from .implementation import MARKER"
        ),
    ) -> Path:
        package = self.source_root / folder
        package.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "id": detector_id,
            "name": "External density detector",
            "version": version,
            "description": "External repository fixture.",
            "supported_modalities": ["metrics"],
            "runtime": {"profile": "python-ml-cpu/v1"},
            "capabilities": {
                "training": {
                    "enabled": True,
                    "protocol": "cloudsentinel.training/v1",
                    "input_profile": "metrics-partition-v1",
                }
            },
            "input_requirements": {"training": {"labels_required": True}},
            "training_parameters": {
                "neighbors": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "description": "Fixture parameter.",
                }
            },
            "entry_point": "adapter:ExternalAdapter",
        }
        (package / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )
        (package / "implementation.py").write_text(
            f"MARKER = {marker!r}\n", encoding="utf-8"
        )
        (package / "adapter.py").write_text(
            adapter_import
            + "\n"
            + "class ExternalAdapter:\n"
            + "    marker = MARKER\n"
            + "    estimator_type = LocalOutlierFactor if 'LocalOutlierFactor' in globals() else None\n"
            + "    def validate_training(self, context):\n        return None\n"
            + "    def run_training(self, context, progress):\n        return None\n",
            encoding="utf-8",
        )
        (package / "requirements.lock").write_text(requirement + "\n", encoding="utf-8")
        return package

    def configured_environment(self):
        return patch.dict(
            os.environ,
            {"DETECTOR_PLUGIN_ROOTS": str(self.repository.active)},
            clear=False,
        )

    def test_builtin_and_external_discovery_are_combined_and_lazy(self):
        source = self.write_plugin("external")
        release = self.repository.install(source)
        self.repository.activate(release["detector_id"], release["detector_version"])
        with self.configured_environment():
            before = set(__import__("sys").modules)
            report = registry.scan_detectors()
            loaded = set(__import__("sys").modules) - before
        items = {item["id"]: item for item in report["detectors"]}
        self.assertEqual(items["cgnn"]["source"], "builtin")
        self.assertEqual(items["isolation-forest"]["source"], "builtin")
        self.assertEqual(items["external-density"]["source"], "external")
        self.assertEqual(items["external-density"]["training_runtime_status"], "ready")
        self.assertNotIn("entry_point", items["external-density"])
        self.assertNotIn("package_sha256", items["external-density"])
        self.assertFalse(any(name.startswith("_cloudsentinel_external_plugins") for name in loaded))

    def test_missing_and_invalid_external_roots_do_not_hide_builtins(self):
        missing = self.root / "missing" / "active"
        with patch.dict(os.environ, {"DETECTOR_PLUGIN_ROOTS": str(missing)}):
            report = registry.scan_detectors()
        self.assertEqual([item["id"] for item in report["detectors"]], ["cgnn", "isolation-forest"])
        self.assertIn("directory_unavailable", {item["code"] for item in report["discovery_errors"]})

        invalid_repository = self.root / "invalid-repository"
        active = invalid_repository / "active"
        release = invalid_repository / "releases" / "broken" / "1.0.0" / "sha256_bad"
        release.mkdir(parents=True)
        (release / "manifest.yaml").write_text("id: [broken", encoding="utf-8")
        active.mkdir(parents=True)
        (active / "broken").symlink_to(os.path.relpath(release, active), target_is_directory=True)
        with patch.dict(os.environ, {"DETECTOR_PLUGIN_ROOTS": str(active)}):
            report = registry.scan_detectors()
        self.assertEqual([item["id"] for item in report["detectors"]], ["cgnn", "isolation-forest"])
        self.assertIn("invalid_manifest", {item["code"] for item in report["discovery_errors"]})

    def test_builtin_id_is_reserved_and_duplicate_externals_are_quarantined(self):
        with self.assertRaisesRegex(PluginRepositoryError, "reserved"):
            self.repository.install(self.write_plugin("fake-cgnn", detector_id="cgnn"))

        duplicate_repository = self.root / "duplicates"
        active = duplicate_repository / "active"
        active.mkdir(parents=True)
        for folder in ("one", "two"):
            package = self.write_plugin(folder, detector_id="duplicate-external")
            digest = package_sha256(package)
            release = duplicate_repository / "releases" / folder / "1.0.0" / f"sha256_{digest}"
            release.parent.mkdir(parents=True)
            __import__("shutil").copytree(package, release)
            (active / folder).symlink_to(os.path.relpath(release, active), target_is_directory=True)
        with patch.dict(os.environ, {"DETECTOR_PLUGIN_ROOTS": str(active)}):
            report = registry.scan_detectors()
        self.assertNotIn("duplicate-external", [item["id"] for item in report["detectors"]])
        duplicate_errors = [item for item in report["discovery_errors"] if item["code"] == "duplicate_detector_id"]
        self.assertEqual(len(duplicate_errors), 2)

    def test_relative_import_and_version_pinning_survive_upgrade_and_rollback(self):
        version_one = self.repository.install(self.write_plugin("v1", marker="v1"))
        self.repository.activate("external-density", "1.0.0")
        with self.configured_environment():
            reference_one = registry.get_plugin_reference("external-density")
            adapter_one = registry.get_training_adapter(
                "external-density", plugin_reference=reference_one
            )
            self.assertEqual(adapter_one.marker, "v1")

        version_two = self.repository.install(
            self.write_plugin("v2", version="2.0.0", marker="v2")
        )
        self.repository.activate("external-density", "2.0.0")
        with self.configured_environment():
            active_reference = registry.get_plugin_reference("external-density")
            self.assertNotEqual(reference_one["package_sha256"], active_reference["package_sha256"])
            adapter_two = registry.get_training_adapter(
                "external-density", plugin_reference=active_reference
            )
            queued_adapter = registry.get_training_adapter(
                "external-density", plugin_reference=reference_one
            )
        self.assertEqual(adapter_two.marker, "v2")
        self.assertEqual(queued_adapter.marker, "v1")
        self.assertNotEqual(adapter_one.__class__.__module__, adapter_two.__class__.__module__)

        rolled_back = self.repository.rollback("external-density", "1.0.0")
        self.assertTrue(rolled_back["active"])
        with self.configured_environment():
            self.assertEqual(
                registry.get_plugin_reference("external-density")["package_sha256"],
                reference_one["package_sha256"],
            )

    def test_integrity_mismatch_is_rejected_before_import(self):
        release = self.repository.install(self.write_plugin("integrity"))
        self.repository.activate("external-density", "1.0.0")
        with self.configured_environment():
            reference = registry.get_plugin_reference("external-density")
        active_target = (self.repository.active / "external-density").resolve()
        (active_target / "implementation.py").chmod(0o644)
        (active_target / "implementation.py").write_text("MARKER = 'tampered'\n", encoding="utf-8")
        with self.configured_environment():
            with self.assertRaises((registry.UnknownDetectorError, registry.AdapterImportError)):
                registry.get_training_adapter("external-density", plugin_reference=reference)

    def test_adapter_import_failure_isolated_from_discovery_and_builtins(self):
        source = self.write_plugin(
            "import-failure",
            adapter_import=(
                "import os\n"
                "if os.getenv('BREAK_EXTERNAL_PLUGIN_IMPORT'):\n"
                "    raise RuntimeError('controlled import failure')\n"
                "from .implementation import MARKER"
            ),
        )
        self.repository.install(source)
        self.repository.activate("external-density", "1.0.0")
        for name in tuple(__import__("sys").modules):
            if name.startswith("_cloudsentinel_external_plugins"):
                __import__("sys").modules.pop(name, None)
        with self.configured_environment(), patch.dict(
            os.environ, {"BREAK_EXTERNAL_PLUGIN_IMPORT": "1"}
        ):
            report = registry.scan_detectors()
            self.assertIn("external-density", [item["id"] for item in report["detectors"]])
            self.assertIn("cgnn", [item["id"] for item in report["detectors"]])
            with self.assertRaises(registry.AdapterImportError):
                registry.get_training_adapter("external-density")

    def test_incompatible_requirements_and_unsafe_packages_are_rejected_atomically(self):
        incompatible = self.write_plugin(
            "incompatible", requirement="cloudsentinel-package-that-does-not-exist==99.0"
        )
        with self.assertRaisesRegex(PluginRepositoryError, "incompatible"):
            self.repository.install(incompatible)
        self.assertFalse((self.repository.active / "external-density").exists())

        symlinked = self.write_plugin("symlinked", detector_id="symlinked-detector")
        (symlinked / "escape.py").symlink_to(symlinked / "adapter.py")
        with self.assertRaisesRegex(PluginPackageError, "symlinks"):
            self.repository.install(symlinked)

        fifo = self.write_plugin("fifo", detector_id="fifo-detector")
        os.mkfifo(fifo / "pipe")
        with self.assertRaisesRegex(PluginPackageError, "regular files"):
            self.repository.install(fifo)

        traversal = self.write_plugin("traversal", detector_id="traversal-detector")
        manifest = yaml.safe_load((traversal / "manifest.yaml").read_text(encoding="utf-8"))
        manifest["entry_point"] = "../adapter:ExternalAdapter"
        (traversal / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
        with self.assertRaises(registry.ManifestValidationError):
            self.repository.install(traversal)

    def test_failed_install_preserves_active_release_and_cli_output_is_safe(self):
        release = self.repository.install(self.write_plugin("stable"))
        self.repository.activate("external-density", "1.0.0")
        active_before = (self.repository.active / "external-density").resolve()
        broken = self.write_plugin(
            "broken",
            version="2.0.0",
            adapter_import="from .missing_module import MARKER",
        )
        with self.assertRaises(PluginRepositoryError):
            self.repository.install(broken)
        self.assertEqual((self.repository.active / "external-density").resolve(), active_before)

        output = io.StringIO()
        with patch("sys.stdout", output):
            code = main([
                "--repository-root",
                str(self.repository_root),
                "list",
            ])
        self.assertEqual(code, 0)
        rendered = output.getvalue()
        self.assertNotIn(str(self.repository_root), rendered)
        self.assertNotIn("entry_point", rendered)
        self.assertIn(release["detector_id"], rendered)

    def test_kubernetes_runtime_mounts_are_read_only_and_admin_is_read_write(self):
        repository_root = Path(__file__).resolve().parents[1]
        runtime_images = []
        for manifest_name in (
            "learning_adaptation-deployment.yml",
            "learning_adaptation-celery-deployment.yml",
        ):
            value = yaml.safe_load_all(
                (repository_root / "k8s" / manifest_name).read_text(encoding="utf-8")
            )
            deployment = next(item for item in value if item.get("kind") == "Deployment")
            container = deployment["spec"]["template"]["spec"]["containers"][0]
            runtime_images.append(container["image"])
            mount = next(item for item in container["volumeMounts"] if item["name"] == "detector-plugins")
            self.assertEqual(mount["mountPath"], "/opt/cloudsentinel/detectors")
            self.assertTrue(mount["readOnly"])
            environment = {item["name"]: item["value"] for item in container["env"]}
            self.assertEqual(environment["DETECTOR_PLUGIN_ROOTS"], "/opt/cloudsentinel/detectors/active")

        admin = yaml.safe_load(
            (repository_root / "k8s" / "plugin-admin-pod.yml").read_text(encoding="utf-8")
        )
        mount = admin["spec"]["containers"][0]["volumeMounts"][0]
        self.assertFalse(mount["readOnly"])
        self.assertEqual(admin["spec"]["restartPolicy"], "Never")
        self.assertEqual(len(set(runtime_images)), 1)
        self.assertEqual(admin["spec"]["containers"][0]["image"], runtime_images[0])


if __name__ == "__main__":
    unittest.main()
