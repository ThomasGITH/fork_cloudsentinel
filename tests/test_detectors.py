import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

from detectors.api import create_detectors_blueprint
from detectors.registry import ManifestValidationError, discover_detectors, load_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CGNN_MANIFEST = REPOSITORY_ROOT / "detectors" / "cgnn" / "manifest.yaml"


class DetectorRegistryTests(unittest.TestCase):
    def setUp(self):
        self.valid_manifest = load_manifest(CGNN_MANIFEST)

    def write_manifest(self, root: Path, folder: str, manifest) -> Path:
        plugin_dir = root / folder
        plugin_dir.mkdir()
        path = plugin_dir / "manifest.yaml"
        path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        return path

    def test_discovers_current_cgnn_manifest(self):
        manifests = discover_detectors(REPOSITORY_ROOT / "detectors")
        self.assertEqual(
            [manifest["id"] for manifest in manifests],
            ["cgnn", "isolation-forest"],
        )
        self.assertEqual(manifests[0]["schema_version"], 1)
        self.assertEqual(manifests[0]["entry_point"], "detectors.cgnn.adapter:CGNNAdapter")

    def test_manifest_training_defaults_match_current_configuration(self):
        config_path = REPOSITORY_ROOT / "learning_adaptation" / "config.json"
        current_config = json.loads(config_path.read_text(encoding="utf-8"))
        for name, metadata in self.valid_manifest["training_parameters"].items():
            self.assertIn(name, current_config)
            self.assertEqual(metadata["default"], current_config[name], name)

    def test_missing_required_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = copy.deepcopy(self.valid_manifest)
            del invalid["entry_point"]
            path = self.write_manifest(Path(directory), "invalid", invalid)
            with self.assertRaisesRegex(ManifestValidationError, "missing required fields: entry_point"):
                load_manifest(path)

    def test_unsupported_schema_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = copy.deepcopy(self.valid_manifest)
            invalid["schema_version"] = 2
            path = self.write_manifest(Path(directory), "invalid", invalid)
            with self.assertRaisesRegex(ManifestValidationError, "unsupported schema_version"):
                load_manifest(path)

    def test_malformed_yaml_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin_dir = Path(directory) / "invalid"
            plugin_dir.mkdir()
            path = plugin_dir / "manifest.yaml"
            path.write_text("id: [unterminated", encoding="utf-8")
            with self.assertRaisesRegex(ManifestValidationError, "cannot read YAML"):
                load_manifest(path)

    def test_invalid_parameter_default_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = copy.deepcopy(self.valid_manifest)
            invalid["training_parameters"]["epochs"]["default"] = "one"
            path = self.write_manifest(Path(directory), "invalid", invalid)
            with self.assertRaisesRegex(ManifestValidationError, "default does not match"):
                load_manifest(path)

    def test_duplicate_detector_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_manifest(root, "first", self.valid_manifest)
            self.write_manifest(root, "second", self.valid_manifest)
            with self.assertRaisesRegex(ManifestValidationError, "duplicate detector id 'cgnn'"):
                discover_detectors(root)

    def test_discovery_does_not_import_detector_or_runtime_modules(self):
        blocked = ("torch", "celery", "tasks", "cgnn", "anomaly_detection", "learning_adaptation")
        discover_detectors(REPOSITORY_ROOT / "detectors")
        unexpected = sorted(
            name
            for name in sys.modules
            if name in blocked or name.startswith(tuple(f"{item}." for item in blocked))
        )
        self.assertEqual(unexpected, [])

    def test_endpoint_lists_manifests_without_loading_entry_point(self):
        flask = importlib.import_module("flask")
        app = flask.Flask(__name__)
        app.register_blueprint(create_detectors_blueprint(REPOSITORY_ROOT / "detectors"))
        modules_before_request = set(sys.modules)

        response = app.test_client().get("/detectors")
        modules_loaded_by_request = set(sys.modules).difference(modules_before_request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [manifest["id"] for manifest in response.get_json()["detectors"]],
            ["cgnn", "isolation-forest"],
        )
        self.assertNotIn("detectors.cgnn.adapter", modules_loaded_by_request)
        self.assertNotIn("detectors.isolation_forest.adapter", modules_loaded_by_request)


if __name__ == "__main__":
    unittest.main()
