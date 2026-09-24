import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest

import yaml

from detectors.contracts import DetectorAdapter
from detectors.registry import (
    AdapterContractError,
    AdapterImportError,
    InvalidEntryPointError,
    UnknownDetectorError,
    discover_detectors,
    get_adapter,
    load_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CGNN_MANIFEST = REPOSITORY_ROOT / "detectors" / "cgnn" / "manifest.yaml"


class DetectorResolverTests(unittest.TestCase):
    def setUp(self):
        self.valid_manifest = load_manifest(CGNN_MANIFEST)

    def tearDown(self):
        for module_name in (
            "detectors.cgnn.adapter",
            "phase_c_missing_attribute",
            "phase_c_invalid_adapter",
        ):
            sys.modules.pop(module_name, None)

    def write_manifest(self, root: Path, manifest) -> None:
        plugin_dir = root / "plugin"
        plugin_dir.mkdir()
        (plugin_dir / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )

    def test_discovery_does_not_import_adapter_or_runtime(self):
        discover_detectors(REPOSITORY_ROOT / "detectors")

        self.assertNotIn("detectors.cgnn.adapter", sys.modules)
        for module_name in ("torch", "celery", "cgnn.train", "cgnn.evaluate_prediction", "predict"):
            self.assertNotIn(module_name, sys.modules)

    def test_get_adapter_returns_cgnn_instance_and_resolves_lazily(self):
        self.assertNotIn("detectors.cgnn.adapter", sys.modules)

        adapter = get_adapter("cgnn", REPOSITORY_ROOT / "detectors")

        self.assertEqual(type(adapter).__name__, "CGNNAdapter")
        self.assertNotIsInstance(adapter, type)
        self.assertIsInstance(adapter, DetectorAdapter)
        self.assertIn("detectors.cgnn.adapter", sys.modules)
        for module_name in ("torch", "celery", "cgnn.train", "cgnn.evaluate_prediction", "predict"):
            self.assertNotIn(module_name, sys.modules)

    def test_unknown_detector_has_targeted_error(self):
        with self.assertRaisesRegex(UnknownDetectorError, "Unknown detector id: 'missing'"):
            get_adapter("missing", REPOSITORY_ROOT / "detectors")

    def test_malformed_entry_point_has_targeted_error(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = copy.deepcopy(self.valid_manifest)
            invalid["entry_point"] = "not an entry point"
            self.write_manifest(Path(directory), invalid)

            with self.assertRaisesRegex(InvalidEntryPointError, "must have the form"):
                get_adapter("cgnn", directory)

    def test_non_importable_entry_point_has_targeted_error(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = copy.deepcopy(self.valid_manifest)
            invalid["entry_point"] = "cloudsentinel_missing_adapter:Adapter"
            self.write_manifest(Path(directory), invalid)

            with self.assertRaisesRegex(AdapterImportError, "Could not import adapter module"):
                get_adapter("cgnn", directory)

    def test_missing_entry_point_attribute_has_targeted_error(self):
        module = types.ModuleType("phase_c_missing_attribute")
        sys.modules[module.__name__] = module
        with tempfile.TemporaryDirectory() as directory:
            invalid = copy.deepcopy(self.valid_manifest)
            invalid["entry_point"] = f"{module.__name__}:MissingAdapter"
            self.write_manifest(Path(directory), invalid)

            with self.assertRaisesRegex(InvalidEntryPointError, "does not exist"):
                get_adapter("cgnn", directory)

    def test_adapter_contract_is_validated(self):
        module = types.ModuleType("phase_c_invalid_adapter")

        class IncompleteAdapter:
            @staticmethod
            def train(*args, **kwargs):
                return None

            @staticmethod
            def evaluate(*args, **kwargs):
                return None

        module.IncompleteAdapter = IncompleteAdapter
        sys.modules[module.__name__] = module
        with tempfile.TemporaryDirectory() as directory:
            invalid = copy.deepcopy(self.valid_manifest)
            invalid["entry_point"] = f"{module.__name__}:IncompleteAdapter"
            self.write_manifest(Path(directory), invalid)

            with self.assertRaisesRegex(AdapterContractError, "DetectorAdapter: predict"):
                get_adapter("cgnn", directory)


if __name__ == "__main__":
    unittest.main()
