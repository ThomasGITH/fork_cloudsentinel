import ast
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "examples" / "external_plugins" / "template_detector"


class ExternalPluginTemplateTests(unittest.TestCase):
    def test_template_has_minimum_external_package_files(self):
        for filename in ("manifest.yaml", "adapter.py", "requirements.lock", "README.md"):
            self.assertTrue((TEMPLATE / filename).is_file(), filename)
        self.assertFalse((ROOT / "detectors" / "template_detector").exists())

    def test_template_manifest_is_parseable_but_deliberately_unpublishable(self):
        manifest = yaml.safe_load((TEMPLATE / "manifest.yaml").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertTrue(manifest["id"].startswith("<replace-"))
        self.assertEqual(manifest["capabilities"]["training"]["protocol"], "cloudsentinel.training/v1")
        self.assertEqual(manifest["training_parameters"], {})
        self.assertEqual(manifest["entry_point"], "adapter:TemplateDetectorAdapter")

    def test_adapter_skeleton_is_valid_python_and_has_training_methods(self):
        source = (TEMPLATE / "adapter.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        adapter = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TemplateDetectorAdapter"
        )
        methods = {node.name for node in adapter.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(methods, {"validate_training", "run_training"})

    def test_tutorial_points_to_template_and_minimum_contract(self):
        tutorial = (ROOT / "docs" / "external-plugin-tutorial.md").read_text(encoding="utf-8")
        self.assertIn("examples/external_plugins/template_detector", tutorial)
        for filename in ("manifest.yaml", "adapter.py", "requirements.lock"):
            self.assertIn(filename, tutorial)


if __name__ == "__main__":
    unittest.main()
