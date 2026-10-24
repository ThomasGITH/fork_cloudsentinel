import io
import os
import unittest
from unittest.mock import Mock, patch

from flask import Flask

from learning_adaptation.plugin_upload_api import create_plugin_upload_blueprint


class PluginUploadApiTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(create_plugin_upload_blueprint())
        self.client = self.app.test_client()
        self.environment = patch.dict(
            os.environ,
            {
                "PLUGIN_OPERATOR_KEY": "operator-secret",
                "DETECTOR_PLUGIN_REPOSITORY_ROOT": "/tmp/test-detectors",
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    @staticmethod
    def package():
        return {
            "manifest": (io.BytesIO(b"schema_version: 1\n"), "manifest.yaml"),
            "adapter": (io.BytesIO(b"class Adapter: pass\n"), "adapter.py"),
            "requirements": (io.BytesIO(b"numpy==1.26.4\n"), "requirements.lock"),
        }

    def test_invalid_operator_key_is_rejected_before_staging(self):
        response = self.client.post(
            "/detectors/plugins/upload",
            headers={"X-Plugin-Operator-Key": "incorrect"},
            data=self.package(),
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json()["error"], "The operator access key is invalid")

    @patch("learning_adaptation.plugin_upload_api.PluginRepository")
    def test_validate_checks_manifest_and_adapter_without_installing(self, repository_type):
        repository = Mock()
        repository.validate_source.return_value = {
            "manifest": {"id": "browser-detector", "version": "1.0.0"},
            "package_sha256": "a" * 64,
        }
        repository_type.return_value = repository
        response = self.client.post(
            "/detectors/plugins/upload",
            headers={"X-Plugin-Operator-Key": "operator-secret"},
            data={**self.package(), "action": "validate"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "valid")
        repository._validate_adapter_contract.assert_called_once()
        repository.install.assert_not_called()

    @patch("learning_adaptation.plugin_upload_api.PluginRepository")
    def test_publish_can_activate_the_immutable_release(self, repository_type):
        repository = Mock()
        repository.validate_source.return_value = {
            "manifest": {"id": "browser-detector", "version": "1.0.0"},
            "package_sha256": "b" * 64,
        }
        repository.install.return_value = {
            "detector_id": "browser-detector",
            "detector_version": "1.0.0",
            "package_sha256": "b" * 64,
        }
        repository.activate.return_value = {
            "detector_id": "browser-detector",
            "detector_version": "1.0.0",
            "package_sha256": "b" * 64,
            "active": True,
        }
        repository_type.return_value = repository
        response = self.client.post(
            "/detectors/plugins/upload",
            headers={"X-Plugin-Operator-Key": "operator-secret"},
            data={**self.package(), "action": "publish", "activate": "true"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "published")
        self.assertTrue(response.get_json()["active"])
        repository.activate.assert_called_once_with("browser-detector", "b" * 64)


if __name__ == "__main__":
    unittest.main()
