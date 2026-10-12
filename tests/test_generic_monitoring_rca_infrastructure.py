from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


def deployment(path):
    documents = list(yaml.safe_load_all((ROOT / path).read_text()))
    return next(item for item in documents if item and item.get("kind") == "Deployment")


def container(path):
    return deployment(path)["spec"]["template"]["spec"]["containers"][0]


def environment(path):
    return {item["name"]: item.get("value") for item in container(path).get("env", [])}


class GenericMonitoringRcaInfrastructureTests(unittest.TestCase):
    def test_changed_images_use_one_immutable_tag(self):
        tag = "generic-monitoring-rca-readiness-fix-1"
        self.assertEqual(container("k8s/data_ingestion-deployment.yml")["image"], f"jojojochem/data_ingestion:{tag}")
        self.assertEqual(container("k8s/data_ingestion_celery-deployment.yml")["image"], f"jojojochem/data_ingestion:{tag}")
        self.assertEqual(container("k8s/monitoring_project-deployment.yml")["image"], f"jojojochem/monitoring_project:{tag}")
        self.assertEqual(container("k8s/crca_anomaly_detection-deployment.yml")["image"], f"jojojochem/anomaly_detection_crca:{tag}")
        self.assertEqual(container("k8s/crca_anomaly_detection-celery-deployment.yml")["image"], f"jojojochem/anomaly_detection_crca:{tag}")

    def test_crca_worker_has_shared_write_storage_and_api_has_health_probe(self):
        worker = container("k8s/crca_anomaly_detection-celery-deployment.yml")
        mount = next(item for item in worker["volumeMounts"] if item["name"] == "catalogue-storage")
        self.assertEqual(mount["mountPath"], "/app/storage")
        self.assertFalse(mount.get("readOnly", False))
        self.assertEqual(environment("k8s/crca_anomaly_detection-celery-deployment.yml")["MONITORING_HISTORY_STORAGE_ROOT"], "/app/storage/monitoring_history")
        api = container("k8s/crca_anomaly_detection-deployment.yml")
        self.assertEqual(api["readinessProbe"]["httpGet"]["path"], "/healthz")

    def test_data_ingestion_uses_only_server_configured_rca_url(self):
        self.assertIn("API_CRCA_ANOMALY_DETECTION_URL", environment("k8s/data_ingestion-deployment.yml"))
        self.assertIn("API_CRCA_ANOMALY_DETECTION_URL", environment("k8s/data_ingestion_celery-deployment.yml"))
        source = (ROOT / "data_ingestion/app.py").read_text()
        self.assertNotIn("payload.get(\"rca_url\")", source)

    def test_legacy_cgnn_resources_are_untouched_and_generic_runtime_image_is_unchanged(self):
        self.assertTrue((ROOT / "k8s/cgnn_anomaly_detection-deployment.yml").is_file())
        self.assertEqual(
            container("k8s/generic_anomaly_detection-deployment.yml")["image"],
            "jojojochem/anomaly_detection_generic:generic-live-monitoring-1.1",
        )

    def test_crca_build_context_copies_shared_history_without_runtime_install(self):
        dockerfile = (ROOT / "anomaly_detection/crca/Dockerfile").read_text()
        self.assertIn("COPY data_ingestion/monitoring_history.py", dockerfile)
        self.assertNotIn("pip install -e", dockerfile)

    def test_crca_image_uses_python_312_compatible_runtime_dependencies(self):
        requirements = (ROOT / "anomaly_detection/crca/requirements.txt").read_text()
        self.assertIn("pandas==2.2.2", requirements)
        self.assertIn("matplotlib==3.9.0", requirements)
        self.assertIn("scikit-learn==1.5.0", requirements)
        self.assertNotIn("pandas==1.3.5", requirements)
        self.assertNotIn("matplotlib==3.5.1", requirements)

    def test_crca_image_installs_torch_from_cpu_only_index(self):
        dockerfile = (ROOT / "anomaly_detection/crca/Dockerfile").read_text()
        requirements = (ROOT / "anomaly_detection/crca/requirements.txt").read_text()
        self.assertIn("https://download.pytorch.org/whl/cpu", dockerfile)
        self.assertIn('"torch==${PYTORCH_VERSION}+cpu"', dockerfile)
        self.assertIn("ARG PYTORCH_VERSION=2.2.2", dockerfile)
        self.assertNotRegex(requirements, r"(?m)^torch(?:==|>=|<=|~=)")
        self.assertNotRegex(requirements, r"(?mi)^nvidia-")

    def test_crca_image_uses_python_312_scikit_network_wheel(self):
        dockerfile = (ROOT / "anomaly_detection/crca/Dockerfile").read_text()
        requirements = (ROOT / "anomaly_detection/crca/requirements.txt").read_text()
        self.assertIn("ARG SCIKIT_NETWORK_VERSION=0.33.0", dockerfile)
        self.assertIn("--only-binary=:all:", dockerfile)
        self.assertIn('"scikit-network==${SCIKIT_NETWORK_VERSION}"', dockerfile)
        self.assertNotIn("scikit-network==0.31.0", requirements)


if __name__ == "__main__":
    unittest.main()
