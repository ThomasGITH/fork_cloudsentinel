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
        self.assertEqual(container("k8s/data_ingestion-deployment.yml")["image"], "jojojochem/data_ingestion:generic-monitoring-rca-1")
        self.assertEqual(container("k8s/data_ingestion_celery-deployment.yml")["image"], "jojojochem/data_ingestion:generic-monitoring-rca-1")
        self.assertEqual(container("k8s/monitoring_project-deployment.yml")["image"], "jojojochem/monitoring_project:generic-monitoring-rca-1")
        self.assertEqual(container("k8s/crca_anomaly_detection-deployment.yml")["image"], "jojojochem/anomaly_detection_crca:generic-monitoring-rca-1")
        self.assertEqual(container("k8s/crca_anomaly_detection-celery-deployment.yml")["image"], "jojojochem/anomaly_detection_crca:generic-monitoring-rca-1")

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


if __name__ == "__main__":
    unittest.main()
