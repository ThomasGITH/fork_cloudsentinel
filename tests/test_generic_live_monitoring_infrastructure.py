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


def mounts(path):
    return {item["name"]: item for item in container(path).get("volumeMounts", [])}


class GenericLiveMonitoringInfrastructureTests(unittest.TestCase):
    def test_data_ingestion_pair_uses_same_new_image(self):
        api = container("k8s/data_ingestion-deployment.yml")
        worker = container("k8s/data_ingestion_celery-deployment.yml")
        self.assertEqual(api["image"], worker["image"])
        self.assertEqual(api["image"], "jojojochem/data_ingestion:generic-monitoring-rca-reanalysis-1")
        self.assertEqual(environment("k8s/data_ingestion-deployment.yml")["MONITORING_HISTORY_STORAGE_ROOT"], "/app/storage/monitoring_history")
        self.assertEqual(environment("k8s/data_ingestion_celery-deployment.yml")["MONITORING_HISTORY_STORAGE_ROOT"], "/app/storage/monitoring_history")
        self.assertEqual(mounts("k8s/data_ingestion-deployment.yml")["catalogue-storage"]["mountPath"], "/app/storage")
        self.assertEqual(mounts("k8s/data_ingestion_celery-deployment.yml")["catalogue-storage"]["mountPath"], "/app/storage")
        self.assertFalse(mounts("k8s/data_ingestion-deployment.yml")["catalogue-storage"].get("readOnly", False))
        self.assertFalse(mounts("k8s/data_ingestion_celery-deployment.yml")["catalogue-storage"].get("readOnly", False))
        dockerfile = (ROOT / "data_ingestion/Dockerfile").read_text()
        self.assertIn("/app/storage/monitoring_history", dockerfile)

    def test_service_urls_are_server_side_and_minimally_scoped(self):
        api = environment("k8s/data_ingestion-deployment.yml")
        worker = environment("k8s/data_ingestion_celery-deployment.yml")
        self.assertIn("API_LEARNING_ADAPTATION_URL", api)
        self.assertNotIn("API_GENERIC_ANOMALY_DETECTION_URL", api)
        self.assertIn("API_GENERIC_ANOMALY_DETECTION_URL", worker)
        self.assertNotIn("API_LEARNING_ADAPTATION_URL", worker)

    def test_learning_pair_and_ui_use_immutable_feature_tag(self):
        learning_api = container("k8s/learning_adaptation-deployment.yml")
        learning_worker = container("k8s/learning_adaptation-celery-deployment.yml")
        ui = container("k8s/monitoring_project-deployment.yml")
        self.assertEqual(learning_api["image"], learning_worker["image"])
        self.assertTrue(learning_api["image"].endswith(":generic-monitoring-history-1"))
        self.assertTrue(ui["image"].endswith(":generic-monitoring-rca-reanalysis-1"))
        self.assertEqual(ui["imagePullPolicy"], "IfNotPresent")

        admin = yaml.safe_load((ROOT / "k8s/plugin-admin-pod.yml").read_text())
        self.assertEqual(
            admin["spec"]["containers"][0]["image"], learning_api["image"]
        )

    def test_legacy_cgnn_resources_are_intentionally_retained(self):
        path = ROOT / "k8s/cgnn_anomaly_detection-deployment.yml"
        self.assertTrue(path.is_file())
        documents = list(yaml.safe_load_all(path.read_text()))
        self.assertIn("Deployment", {item.get("kind") for item in documents if item})
        self.assertIn("Service", {item.get("kind") for item in documents if item})

    def test_generic_runtime_image_is_unchanged_by_history_tranche(self):
        generic = container("k8s/generic_anomaly_detection-deployment.yml")
        self.assertEqual(
            generic["image"],
            "jojojochem/anomaly_detection_generic:generic-live-monitoring-1.1",
        )

    def test_monitoring_uses_one_generic_evaluation_and_recipe_consistent_rca(self):
        source = (ROOT / "data_ingestion/app.py").read_text()
        task = source[source.index("def monitoring_task"):source.index("@app.route('/start_monitoring'")]
        self.assertIn("_generic_detect(session, bounded)", task)
        self.assertIn("_reconcile_rca_history(session_id)", task)
        self.assertNotIn("collect_crca_data", task)
        generic_call = source[source.index("def _generic_detect"):source.index("def _cycle_id")]
        self.assertIn("/internal/evaluate", generic_call)
        self.assertNotIn('f"{base_url}/detect"', generic_call)


if __name__ == "__main__":
    unittest.main()
