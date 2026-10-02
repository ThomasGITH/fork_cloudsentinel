from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
K8S = ROOT / "k8s"
GENERIC_IMAGE = "jojojochem/anomaly_detection_generic:comparison-mvp-1"
LEARNING_IMAGE = "jojojochem/learning_adaptation:comparison-mvp-1.2"


def documents(name):
    return list(yaml.safe_load_all((K8S / name).read_text(encoding="utf-8")))


def deployment(name):
    return next(item for item in documents(name) if item.get("kind") == "Deployment")


def container(value):
    return value["spec"]["template"]["spec"]["containers"][0]


def by_name(items):
    return {item["name"]: item for item in items}


class GenericDetectionInfrastructureTests(unittest.TestCase):
    def test_persistent_volume_claims_are_single_node_and_storage_class_agnostic(self):
        expected = {
            "model_artifact-pvc.yml": ("model-artifact-pvc", "5Gi"),
            "detection_results-pvc.yml": ("detection-results-pvc", "1Gi"),
        }
        for filename, (claim_name, size) in expected.items():
            with self.subTest(filename=filename):
                claim = yaml.safe_load((K8S / filename).read_text(encoding="utf-8"))
                self.assertEqual(claim["kind"], "PersistentVolumeClaim")
                self.assertEqual(claim["metadata"]["name"], claim_name)
                self.assertEqual(claim["metadata"]["namespace"], "cloudsentinel")
                self.assertEqual(claim["spec"]["accessModes"], ["ReadWriteOnce"])
                self.assertEqual(claim["spec"]["resources"]["requests"]["storage"], size)
                self.assertNotIn("storageClassName", claim["spec"])

    def test_generic_runtime_is_internal_single_replica_and_hardened(self):
        value = deployment("generic_anomaly_detection-deployment.yml")
        self.assertEqual(value["metadata"]["namespace"], "cloudsentinel")
        self.assertEqual(value["spec"]["replicas"], 1)
        self.assertEqual(value["spec"]["strategy"]["type"], "Recreate")
        pod = value["spec"]["template"]["spec"]
        self.assertFalse(pod["automountServiceAccountToken"])
        pod_security = pod["securityContext"]
        self.assertTrue(pod_security["runAsNonRoot"])
        self.assertEqual(pod_security["runAsUser"], 10001)
        self.assertEqual(pod_security["runAsGroup"], 10001)
        self.assertEqual(pod_security["fsGroup"], 10001)
        self.assertEqual(pod_security["seccompProfile"]["type"], "RuntimeDefault")
        runtime = container(value)
        self.assertEqual(runtime["image"], GENERIC_IMAGE)
        self.assertNotIn(":latest", runtime["image"])
        security = runtime["securityContext"]
        self.assertFalse(security["allowPrivilegeEscalation"])
        self.assertTrue(security["readOnlyRootFilesystem"])
        self.assertEqual(security["capabilities"]["drop"], ["ALL"])
        self.assertEqual(runtime["ports"][0]["containerPort"], 5015)
        self.assertEqual(runtime["readinessProbe"]["httpGet"]["path"], "/healthz")
        self.assertEqual(runtime["livenessProbe"]["httpGet"]["path"], "/healthz")
        self.assertIn("requests", runtime["resources"])
        self.assertIn("limits", runtime["resources"])

    def test_generic_runtime_mount_and_environment_access_boundaries(self):
        value = deployment("generic_anomaly_detection-deployment.yml")
        runtime = container(value)
        mounts = by_name(runtime["volumeMounts"])
        self.assertTrue(mounts["learning-storage"]["readOnly"])
        self.assertEqual(mounts["learning-storage"]["mountPath"], "/app/learning-storage")
        self.assertTrue(mounts["model-artifacts"]["readOnly"])
        self.assertEqual(mounts["model-artifacts"]["mountPath"], "/app/storage/model_artifacts")
        self.assertTrue(mounts["detector-plugins"]["readOnly"])
        self.assertEqual(mounts["detector-plugins"]["mountPath"], "/opt/cloudsentinel/detectors")
        self.assertFalse(mounts["detection-results"]["readOnly"])
        self.assertEqual(mounts["detection-results"]["mountPath"], "/app/storage/results")
        environment = {item["name"]: item["value"] for item in runtime["env"]}
        self.assertEqual(environment["MODEL_CATALOGUE_STORAGE_ROOT"], "/app/learning-storage/models")
        self.assertEqual(environment["MODEL_ARTIFACT_STORAGE_ROOT"], "/app/storage/model_artifacts")
        self.assertEqual(environment["GENERIC_DETECTION_RESULTS_ROOT"], "/app/storage/results")
        self.assertEqual(environment["DETECTOR_PLUGIN_ROOTS"], "/opt/cloudsentinel/detectors/active")
        volumes = by_name(value["spec"]["template"]["spec"]["volumes"])
        self.assertTrue(volumes["learning-storage"]["persistentVolumeClaim"]["readOnly"])
        self.assertTrue(volumes["model-artifacts"]["persistentVolumeClaim"]["readOnly"])
        self.assertTrue(volumes["detector-plugins"]["persistentVolumeClaim"]["readOnly"])
        self.assertEqual(
            volumes["detection-results"]["persistentVolumeClaim"]["claimName"],
            "detection-results-pvc",
        )

    def test_service_is_cluster_only_and_maps_port_80_to_named_runtime_port(self):
        service = yaml.safe_load(
            (K8S / "generic_anomaly_detection-service.yml").read_text(encoding="utf-8")
        )
        self.assertEqual(service["kind"], "Service")
        self.assertEqual(service["metadata"]["namespace"], "cloudsentinel")
        self.assertEqual(service["spec"]["type"], "ClusterIP")
        self.assertEqual(service["spec"]["ports"][0]["port"], 80)
        self.assertEqual(service["spec"]["ports"][0]["targetPort"], "http")
        self.assertNotIn("nodePort", service["spec"]["ports"][0])

    def test_learning_worker_publishes_artifacts_and_api_has_no_artifact_mount(self):
        api = deployment("learning_adaptation-deployment.yml")
        worker = deployment("learning_adaptation-celery-deployment.yml")
        api_container = container(api)
        worker_container = container(worker)
        self.assertEqual(api_container["image"], LEARNING_IMAGE)
        self.assertEqual(worker_container["image"], LEARNING_IMAGE)
        self.assertNotIn("model-artifacts", by_name(api_container["volumeMounts"]))
        worker_mount = by_name(worker_container["volumeMounts"])["model-artifacts"]
        self.assertEqual(worker_mount["mountPath"], "/app/storage/model_artifacts")
        self.assertFalse(worker_mount["readOnly"])
        worker_environment = {
            item["name"]: item["value"] for item in worker_container["env"]
        }
        self.assertEqual(
            worker_environment["MODEL_ARTIFACT_STORAGE_ROOT"],
            "/app/storage/model_artifacts",
        )
        self.assertNotIn("API_ISOLATION_FOREST_ANOMALY_DETECTION_URL", worker_environment)
        worker_volumes = by_name(worker["spec"]["template"]["spec"]["volumes"])
        self.assertEqual(
            worker_volumes["model-artifacts"]["persistentVolumeClaim"]["claimName"],
            "model-artifact-pvc",
        )

    def test_learning_image_build_verifies_comparison_runtime_modules(self):
        dockerfile = (ROOT / "learning_adaptation" / "Dockerfile").read_text(
            encoding="utf-8"
        )
        for module in (
            "comparison_api.py",
            "comparison_execution.py",
            "comparison_evaluation.py",
            "comparison_storage.py",
        ):
            self.assertIn(f"test -f /app/{module}", dockerfile)
        self.assertIn(
            "import comparison_api, comparison_evaluation, comparison_execution, comparison_storage, tasks",
            dockerfile,
        )
        self.assertIn("tasks.execute_comparison_task", dockerfile)

    def test_external_plugin_mount_and_learning_image_remain_consistent(self):
        images = []
        for filename in (
            "learning_adaptation-deployment.yml",
            "learning_adaptation-celery-deployment.yml",
        ):
            value = deployment(filename)
            runtime = container(value)
            images.append(runtime["image"])
            plugin_mount = by_name(runtime["volumeMounts"])["detector-plugins"]
            self.assertTrue(plugin_mount["readOnly"])
        admin = yaml.safe_load((K8S / "plugin-admin-pod.yml").read_text(encoding="utf-8"))
        self.assertEqual(admin["spec"]["containers"][0]["image"], LEARNING_IMAGE)
        self.assertEqual(images, [LEARNING_IMAGE, LEARNING_IMAGE])

    def test_generic_dockerfile_uses_root_context_and_does_not_bake_external_lof(self):
        dockerfile = (ROOT / "anomaly_detection" / "generic" / "Dockerfile").read_text(
            encoding="utf-8"
        )
        requirements_text = (
            ROOT / "anomaly_detection" / "generic" / "requirements.txt"
        ).read_text(encoding="utf-8")
        requirements = {
            line.strip()
            for line in requirements_text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("FROM python:3.12-slim", dockerfile)
        self.assertIn("COPY pyproject.toml", dockerfile)
        self.assertIn("COPY detectors", dockerfile)
        self.assertIn("learning_adaptation /app/learning_adaptation", dockerfile)
        self.assertIn("anomaly_detection/generic /app/anomaly_detection/generic", dockerfile)
        self.assertIn("USER 10001:10001", dockerfile)
        self.assertIn("gunicorn", dockerfile)
        self.assertNotIn("examples/external_plugins", dockerfile)
        for dependency in (
            "Flask==",
            "gunicorn==",
            "scikit-learn==",
            "joblib==",
            "numpy==",
            "pandas==",
            "scipy==",
            "PyYAML==",
        ):
            self.assertTrue(
                any(item.startswith(dependency) for item in requirements),
                dependency,
            )

    def test_generic_image_installs_only_the_pinned_cpu_torch_wheel(self):
        dockerfile = (ROOT / "anomaly_detection" / "generic" / "Dockerfile").read_text(
            encoding="utf-8"
        )
        requirements = (
            ROOT / "anomaly_detection" / "generic" / "requirements.txt"
        ).read_text(encoding="utf-8")
        normalized_requirements = {
            line.split("#", 1)[0].strip().lower()
            for line in requirements.splitlines()
            if line.split("#", 1)[0].strip()
        }

        self.assertIn("ARG PYTORCH_VERSION=2.2.2", dockerfile)
        self.assertIn("https://download.pytorch.org/whl/cpu", dockerfile)
        self.assertIn('"torch==${PYTORCH_VERSION}+cpu"', dockerfile)
        self.assertFalse(any(item.startswith("torch") for item in normalized_requirements))

        dependency_configuration = f"{dockerfile}\n{requirements}".lower()
        for gpu_package_marker in ("nvidia-", "nvidia_", "cu12", "cudnn", "cublas"):
            self.assertNotIn(gpu_package_marker, dependency_configuration)

        # CGNN inference remains available through the explicitly installed CPU wheel
        # and the built-in detector/application modules copied into the image.
        self.assertIn("COPY detectors", dockerfile)
        self.assertIn("learning_adaptation /app/learning_adaptation", dockerfile)

    def test_generic_image_build_context_is_selective_and_ignores_runtime_data(self):
        dockerfile = (ROOT / "anomaly_detection" / "generic" / "Dockerfile").read_text(
            encoding="utf-8"
        )
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

        self.assertNotIn("COPY . ", dockerfile)
        self.assertNotIn("ADD . ", dockerfile)
        for ignored in (
            ".git",
            "**/env",
            "**/.venv",
            "**/storage",
            "**/model_artifacts",
            "**/training_runs",
            "**/trained_models_temp",
        ):
            self.assertIn(ignored, dockerignore)

    def test_cgnn_legacy_resources_remain_but_if_resources_are_retired(self):
        cgnn = (K8S / "cgnn_anomaly_detection-deployment.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("cgnn-anomaly-detection-deployment", cgnn)
        self.assertIn("cgnn-anomaly-detection-service", cgnn)
        self.assertFalse(
            (K8S / "isolation_forest_anomaly_detection-deployment.yml").exists()
        )
        self.assertNotIn("model-artifact-pvc", cgnn)
        generic = (
            K8S / "generic_anomaly_detection-deployment.yml"
        ).read_text(encoding="utf-8")
        for detector_id in ("local-outlier-factor", "isolation-forest", "cgnn"):
            self.assertNotIn(f'app: {detector_id}', generic)

    def test_data_processing_calls_only_the_server_configured_generic_runtime(self):
        value = deployment("data_processing-deployment.yml")
        runtime = container(value)
        environment = {item["name"]: item["value"] for item in runtime["env"]}
        self.assertEqual(
            environment["API_GENERIC_ANOMALY_DETECTION_URL"],
            "http://generic-anomaly-detection-service.cloudsentinel.svc.cluster.local:80",
        )
        self.assertEqual(
            environment["API_LEARNING_ADAPTATION_URL"],
            "http://learning-adaptation-service.cloudsentinel.svc.cluster.local:80",
        )
        self.assertNotIn("API_CGNN_ANOMALY_DETECTION_URL", environment)


if __name__ == "__main__":
    unittest.main()
