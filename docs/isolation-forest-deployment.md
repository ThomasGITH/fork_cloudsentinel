# Isolation Forest detection deployment

IF-C4 packages the standalone Isolation Forest detection service for local
Docker and Kubernetes deployment. It does not deploy to a cluster or EC2 and
does not change model training, promotion, prediction, RCA, or CGNN behavior.

Build both affected images from the repository root:

```shell
docker build -f anomaly_detection/isolation_forest/Dockerfile \
  -t anomaly_detection_isolation_forest:local .
docker build -f learning_adaptation/Dockerfile \
  -t learning_adaptation:local .
```

The Isolation Forest image contains the service runtime from
`anomaly_detection/isolation_forest/requirements.txt` and the shared
`detectors` package. It listens on port 5014 and exposes `GET /healthz`.

The Kubernetes resources are in
`k8s/isolation_forest_anomaly_detection-deployment.yml`:

- `isolation-forest-anomaly-detection-deployment`
- `isolation-forest-anomaly-detection-service`
- `isolation-forest-detection-pvc`

Within the `cloudsentinel` namespace, the learning Celery worker uses:

```text
API_ISOLATION_FOREST_ANOMALY_DETECTION_URL=http://isolation-forest-anomaly-detection-service.cloudsentinel.svc.cluster.local
```

The ClusterIP Service is internal and maps port 80 to container port 5014.
The PVC is mounted once at `/app/storage`; models and results use:

```text
IF_MODEL_STORAGE_ROOT=/app/storage/trained_models
IF_RESULTS_STORAGE_ROOT=/app/storage/results
IF_MAX_UPLOAD_BYTES=104857600
```

The initial deployment uses one replica and ReadWriteOnce storage. Multiple
replicas require shared storage that supports the chosen access pattern,
cross-replica result locking, and result coordination. Without a PVC,
promoted models and detection results are lost when the pod is replaced.
No storage class is selected; the cluster's default provisioner must satisfy
the claim.
