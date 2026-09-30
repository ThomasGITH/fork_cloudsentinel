# Generic Detection staging deployment

This procedure targets the current single-node EC2/Minikube staging setup. It
does not migrate callers away from the legacy CGNN or Isolation Forest
services.

## Build and load the images

Both images use the repository root as build context. The manifests use these
immutable staging tags:

- `jojojochem/learning_adaptation:generic-detection-1`
- `jojojochem/anomaly_detection_generic:generic-detection-1`

Run from the repository root:

```bash
docker build \
  -f learning_adaptation/Dockerfile \
  -t jojojochem/learning_adaptation:generic-detection-1 \
  .

docker build \
  -f anomaly_detection/generic/Dockerfile \
  -t jojojochem/anomaly_detection_generic:generic-detection-1 \
  .
```

The generic image contains the common CPU runtime, registry, built-in adapters,
and CGNN modules. External LOF code is loaded from the Plugin PVC and is not
baked into this image.

Load both images into Minikube:

```bash
minikube image load jojojochem/learning_adaptation:generic-detection-1
minikube image load jojojochem/anomaly_detection_generic:generic-detection-1

minikube image ls | grep -E 'learning_adaptation|anomaly_detection_generic'
```

For a registry-based cluster, push these exact tags and confirm that the node
can pull them before applying the Deployments. Keep the Learning API and worker
on the same tag.

## Apply order and storage

Apply the manifests in this order:

```bash
kubectl apply -f k8s/namespace-deployment.yml

kubectl apply -f k8s/learning_adaptation-pvc.yml
kubectl apply -f k8s/detector-plugin-repository-pvc.yml
kubectl apply -f k8s/model_artifact-pvc.yml
kubectl apply -f k8s/detection_results-pvc.yml

kubectl apply -f k8s/learning_adaptation-deployment.yml
kubectl apply -f k8s/learning_adaptation-celery-deployment.yml

kubectl apply -f k8s/generic_anomaly_detection-deployment.yml
kubectl apply -f k8s/generic_anomaly_detection-service.yml
```

`learning-adaptation-pvc` remains the source for snapshots, run records, model
catalogue records, and existing temporary CGNN data. The learning worker writes
new immutable inference artifacts to `model-artifact-pvc`. The generic runtime
reads both claims and the Plugin PVC without write access. It writes compact
results only to `detection-results-pvc` and process-temporary files to `/tmp`.

All claims are `ReadWriteOnce` and omit `storageClassName`. This is suitable
only while all consumers run on the same Minikube node.

## Rollout and health checks

```bash
kubectl rollout status -n cloudsentinel \
  deployment/learning-adaptation-deployment --timeout=180s
kubectl rollout status -n cloudsentinel \
  deployment/learning-adaptation-celery-deployment --timeout=180s
kubectl rollout status -n cloudsentinel \
  deployment/generic-anomaly-detection-deployment --timeout=180s

kubectl get pods,pvc,svc -n cloudsentinel
kubectl get deployment -n cloudsentinel generic-anomaly-detection-deployment \
  -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'
```

Use a temporary local port-forward:

```bash
kubectl port-forward -n cloudsentinel \
  service/generic-anomaly-detection-service 5015:80
```

In another shell:

```bash
curl --fail --silent http://127.0.0.1:5015/healthz
```

The health probe performs no model load and does not depend on Redis,
Prometheus, or PVC contents.

The cluster-internal URL is:

```text
http://generic-anomaly-detection-service.cloudsentinel.svc.cluster.local:80
```

It is a `ClusterIP` service and is not exposed through a NodePort or load
balancer.

## Publish LOF and train a fresh model

Publish and activate the external LOF package:

```bash
./scripts/external_plugin.sh publish \
  "$(pwd)/examples/external_plugins/local_outlier_factor" \
  --id local-outlier-factor \
  --version 1.0.0 \
  --activate
```

If `1.0.0` was already published with different contents, assign a new
manifest version before publishing, or use `./scripts/external_plugin.sh list`
and activate the intended immutable digest through the operator fallback.

After the updated Learning API and worker are ready, create a new TrainingRun.
Use an available Data Catalogue version with an active labelled
`metrics-partition-v1` partition and select Local Outlier Factor. Existing
Saved Models are not migrated. A new successful inference-enabled run writes
the artifact PVC and produces `inference.status = ready` in its Saved Model.

## Activation and `/detect` smoke test

Keep the port-forward active. Substitute the new model ID and the exact Saved
Model feature order:

```bash
export GENERIC_MODEL_ID='model_replace_me'

curl --fail --silent --request POST \
  "http://127.0.0.1:5015/models/${GENERIC_MODEL_ID}/activate"

cat > /tmp/cloudsentinel-matrix.csv <<'EOF'
0.12,0.31
0.15,0.29
8.00,7.50
EOF

cat > /tmp/cloudsentinel-detection-metadata.json <<EOF
{
  "model_id": "${GENERIC_MODEL_ID}",
  "feature_order": ["replace_with_feature_1", "replace_with_feature_2"]
}
EOF

curl --fail --silent --request POST \
  --form 'matrix=@/tmp/cloudsentinel-matrix.csv;type=text/csv' \
  --form 'metadata=</tmp/cloudsentinel-detection-metadata.json' \
  http://127.0.0.1:5015/detect
```

The matrix width and feature order must exactly match the model. The compact
response contains counts, anomaly percentage, warm-up observations, and a
logical result reference; raw predictions and scores are not returned.

## Parity and rollback

Before migrating a caller, run the same bounded matrix through the legacy and
generic endpoints for the same trained model. Compare prediction count,
warm-up handling, anomaly count, percentage, feature validation, and errors.
CGNN requires more observations than its stored lookback; IF and LOF have no
warm-up.

Roll back application revisions without deleting PVC data:

```bash
kubectl rollout undo -n cloudsentinel deployment/learning-adaptation-deployment
kubectl rollout undo -n cloudsentinel deployment/learning-adaptation-celery-deployment
kubectl rollout undo -n cloudsentinel deployment/generic-anomaly-detection-deployment

kubectl rollout status -n cloudsentinel deployment/learning-adaptation-deployment
kubectl rollout status -n cloudsentinel deployment/learning-adaptation-celery-deployment
kubectl rollout status -n cloudsentinel deployment/generic-anomaly-detection-deployment
```

To disable only the generic runtime while preserving artifacts and results:

```bash
kubectl scale -n cloudsentinel \
  deployment/generic-anomaly-detection-deployment --replicas=0
```

Do not delete the PVCs during rollback. The legacy CGNN and Isolation Forest
services remain deployed and independent.

## Known staging limitations

- RWO storage and file-backed records assume a single node and one runtime
  replica. Multi-node operation needs RWX or object storage and locking.
- The in-process model cache is local and bounded to two entries here.
- Resource requests and limits reuse the existing PyTorch learning-worker
  envelope. Measure real models before production tuning.
- No old model migration, caller migration, generic RCA dispatch, streaming
  CGNN state, GPU pool, or runtime dependency installation is included.
