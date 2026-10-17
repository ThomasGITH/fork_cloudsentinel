# EXPANSIONS-P2

## Robustness interpretation

Robustness remains a read-only projection over stored terminal comparisons. A
row is eligible only when the comparison is `completed` or `partial`, the model
result completed successfully, and the immutable Saved Model identity matches
`model_id + artifact_id + artifact_manifest_sha256`. Repeated evaluations of
the same artifact and partition checksum are deduplicated; the newest terminal
result is canonical.

Each model card reports measured coverage, labelled coverage, F1 median,
minimum, maximum and range, the lowest-F1 labelled context, and runtime
statistics. It makes no composite score or qualitative robustness claim.

The fair shared-context summary has a stricter boundary. It includes only
immutable partitions for which every selected artifact has a successful result
and finite ground-truth F1. At least two such contexts are required before the
UI identifies the lowest observed F1 variation. Ties remain ties. Results from
different context sets are never used for that conclusion.

## Evaluate in another context

The Robustness tab links to the existing Comparison wizard using
`source_comparison`. The Django server resolves that ID through the Learning
API and signs the resulting model seed state. URL-supplied model IDs are not
trusted. Compatibility is re-evaluated after the operator chooses a dataset.
Choosing the exact source dataset version and partition checksum shows a
warning because it replaces the canonical repeated result rather than adding a
new robustness context. `source_run` and `source_comparison` cannot be combined.

## Workload context

Workload intensity, dominant workload characteristic, and anomaly scenario are
operator-supplied metadata. CloudSentinel does not infer them. Standardized
categories remain the aggregation and filtering keys.

`Other / custom` stores:

```json
{
  "anomaly_scenario": "Other",
  "anomaly_scenario_detail": "Black Friday checkout traffic"
}
```

The detail is trimmed, required for `Other`, and limited to 160 characters. A
detail submitted with a standard category is discarded. Existing records that
do not contain the field remain valid. Dataset anomaly context remains separate
from incident-window semantics.

## Build and deploy on EC2/Minikube

Run from the CloudSentinel repository root. These builds include EXPANSIONS-P2,
P1 already present in the checkout, and the recent Comparison timeline and
monitoring model-label UI fixes present in the same source tree.

```bash
docker build \
  -t jojojochem/data_ingestion:expansions-p2 \
  data_ingestion

docker build \
  -f learning_adaptation/Dockerfile \
  -t jojojochem/learning_adaptation:expansions-p2 \
  .

docker build \
  -t jojojochem/monitoring_project:expansions-p2 \
  user_interface/monitoring_project
```

Load the exact local tags into Minikube:

```bash
minikube image load jojojochem/data_ingestion:expansions-p2
minikube image load jojojochem/learning_adaptation:expansions-p2
minikube image load jojojochem/monitoring_project:expansions-p2
```

The manifests use `imagePullPolicy: IfNotPresent`; do not reuse these tags for
different source trees. Confirm the images are present:

```bash
minikube image ls | grep -E 'jojojochem/(data_ingestion|learning_adaptation|monitoring_project):expansions-p2'
```

Apply API/worker components first and the UI last:

```bash
kubectl apply -n cloudsentinel -f k8s/data_ingestion-deployment.yml
kubectl apply -n cloudsentinel -f k8s/data_ingestion_celery-deployment.yml
kubectl apply -n cloudsentinel -f k8s/learning_adaptation-deployment.yml
kubectl apply -n cloudsentinel -f k8s/learning_adaptation-celery-deployment.yml
kubectl apply -n cloudsentinel -f k8s/monitoring_project-deployment.yml
```

Check rollout and the running image identities:

```bash
kubectl rollout status -n cloudsentinel deployment/data-ingestion-deployment --timeout=180s
kubectl rollout status -n cloudsentinel deployment/celery-worker-deployment --timeout=180s
kubectl rollout status -n cloudsentinel deployment/learning-adaptation-deployment --timeout=180s
kubectl rollout status -n cloudsentinel deployment/learning-adaptation-celery-deployment --timeout=180s
kubectl rollout status -n cloudsentinel deployment/monitoring-project-deployment --timeout=180s

kubectl get pods -n cloudsentinel \
  -l 'app in (data-ingestion,celery-worker,learning-adaptation,learning-adaptation-celery,monitoring-project)' \
  -o custom-columns='POD:.metadata.name,IMAGE:.spec.containers[0].image,READY:.status.containerStatuses[0].ready'
```

If the UI is exposed through the existing NodePort, open its current address.
For a local tunnel use the existing Service:

```bash
kubectl port-forward -n cloudsentinel service/monitoring-project-service 8000:80
```

Then visit `http://127.0.0.1:8000/comparison/`.

Rollback does not alter catalogues, comparisons, artifacts, or PVC data:

```bash
kubectl rollout undo -n cloudsentinel deployment/monitoring-project-deployment
kubectl rollout undo -n cloudsentinel deployment/learning-adaptation-celery-deployment
kubectl rollout undo -n cloudsentinel deployment/learning-adaptation-deployment
kubectl rollout undo -n cloudsentinel deployment/celery-worker-deployment
kubectl rollout undo -n cloudsentinel deployment/data-ingestion-deployment
```

No Generic Detection Runtime, CRCA, Redis, Prometheus, or detector image changed
in this tranche.

The temporary plugin-admin manifest references the same Learning image so the
existing publication wrapper remains image-compatible. It is created on demand;
there is no long-running plugin-admin resource to apply during this rollout.
