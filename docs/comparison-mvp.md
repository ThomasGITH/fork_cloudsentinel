# Anomaly Detection Comparison MVP

COMPARISON-MVP-1 compares two or more immutable Saved Model artefacts on the
same Data Catalogue evaluation partition. It supports anomaly detection only.
Root-cause analysis, remediation, exports, robustness scoring and model
training are outside this feature.

## Execution and persistence

The learning-adaptation API stores each `ComparisonRun` atomically under
`COMPARISON_STORAGE_ROOT/<comparison-id>/comparison.json`. A submit request
captures the exact dataset version, partition checksum, feature-order hash,
model IDs and model artifact-manifest hashes before dispatching one generic
Celery task.

The worker obtains a checksummed `metrics-evaluation/v1` ZIP from Data
Catalogue. It contains the partition's evaluation matrix, optional labels,
optional timestamps and a manifest. The worker never persists a second copy of
the raw matrix. For each selected model it:

1. activates the immutable Saved Model in Generic Detection Runtime;
2. calls the internal, bounded `/internal/evaluate` endpoint by `model_id`;
3. verifies the returned artifact-manifest hash;
4. calculates metrics and bounded timeline data; and
5. writes the child result atomically.

The comparison is `completed` when all models succeed, `partial` when at least
one succeeds and one fails, and `failed` when none succeeds. Completed,
partial and failed records are immutable. One model failure does not prevent
the remaining models from being evaluated.

## API

Learning adaptation exposes:

- `GET /api/comparisons` with `search`, `status`, `modality`, `workload`,
  `page`, and `page_size`;
- `POST /api/comparisons`;
- `GET /api/comparisons/<comparison_id>`;
- `GET /api/comparisons/<comparison_id>/status`;
- `GET /api/comparisons/evaluation-datasets`;
- `GET /api/comparisons/compatible-models?dataset_id=...&version=...&partition_id=...`.

Example submit body:

```json
{
  "name": "CPU stress comparison",
  "evaluation_dataset": {
    "dataset_id": "ds_example",
    "version": 4,
    "partition_id": "partition_example"
  },
  "model_ids": ["model_one", "model_two"],
  "client_request_id": "stable-browser-generated-value"
}
```

The idempotency value prevents duplicate submit. The server revalidates that
every model is available, uses `cloudsentinel.inference/v1`, has complete
immutable artifact metadata, matches the evaluation modality and has the exact
same feature-order hash. Compatibility never branches on detector ID.

Data Catalogue additionally exposes the internal streamed transport endpoint:

```text
GET /datasets/<dataset_id>/versions/<version>/partitions/<partition_id>/evaluation-bundle
```

Generic Detection Runtime additionally exposes the internal bounded endpoint:

```text
POST /internal/evaluate
```

Neither internal endpoint is called by browser code. Public responses contain
no filesystem paths, adapter entry points, task IDs, service URLs, model bytes
or stack traces.

## Metrics and time semantics

Labels always use `0 = normal` and `1 = anomaly`. With ground truth, the worker
stores precision, recall, F1 and TP/FP/TN/FN. Without ground truth those fields
are absent; the UI states that they are unavailable.

The first detection is the timestamp of the first predicted anomaly after any
model warmup. Detection lead time is:

```text
known incident start - first detected anomaly timestamp
```

A non-negative result means detection occurred before or at incident start. A
detection after incident start has no positive lead-time value and is shown as
"Detection occurred after incident start". Missing timestamps, missing incident
context and no detection are represented explicitly, never as a fabricated
zero.

Timeline data is sampled to `COMPARISON_MAX_TIMELINE_POINTS` per model. Full
prediction and score arrays stay between the worker and the internal runtime
for the duration of the task and are not exposed to the Django browser client.

## Django UI

The server-rendered UI provides:

- `/comparison/` for Past comparisons;
- `/comparison/new/` for the three-step dataset, model and review flow;
- `/comparison/<comparison_id>/` for Overview, Timeline, Model details and the
  honest Robustness fallback;
- `/comparison/<comparison_id>/status/` as the safe polling proxy.

The wizard signs its state with Django signing. It permits datasets without
labels while warning that accuracy and lead-time metrics may be unavailable.
At least two compatible Saved Models are required. Polling runs only for queued
or running comparisons, uses non-overlapping recursive requests and stops on a
terminal state or page navigation.

## Staging deployment

Build from the repository root:

```bash
docker build -f data_ingestion/Dockerfile \
  -t jojojochem/data_ingestion:comparison-mvp-1 data_ingestion
docker build -f learning_adaptation/Dockerfile \
  -t jojojochem/learning_adaptation:comparison-mvp-1.2 .
docker build -f anomaly_detection/generic/Dockerfile \
  -t jojojochem/anomaly_detection_generic:comparison-mvp-1 .
docker build -f user_interface/monitoring_project/Dockerfile \
  -t jojojochem/monitoring_project:comparison-mvp-1.1 \
  user_interface/monitoring_project
```

For a local Minikube Docker driver, load the four exact tags:

```bash
minikube image load jojojochem/data_ingestion:comparison-mvp-1
minikube image load --overwrite=true jojojochem/learning_adaptation:comparison-mvp-1.2
minikube image load jojojochem/anomaly_detection_generic:comparison-mvp-1
minikube image load --overwrite=true jojojochem/monitoring_project:comparison-mvp-1.1
```

Apply the already-provisioned storage and changed workloads in this order:

```bash
kubectl apply -f k8s/data_ingestion-deployment.yml
kubectl apply -f k8s/data_ingestion_celery-deployment.yml
kubectl apply -f k8s/learning_adaptation-deployment.yml
kubectl apply -f k8s/learning_adaptation-celery-deployment.yml
kubectl apply -f k8s/generic_anomaly_detection-deployment.yml
kubectl apply -f k8s/generic_anomaly_detection-service.yml
kubectl apply -f k8s/monitoring_project-deployment.yml
```

Then verify rollouts:

```bash
kubectl rollout status -n cloudsentinel deployment/data-ingestion-deployment
kubectl rollout status -n cloudsentinel deployment/celery-worker-deployment
kubectl rollout status -n cloudsentinel deployment/learning-adaptation-deployment
kubectl rollout status -n cloudsentinel deployment/learning-adaptation-celery-deployment
kubectl rollout status -n cloudsentinel deployment/generic-anomaly-detection-deployment
kubectl rollout status -n cloudsentinel deployment/monitoring-project-deployment
```

Deployment names should be checked with `kubectl get deploy -n cloudsentinel`
if a staging cluster still uses an older data-ingestion worker name.

### Learning worker reports a missing comparison module

If the worker receives `tasks.execute_comparison_task` but reports
`No module named 'comparison_execution'`, the worker accepted the task without
having loaded its executor. In `comparison-mvp-1.2` the executor is imported
during worker startup, so an incomplete worker cannot become ready. The
Dockerfile also fails its build when the task and its runtime modules cannot be
imported together. Build and load that Learning image, then apply both Learning
manifests. Verify every matching worker pod before retrying:

```bash
WORKER_POD="$(kubectl get pod -n cloudsentinel \
  -l app=learning-adaptation-celery \
  -o jsonpath='{.items[0].metadata.name}')"

kubectl get pods -n cloudsentinel -l app=learning-adaptation-celery \
  -o custom-columns='NAME:.metadata.name,IMAGE:.spec.containers[0].image,IMAGE_ID:.status.containerStatuses[0].imageID,PHASE:.status.phase'

kubectl exec -n cloudsentinel "$WORKER_POD" -- sh -c \
  'ls -l /app/comparison_*.py && python -c "import tasks; print(tasks.execute_comparison.__module__); assert '\''tasks.execute_comparison_task'\'' in tasks.celery.tasks"'
```

There must be exactly one active Learning worker for the staging setup. If
another old worker consumes the same Redis queue, remove or scale down that old
deployment before submitting a new comparison.

A comparison whose task already crashed before importing the executor remains
`queued`; create a new comparison after the corrected worker is running.

## Live acceptance flow

1. Confirm Data Catalogue, learning adaptation and generic runtime health.
2. Use an available Data Catalogue version with an explicit partition and at
   least two `inference.status=ready` Saved Models sharing its feature hash.
3. Open **Comparison**, choose **New comparison**, select the evaluation
   dataset, then two compatible models.
4. Submit once and confirm the detail page moves from queued/running to
   completed or partial without revealing a task ID.
5. For labelled data, independently verify TP/FP/TN/FN and derived metrics on a
   small known fixture. For unlabelled data, verify that precision, recall and
   F1 are absent and the warning is shown.
6. Confirm the Timeline shows only bounded points and Model details links to
   the existing Saved Model provenance pages.
7. Restart the learning API and worker separately and confirm the terminal
   comparison remains readable from shared learning storage.

## Known MVP limits

- Only the active explicit partition of an available dataset version is listed.
- Evaluation is bounded by the generic runtime request limits and uses one
  Celery worker task; there is no distributed fan-out per model.
- The UI displays detector version because Saved Models do not yet have a
  separate user-managed model version or display-name field.
- Timeline scores from different detector types may have different numerical
  scales; they are overlaid for temporal inspection, not absolute calibration.
- Robustness aggregation across workload contexts is intentionally absent.
- No comparison deletion, cancellation, export, RCA, remediation or automatic
  action is implemented.
