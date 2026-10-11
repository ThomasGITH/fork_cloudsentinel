# Generic Monitoring History

> This document records the HISTORY-1 deployment. For the current image tags
> and deployment sequence after automatic RCA restoration, use
> [`generic-monitoring-rca.md`](generic-monitoring-rca.md).

`GENERIC-MONITORING-HISTORY-1` restores the existing monitoring screen as a
master-detail Monitoring History view while retaining explicit Saved Model
selection, immutable live-input recipes and Generic Detection Runtime.
Automatic CausalRCA was disconnected in this history tranche. It is reconnected
additively by `GENERIC-MONITORING-RCA-1`; see
[`generic-monitoring-rca.md`](generic-monitoring-rca.md). Existing sessions stay
RCA-disabled.

## Architecture

The Data Ingestion API creates a durable session before dispatching the first
Celery task. Redis contains only temporary execution state and the bounded
per-session context buffer. Durable records are stored under
`MONITORING_HISTORY_STORAGE_ROOT`, which is `/app/storage/monitoring_history`
in Kubernetes and therefore uses the existing `data-catalogue-pvc`.

The API and worker both mount that claim read-write. This is appropriate for
the current single-node, one-worker staging topology. No database, service or
new PVC is introduced.

The private layout is:

```text
monitoring_history/
├── artifacts/
│   ├── input/sha256_<digest>.json
│   └── result/sha256_<digest>.json
└── sessions/<session_id>/
    ├── session.json
    ├── recipe.json
    └── cycles/<cycle_id>/cycle.json
```

Input and result artifacts are content-addressed. An identical artifact is
written once. `cycle.json` contains logical references and checksums, and its
directory is published by atomic rename. `session.json` is atomically updated
only after the cycle exists. Temporary cycle directories are ignored after a
crash. Content blobs left by a crash before cycle publication are harmless;
automatic orphan collection is not implemented.

## Cycle and watermark semantics

Each task invocation has four different time concepts:

- **query window**: requested Prometheus start and end time;
- **inference window**: the complete bounded context buffer sent to the model;
- **prediction window**: prediction timestamps newer than the committed
  watermark;
- **processing time**: task start, inference completion and commit time.

The Generic Detection Runtime is called exactly once through
`POST /internal/evaluate`. Its `warmup_observations` aligns predictions and
scores to `input_timestamps[warmup_observations:]`. Lengths, order, binary
labels and finite scores are validated before commit.

On the first successful cycle, every aligned prediction timestamp is eligible.
On subsequent cycles only timestamps strictly newer than
`prediction_watermark` are eligible. The watermark advances to the newest
eligible timestamp only after input, result and cycle records are committed.
Retries of the same deterministic cycle ID return the existing cycle and do
not increment history twice.

Prometheus samples at or before the watermark are frozen. Late samples and
revisions for that range are ignored and reported as warnings; published
history is never silently rewritten. Overlapping uncommitted samples may be
refreshed. Samples newer than the watermark are evaluated once when first
committed.

Cycle metrics use only newly eligible predictions:

```text
prediction_count  = eligible prediction timestamps
anomaly_count     = binary predictions equal to 1
anomaly_percentage = anomaly_count / prediction_count * 100
```

The context-buffer size is never the denominator. When no prediction is
eligible, `anomaly_percentage` is `null`; the UI shows warm-up or no-new-data
instead of 0%.

## Durable provenance and privacy

The session stores model ID, detector metadata, artifact-manifest hash,
recipe/feature-order hashes, an immutable recipe snapshot, timing
configuration and lifecycle timestamps. Every committed cycle stores exact
inference input, timestamps, feature order, normalized predictions, scores,
aligned prediction timestamps, safe warnings and checksums.

The private artifacts are required because a recipe cannot reconstruct old
input after Prometheus retention expires. Public APIs never return the matrix,
PromQL, predictions, scores, internal paths, task IDs, service URLs or
tracebacks. They expose only safe summaries and checksums.

## API

- `POST /start_monitoring`: accepts `model_id`, optional `display_name`,
  `window_seconds` and `poll_interval_seconds`.
- `GET /monitoring-sessions?page=1&page_size=20`: paginated durable sessions.
- `GET /monitoring-sessions/<session_id>?cycle_page=1&cycle_page_size=50`:
  safe session provenance and paginated cycle summaries.
- `DELETE /stop_monitoring/<session_id>`: durably stops one session.
- `GET /get_active_tasks`: bounded compatibility projection for older callers.

The Django routes proxy these APIs without exposing internal service URLs.

## Restart, stop and failure behavior

Durable history remains readable when Redis data expires or is lost. An
otherwise active historical session without Redis execution state is shown as
`interrupted`; it is not resumed automatically.

Stopping writes durable state first. A late worker cannot publish a cycle once
the durable session is stopped. Revocation remains best effort: inference
already executing may finish computationally, but its result cannot commit or
reactivate the session.

An inference failure commits a safe failed cycle when input was already
collected, does not advance the watermark and marks the session failed. Failed
or dispatch-failed sessions do not automatically restart.

## Dashboard states

The dashboard keeps the selected session in `?session=<session_id>` during
polling and browser refresh. It shows:

- active, collecting, queued, warm-up, stopped, failed and interrupted state;
- stale warning after two poll intervals without a successful commit;
- no-new-data without treating it as 0%;
- safe warnings/errors;
- one chart point per committed cycle with a non-null percentage;
- paginated sessions and cycle pages;
- expandable checksum provenance.

Chart values are per newly evaluated cycle and are not cumulative incident
counts. Polling errors leave the existing selection and chart visible.

## Storage growth and retention

Storage grows with the exact bounded matrix and prediction output of every
cycle. Content addressing avoids duplicate identical payloads, but overlapping
buffers that differ still consume additional space. The worker enforces the
existing observation/feature bounds.

There is no automatic retention or cleanup in this tranche. Operators must
monitor the 5 GiB `data-catalogue-pvc`. Do not remove session directories or
artifacts while history is in use. A later retention feature should delete
unreferenced content only after reference analysis.

Sessions created before this tranche remain Redis-only and are not silently
backfilled into durable history. New sessions use durable history from their
creation onward. Existing datasets and model records are not changed.

## Build and deploy on EC2/Minikube

Run from the repository root. These are the changed images:

```bash
docker build \
  -f data_ingestion/Dockerfile \
  -t jojojochem/data_ingestion:generic-monitoring-history-1 \
  data_ingestion

docker build \
  -f learning_adaptation/Dockerfile \
  -t jojojochem/learning_adaptation:generic-monitoring-history-1 \
  .

docker build \
  -f user_interface/monitoring_project/Dockerfile \
  -t jojojochem/monitoring_project:generic-monitoring-history-1 \
  user_interface/monitoring_project
```

The Generic Detection Runtime is unchanged and remains
`jojojochem/anomaly_detection_generic:generic-live-monitoring-1.1`.
Confirm that this previously built image is still present:

```bash
minikube image ls | grep 'jojojochem/anomaly_detection_generic:generic-live-monitoring-1.1'
```

If it is absent, use the existing CPU-only build documented in
`docs/generic-detection-deployment.md`; this tranche does not change that image.

Load the new images:

```bash
minikube image load --overwrite=true \
  jojojochem/data_ingestion:generic-monitoring-history-1

minikube image load --overwrite=true \
  jojojochem/learning_adaptation:generic-monitoring-history-1

minikube image load --overwrite=true \
  jojojochem/monitoring_project:generic-monitoring-history-1
```

The manifests use `imagePullPolicy: IfNotPresent`; the exact tagged images must
exist inside Minikube before rollout. Apply in this order:

```bash
kubectl apply -f k8s/namespace-deployment.yml
kubectl apply -f k8s/data_catalogue-pvc.yml
kubectl apply -f k8s/learning_adaptation-pvc.yml
kubectl apply -f k8s/model_artifact-pvc.yml
kubectl apply -f k8s/detection_results-pvc.yml
kubectl apply -f k8s/detector-plugin-repository-pvc.yml
kubectl apply -f k8s/generic_anomaly_detection-service.yml
kubectl apply -f k8s/generic_anomaly_detection-deployment.yml
kubectl apply -f k8s/learning_adaptation-deployment.yml
kubectl apply -f k8s/learning_adaptation-celery-deployment.yml
kubectl apply -f k8s/data_ingestion-deployment.yml
kubectl apply -f k8s/data_ingestion_celery-deployment.yml
kubectl apply -f k8s/monitoring_project-deployment.yml
```

Redis and the existing catalogue, learning, artifact, result and plugin claims
must already be healthy. Reapplying the PVC manifests is non-destructive and
does not recreate or erase bound volumes. The temporary plugin-admin pod is not
part of this rollout; its manifest now references the same new Learning image
for the next operator-driven plugin publication.

Check rollout and images:

```bash
kubectl rollout status -n cloudsentinel deployment/data-ingestion-deployment
kubectl rollout status -n cloudsentinel deployment/celery-worker-deployment
kubectl rollout status -n cloudsentinel deployment/generic-anomaly-detection-deployment
kubectl rollout status -n cloudsentinel deployment/learning-adaptation-deployment
kubectl rollout status -n cloudsentinel deployment/learning-adaptation-celery-deployment
kubectl rollout status -n cloudsentinel deployment/monitoring-project-deployment

kubectl get pods -n cloudsentinel \
  -o custom-columns='POD:.metadata.name,IMAGE:.spec.containers[*].image,READY:.status.containerStatuses[*].ready'

kubectl logs -n cloudsentinel deployment/celery-worker-deployment --tail=100
```

The UI remains available through the existing NodePort. A local alternative is:

```bash
kubectl port-forward -n cloudsentinel \
  service/monitoring-project-service 8000:80
```

Open `http://127.0.0.1:8000/monitoring-overview/`.

Non-destructive rollback changes only image references:

```bash
kubectl set image -n cloudsentinel deployment/data-ingestion-deployment \
  data-ingestion=jojojochem/data_ingestion:generic-live-monitoring-1

kubectl set image -n cloudsentinel deployment/celery-worker-deployment \
  celery-worker=jojojochem/data_ingestion:generic-live-monitoring-1

kubectl set image -n cloudsentinel deployment/learning-adaptation-deployment \
  learning-adaptation=jojojochem/learning_adaptation:generic-live-monitoring-1

kubectl set image -n cloudsentinel deployment/learning-adaptation-celery-deployment \
  learning-adaptation-celery=jojojochem/learning_adaptation:generic-live-monitoring-1

kubectl set image -n cloudsentinel deployment/monitoring-project-deployment \
  monitoring-project=jojojochem/monitoring_project:generic-live-monitoring-1.1
```

Rollback does not delete Monitoring History. Older code ignores its directory.

## Live acceptance procedure

1. Open a Saved Model with `inference.status=ready` and
   `live_monitoring.status=ready`; choose **Use for live monitoring**.
2. Give the session a recognizable name and start it. Confirm the exact model
   remains selected and no monitoring starts before submitting.
3. Open Monitoring History. Confirm the new session is selected and moves
   through queued/collecting or warm-up to active.
4. Wait for two successful polls. Confirm each new timestamp contributes once,
   the text uses `anomaly_count / prediction_count`, and the chart gains one
   point only when a percentage exists.
5. Use a window longer than the poll interval. Confirm overlapping timestamps
   do not inflate the next cycle's `prediction_count`.
6. Stop the session. Confirm status remains stopped and no later point appears.
7. Restart only the Data Ingestion API pod and reopen the session. Confirm
   history and chart remain. Delete a test Redis session key only in a disposable
   staging validation if desired; the UI must show interrupted history rather
   than active execution. Automatic resume is intentionally absent.
8. Repeat with IF or LOF (warm-up 0) and CGNN (non-zero warm-up). Confirm the
   same UI and API work without detector-specific dispatch.
9. From Saved Model detail and Comparison result, choose **Use for live
   monitoring**. Confirm the correct `model_id` is preselected and the user must
   still explicitly start.

Existing models with a valid stored recipe remain usable; no fresh fetch or
training is required. Older offline-only models stay unavailable for live
monitoring and are not backfilled.

## Troubleshooting

- **Interrupted:** Redis execution state is absent; history is intact and the
  session will not auto-resume.
- **No new observations:** every aligned prediction timestamp was at or before
  the committed watermark.
- **Warm-up:** the adapter returned no predictions beyond model context yet.
- **Missing values:** the pinned policy rejects the window; no imputation occurs.
- **History unavailable:** check the `data-catalogue-pvc` mount and permissions
  on both Data Ingestion deployments.
- **Stale:** inspect the data-ingestion Celery worker and Generic Detection
  Runtime logs; the UI keeps the last committed chart.

The original history contract remains valid. New sessions can additionally opt
into the recipe-compatible, durable CausalRCA flow documented in
[`generic-monitoring-rca.md`](generic-monitoring-rca.md).
