# Selectable monitoring RCA pipeline

This document describes the hardened selectable RCA path introduced by
`EXPAND-RCA-1` through `EXPAND-RCA-3` and the subsequent end-to-end review.
It covers monitoring-triggered RCA only.

## Architecture

The operator explicitly selects one method while creating a monitoring
session. The Django UI sends the method ID to Data Ingestion. Data Ingestion
validates the request, revalidates the selected method against the CRCA
runtime capability response, and pins the method and configuration digest in
the durable session state. Trigger, cooldown, episode and reanalysis policy
remain generic.

For an accepted trigger, Data Ingestion selects input from the immutable
committed monitoring-cycle artifact and creates a durable method-pinned RCA
intent. Its stable identity includes the method, method configuration,
analysis window, immutable model and recipe identities, input checksum, and
change-point evidence where required. The CRCA API dispatches through an
explicit method registry, and the worker repeats the pinned contract and
configuration checks before invoking the corresponding adapter. Results are
written to durable monitoring history before the UI can read them.

An RCA dispatch or execution failure is stored on that RCA run. It does not
fail or stop anomaly-detection monitoring.

## Methods and contracts

| Method | Input | Preprocessing | Output |
| --- | --- | --- | --- |
| `cloudsentinel.causal-rca/legacy-v1` | `cloudsentinel.crca-input/v1` | `cloudsentinel.crca-minmax/v1` | Ranked features and, when produced, a dependency graph |
| `cloudsentinel.baro-robust-scorer/v1` | `cloudsentinel.baro-input/v1` | `cloudsentinel.baro-robust-scorer-raw/v1` | Ranked features plus detector-derived change-point evidence; no causal graph |

CausalRCA keeps its existing per-run `MinMaxScaler` preprocessing. BARO never
uses that transformation. The BARO adapter passes finite raw metric values to
the official RobustScorer component, which applies its robust baseline
normalisation. Feature IDs remain the immutable matrix-column identities for
both methods.

BARO's change point is the timestamp of the first positive stored detector
prediction inside the selected RCA analysis window. It comes from the
checksummed committed monitoring-cycle result, never from wall-clock time or
an arbitrary midpoint. Rows before the point form the baseline; rows at and
after it form the anomalous segment. Missing positive predictions or too few
rows on either side produce a preflight skip and consume no execution attempt.

## Compatibility and backward compatibility

The runtime capability for a selected method must advertise its exact method
ID, input contract, preprocessing contract, bounded minimum requirements and a
valid configuration digest. Data Ingestion rejects inconsistent capability
metadata before session creation. The worker checks the pinned input and
preprocessing contracts again.

Existing configurations, sessions and runs without a method continue to
resolve to `cloudsentinel.causal-rca/legacy-v1`. Already queued legacy
CausalRCA input may omit the newer explicit input/preprocessing fields. An
explicit unknown method never silently falls back.

## Privacy and result validation

Private matrices, PromQL, artifact references, internal URLs, adapter details
and full method configuration never appear in public monitoring responses.
Public run and result objects use nested allowlists. Rankings retain only the
bounded rank, exact feature/column identity, optional safe entity metadata,
metric name and finite score. The UI creates result elements with `textContent`.

Both adapters reject unknown features, invalid or duplicate ranking entries,
non-finite values and inconsistent column identities. BARO additionally
rejects invalid change-point provenance. Scores are relative diagnostic values
within one run; they are neither probabilities nor comparable across methods.

## BARO attribution

CloudSentinel integrates only the `RobustScorer` component from
`fse-baro==0.2.2`; it does not integrate BARO's BOCPD change-point detector.
The dependency is installed from an exact wheel hash without dependency
resolution. The bundled MIT notice and attribution are recorded in
`THIRD_PARTY_NOTICES.md`. The upstream package and archived release are:

- <https://pypi.org/project/fse-baro/>
- <https://zenodo.org/records/15635955>

## Known limits and supported claims

- A ranking is diagnostic evidence, not proof of the actual root cause.
- BARO ranks robust changes around an anomaly-detector-derived point; it does
  not establish causality and produces no causal graph.
- CausalRCA's graph is dependency-oriented evidence and is not proof of cause.
- The current BARO minimum of two baseline and two post-change observations is
  a technical adapter minimum, not a statistical-quality guarantee.
- Persistent queued/running worker intents still require operator diagnosis if
  a worker hangs; the pipeline does not create a parallel replacement run.
- High anomaly percentages may reflect detector behaviour and are not, by
  themselves, validated incident detection.

## Build and Minikube rollout

Run from the repository root on the EC2 host. These are the only changed
images for this tranche:

```bash
docker build \
  -t jojojochem/data_ingestion:expand-rca-hardening-1 \
  data_ingestion

docker build \
  -f anomaly_detection/crca/Dockerfile \
  -t jojojochem/anomaly_detection_crca:expand-rca-hardening-1 \
  .

docker build \
  -t jojojochem/monitoring_project:expand-rca-hardening-1 \
  user_interface/monitoring_project
```

Load the locally built images into Minikube. The manifests use
`imagePullPolicy: IfNotPresent` and the exact immutable tags below.

```bash
minikube image load jojojochem/data_ingestion:expand-rca-hardening-1
minikube image load jojojochem/anomaly_detection_crca:expand-rca-hardening-1
minikube image load jojojochem/monitoring_project:expand-rca-hardening-1
```

Apply the existing manifests. No storage migration or destructive PVC action
is required.

```bash
kubectl apply -f k8s/crca_anomaly_detection-deployment.yml
kubectl apply -f k8s/crca_anomaly_detection-celery-deployment.yml
kubectl apply -f k8s/data_ingestion-deployment.yml
kubectl apply -f k8s/data_ingestion_celery-deployment.yml
kubectl apply -f k8s/monitoring_project-deployment.yml
```

Check every rollout and its selected image:

```bash
kubectl rollout status deployment/crca-anomaly-detection-deployment -n cloudsentinel
kubectl rollout status deployment/crca-anomaly-detection-celery-deployment -n cloudsentinel
kubectl rollout status deployment/data-ingestion-deployment -n cloudsentinel
kubectl rollout status deployment/celery-worker-deployment -n cloudsentinel
kubectl rollout status deployment/monitoring-project-deployment -n cloudsentinel

kubectl get deployment -n cloudsentinel \
  crca-anomaly-detection-deployment \
  crca-anomaly-detection-celery-deployment \
  data-ingestion-deployment \
  celery-worker-deployment \
  monitoring-project-deployment \
  -o custom-columns='NAME:.metadata.name,IMAGE:.spec.template.spec.containers[0].image,READY:.status.readyReplicas'
```

For local HTTP inspection on the EC2 instance:

```bash
kubectl port-forward -n cloudsentinel service/monitoring-project-service 30000:80
```

## Manual acceptance flow

1. Open Monitoring Setup and select a live-ready Saved Model.
2. Enable automatic RCA. Confirm both available methods show their input and
   output explanation and that incompatible methods are disabled with a safe
   reason.
3. Select **CausalRCA**, retain a conservative threshold/window/cooldown, and
   start a session. Produce one eligible threshold episode. Confirm the durable
   run reports the CausalRCA method, a bounded ranking and a graph only when the
   algorithm produced one. Confirm monitoring continues if the RCA run fails.
4. Stop that monitoring session. Create a new session for the same live-ready
   model and select **BARO RobustScorer**. Produce an episode whose committed
   detector result has a positive prediction with enough stored observations
   before and after it. Confirm the result shows the stored change point,
   baseline/anomalous counts, a ranking and no graph.
5. For both sessions, refresh the page and revisit Monitoring History. Confirm
   method, run state, ranking and provenance remain available.
6. Inspect browser/API responses and confirm they contain no matrix, PromQL,
   storage path, internal service URL, traceback or private method
   configuration.
7. Confirm sustained high percentages still obey episode, cooldown,
   reanalysis and budget settings independently of method.

## Non-destructive rollback

Set the five deployments back to the previously deployed immutable tags in
the manifests and apply them again, or use rollout undo while the prior
ReplicaSet remains available:

```bash
kubectl rollout undo deployment/crca-anomaly-detection-deployment -n cloudsentinel
kubectl rollout undo deployment/crca-anomaly-detection-celery-deployment -n cloudsentinel
kubectl rollout undo deployment/data-ingestion-deployment -n cloudsentinel
kubectl rollout undo deployment/celery-worker-deployment -n cloudsentinel
kubectl rollout undo deployment/monitoring-project-deployment -n cloudsentinel
```

Rollback does not delete or rewrite monitoring history, RCA results, models,
recipes or PVC content. New BARO records remain durable but older UI/runtime
versions may not render their additive fields.
