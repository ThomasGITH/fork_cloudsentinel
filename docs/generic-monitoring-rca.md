# Generic monitoring CausalRCA

`GENERIC-MONITORING-RCA-1` reconnects the existing CausalRCA implementation to
recipe-pinned generic monitoring. It does not change anomaly-detector training,
scalers, thresholds or prediction semantics. An anomaly percentage, including a
repeated 100% result, remains a detector signal whose correctness must be
investigated separately.

## Trigger and episode semantics

Automatic RCA is disabled by default. Existing session records do not contain
the new configuration and are therefore treated as disabled.

A new session can explicitly configure:

- threshold: `0..100%`, default `50%`;
- analysis window: `60..86400` seconds, default 600 seconds;
- cooldown: `0..604800` seconds, default 3600 seconds.

The trigger is strictly:

```text
committed cycle anomaly_percentage > configured threshold
```

It uses only newly eligible predictions in that committed cycle. An equal
percentage does not trigger. A warm-up or no-new-data cycle has no percentage,
does not trigger and does not recover an episode.

The first crossing while armed creates one durable run intent. Continued high
cycles are recorded as suppressed for the same episode. A valid cycle at or
below the threshold records recovery. If cooldown has already expired, this
re-arms immediately. Recovery during cooldown enters `recovered_cooldown`. A
new high episode during that period is suppressed and enters
`cooldown_episode`; persistent high values never cause a delayed run. A later
valid cycle at or below the threshold after cooldown re-arms the session. This
conservative policy prevents trigger storms.

Stopping monitoring prevents newly evaluated trigger intents and new dispatch
attempts. A task submitted before the stop may finish and persist its result;
it cannot resume monitoring.

## Compatibility and naming boundary

AD monitoring and RCA readiness are separate. A live-ready model can always be
monitored with RCA disabled.

The RCA gate accepts only recipes where:

- at least two feature IDs exist and are unique;
- every feature has exactly one `series_mapping` entry;
- the referenced query execution explicitly declares `pod` in
  `identity_labels`;
- `metric_labels.pod` and the query's `feature_name` are present;
- each `(pod, metric)` pair is unique;
- stored column order and feature metadata match exactly;
- the selected matrix has at least two observations and two features and all
  values are finite.

No pod or metric is parsed from an arbitrary feature string. The CRCA boundary
now accepts explicit `feature_names`; old callers can still use the legacy
pods-times-metrics naming convention. Ranking rows and graph labels correspond
exactly to matrix columns.

## Windows and immutable input

The three windows have different meanings:

- **AD trigger window:** the newly eligible prediction timestamps summarized by
  the committed monitoring cycle;
- **AD inference/context window:** the complete bounded matrix passed once to
  Generic Detection Runtime;
- **RCA analysis window:** rows selected from that exact committed input,
  ending at the last newly eligible prediction timestamp and extending back by
  the configured duration.

The implementation reads the private, checksummed cycle input artifact. It
does not query Prometheus again, use the legacy collector, impute values, drop
features or infer a mapping. Insufficient context produces a durable `skipped`
run and requires recovery before another episode can trigger.

CausalRCA applies a separately pinned per-column MinMax transformation
(`cloudsentinel.crca-minmax/v1`). This does not touch the anomaly detector's
training or live preprocessing. The existing causal algorithm is identified as
`cloudsentinel.causal-rca/legacy-v1`; its complete configuration and checksum
are captured privately with the completed result.

## Durable lifecycle and recovery

Monitoring history remains under `MONITORING_HISTORY_STORAGE_ROOT`:

```text
monitoring_history/
├── artifacts/
│   ├── input/
│   ├── result/
│   ├── rca_input/
│   └── rca_result/
└── sessions/<session_id>/
    ├── session.json
    ├── recipe.json
    ├── cycles/<cycle_id>/cycle.json
    ├── rca_events/<cycle_id>.json
    └── rca_runs/<run_id>/run.json
```

The stable RCA run ID covers session, cycle, selected input checksum and pinned
RCA configuration identity. Creation and state changes use the existing file
lock and atomic JSON replacement. A cycle retry finds the same event/run.

The sequence is:

```text
monitoring cycle committed
→ durable trigger event and RCA intent
→ idempotent internal dispatch with stable Celery task ID
→ worker atomically claims queued run
→ worker reads checksummed RCA input
→ pinned preprocessing and existing CausalRCA execution
→ result validation
→ durable checksummed ranking/graph result
```

An HTTP dispatch failure leaves the intent `queued` with
`dispatch_status=pending`; a later active monitoring cycle retries that same
intent. `POST /monitoring-sessions/<session_id>/rca-reconcile` offers an
explicit non-destructive retry while a session remains active. A crash after a
worker claim can leave a run `running`; there is no background lease or
automatic resume in this tranche, so an operator must inspect/reconcile it.
This is the remaining delivery limitation. The task catches algorithm failures
and durably marks the run failed, preventing Celery retry storms.

RCA failure never changes the AD session lifecycle. Redis and Celery result
expiry do not remove committed inputs, trigger events, run state or results.

The Data Ingestion API/worker and CRCA worker share the existing
`data-catalogue-pvc`. The CRCA worker mounts `/app/storage` read-write because
it commits run completion. The CRCA API does not need this mount. No new PVC or
service is introduced.

There is no automatic retention or garbage collection. RCA matrices and PNG
graphs add storage per triggered episode. Operators must monitor PVC use and
retain or archive session directories as an explicit maintenance operation;
files must not be deleted while their session is in use.

## Safe APIs and UI

Data Ingestion provides:

- `GET /monitoring-rca/readiness/<model_id>` — safe recipe compatibility;
- `GET /monitoring-sessions/<session_id>` — cycles plus paginated RCA summaries;
- `GET /monitoring-sessions/<session_id>/rca-runs/<run_id>` — bounded ranking
  (maximum 100 rows), at most three validated PNG graphs, safe provenance;
- `POST /monitoring-sessions/<session_id>/rca-reconcile` — retry pending durable
  dispatch intents for an active session.

CRCA provides internal-only:

- `GET /internal/monitoring-capability`;
- `POST /internal/monitoring-runs` with only `session_id` and `run_id`.

Responses omit raw matrices, PromQL, internal paths and URLs, Celery IDs,
tracebacks and private task details. The dashboard shows RCA enabled/readiness,
threshold and windows, a threshold line, trigger markers, run status, stored
ranking, stored graph and safe checksum provenance. It distinguishes started,
suppressed, skipped and incompatible states. Rankings are described as
diagnostic evidence and never as a proven root cause.

## Build, load and apply (do not run from this document automatically)

Run from the repository root on the EC2 host. The CRCA Dockerfile now requires
the repository root as its build context because it copies the shared durable
history module.

```bash
docker build \
  -t jojojochem/data_ingestion:generic-monitoring-rca-1 \
  ./data_ingestion

docker build \
  -t jojojochem/monitoring_project:generic-monitoring-rca-1 \
  ./user_interface/monitoring_project

docker build \
  -f anomaly_detection/crca/Dockerfile \
  -t jojojochem/anomaly_detection_crca:generic-monitoring-rca-1 \
  .
```

Load the three changed images into Minikube:

```bash
minikube image load jojojochem/data_ingestion:generic-monitoring-rca-1
minikube image load jojojochem/monitoring_project:generic-monitoring-rca-1
minikube image load jojojochem/anomaly_detection_crca:generic-monitoring-rca-1
```

The manifests use `imagePullPolicy: IfNotPresent`, which is required for these
locally loaded tags. Apply the existing storage-backed services first, then API,
workers and UI:

```bash
kubectl apply -f k8s/data_ingestion-deployment.yml
kubectl apply -f k8s/crca_anomaly_detection-deployment.yml
kubectl apply -f k8s/data_ingestion_celery-deployment.yml
kubectl apply -f k8s/crca_anomaly_detection-celery-deployment.yml
kubectl apply -f k8s/monitoring_project-deployment.yml
```

Check rollouts:

```bash
kubectl rollout status deployment/data-ingestion-deployment -n cloudsentinel
kubectl rollout status deployment/celery-worker-deployment -n cloudsentinel
kubectl rollout status deployment/crca-anomaly-detection-deployment -n cloudsentinel
kubectl rollout status deployment/crca-anomaly-detection-celery-deployment -n cloudsentinel
kubectl rollout status deployment/monitoring-project-deployment -n cloudsentinel

kubectl get pods -n cloudsentinel
kubectl logs -n cloudsentinel deployment/celery-worker-deployment --tail=100
kubectl logs -n cloudsentinel deployment/crca-anomaly-detection-celery-deployment --tail=100
```

Optional local checks:

```bash
kubectl port-forward -n cloudsentinel service/data-ingestion-service 5001:80
kubectl port-forward -n cloudsentinel service/crca-anomaly-detection-service 5023:80
kubectl port-forward -n cloudsentinel service/monitoring-project-service 8000:80
```

Then, in separate terminals:

```bash
curl --show-error --fail http://127.0.0.1:5001/healthz | jq
curl --show-error --fail http://127.0.0.1:5023/healthz | jq
curl --show-error --fail http://127.0.0.1:5023/internal/monitoring-capability | jq
```

The Learning API/worker, Generic Detection Runtime, data-processing service and
legacy CGNN runtime images are unchanged in this tranche.

For a non-destructive rollback, retain the PVC and roll back only deployments:

```bash
kubectl rollout undo deployment/data-ingestion-deployment -n cloudsentinel
kubectl rollout undo deployment/celery-worker-deployment -n cloudsentinel
kubectl rollout undo deployment/crca-anomaly-detection-deployment -n cloudsentinel
kubectl rollout undo deployment/crca-anomaly-detection-celery-deployment -n cloudsentinel
kubectl rollout undo deployment/monitoring-project-deployment -n cloudsentinel
```

Do not delete `data-catalogue-pvc`; new records are additive and older history
remains readable. Existing live-ready models do not require refetch or retrain
because their inference and recipe contracts are unchanged.

## Live acceptance procedure

1. Open **Live Monitoring**, select an existing live-ready model and leave
   automatic RCA disabled. Start it and verify committed graph cycles continue
   normally and no RCA run appears.
2. Start a second session with automatic RCA enabled for a model shown as
   RCA-compatible. Choose a test threshold, analysis window and cooldown.
3. Verify session detail shows the exact configuration and red threshold line.
4. When a committed cycle is strictly above the threshold, verify exactly one
   marker and RCA run appear. Inspect both workers' logs without expecting a
   Celery task ID in the browser.
5. If subsequent cycles remain high, verify they are marked suppressed and no
   additional run is created, including repeated 100% cycles.
6. Wait for the run to become completed or failed. For completion, inspect the
   stored ranking/graph and provenance. Treat it as diagnostic output, not as a
   proven root cause.
7. Produce a valid cycle at or below threshold. Verify recovery. Test the
   documented cooldown behavior before expecting a later episode to trigger.
8. Stop monitoring while an RCA run is already submitted. Verify no later cycle
   creates a trigger; allow the submitted run to finish and verify it stays in
   historical detail.
9. Restart Redis or let Celery results expire without deleting the PVC. Verify
   the stopped session, cycles, RCA status and completed result still load.

These steps are operator acceptance checks. This implementation has not been
validated with a real live monitoring or RCA run by the code change itself.

## Troubleshooting and limits

- **RCA incompatible:** inspect the safe setup reason. The recipe must provide
  explicit unique pod/metric metadata. Continue AD monitoring with RCA off.
- **RCA runtime readiness unavailable:** verify the CRCA API and `/healthz`, then
  its internal capability endpoint. No monitoring input is sent by that check.
- **Queued/pending:** inspect API/worker connectivity and use the reconcile
  endpoint only while the monitoring session is active.
- **Skipped:** the committed context did not contain enough valid rows for the
  requested analysis window. No refetch or imputation occurs.
- **Running after worker crash:** no automatic lease recovery exists yet.
- **100% anomaly cycles:** this tranche does not diagnose or alter IF/LOF/CGNN
  scores, fitting or thresholds.
- CausalRCA remains the existing algorithm and is not generalized into an RCA
  plugin framework. Automatic remediation and monitoring auto-resume remain out
  of scope.
