# Generic monitoring CausalRCA

## RCA method contract foundation

`EXPAND-RCA-1` added stable method identities. `EXPAND-RCA-2` registers the
version-pinned BARO RobustScorer adapter while retaining the established
CausalRCA calculation:

```text
cloudsentinel.causal-rca/legacy-v1       available
cloudsentinel.baro-robust-scorer/v1      available when fse-baro 0.2.2 loads
```

`automatic_rca.method` is pinned in each new monitoring session, RCA intent,
stable run identity and algorithm provenance. Requests and historical records
without this field resolve to `cloudsentinel.causal-rca/legacy-v1`; stored
records are not rewritten. Unknown IDs are rejected. BARO is advertised as
available only when the API can import the exact `fse-baro==0.2.2` dependency.

The internal capability response retains its legacy single-method fields and
also provides a bounded `methods` list. Each item has an ID, display name,
availability, input contract, preprocessing contract and minimum requirements.
The Monitoring Setup form renders only methods reported as `available`, and
Data Ingestion validates the chosen ID and runtime availability again. The RCA
worker uses an explicit method registry with separate CausalRCA and BARO
execution adapters.

Method identity is part of the configuration material hashed by
`stable_run_id`. Consequently, two otherwise identical intents for different
methods cannot share an ID. Capability responses expose no matrix, PromQL,
filesystem path or private algorithm configuration.

### BARO RobustScorer scope and change point

This integration uses the official
`baro.root_cause_analysis.robust_scorer(data, anomalies=[index])` interface.
It does not run BARO's BOCPD component: CloudSentinel's selected Saved Model,
committed anomaly predictions and threshold policy remain responsible for
detection and triggering.

For a BARO intent, Data Ingestion reads the immutable result artifact belonging
to the triggering monitoring cycle. Within the already selected RCA analysis
window it chooses the chronologically first stored binary prediction equal to
one. Its timestamp and exact matrix index become the pinned change point. Rows
before the index are the baseline; rows at and after it are the anomalous
segment. The input contract requires at least two rows in both segments because
the official implementation drops constant columns independently in each
segment before fitting/transformation. At least two mapped features remain a
CloudSentinel RCA requirement.

The change-point contract, source cycle, result checksum, timestamp, index and
segment counts are included in the private immutable RCA input and therefore in
its hashes and stable run identity. Safe run/result provenance exposes those
bounded evidence fields without exposing predictions, matrices or storage
references. Missing positives or insufficient segment data produce a preflight
skip with attempt number zero; monitoring continues.

CausalRCA keeps its pinned MinMax preprocessing. BARO receives the finite raw
metric matrix and performs its baseline normalization through RobustScorer's
`RobustScaler`. The official package returns ordered feature names but no
numeric scores. The adapter consequently calculates the same maximum robust
z-score used by version 0.2.2, verifies that these scores reproduce the official
ordering, and maps the internal columns back to the exact immutable feature IDs
and optional entity metadata. BARO results intentionally contain no causal
graph.

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
- same-episode reanalysis: disabled by default;
- minimum reanalysis interval: `60..604800` seconds, technical default 900;
- minimum new observations: `1..100000`, technical default 20;
- accepted execution attempts per episode: `1..20`, technical default 3,
  including the first run.

The reanalysis defaults are operational guardrails, not validated analytical
settings. They must suit the recipe sampling interval, RCA analysis window and
actual CausalRCA runtime. Episode cooldown controls when a recovered episode
may re-arm. Reanalysis interval limits additional analyses while one episode
remains continuously high; these are separate settings.

The trigger is strictly:

```text
committed cycle anomaly_percentage > configured threshold
```

It uses only newly eligible predictions in that committed cycle. An equal
percentage does not trigger. A warm-up or no-new-data cycle has no percentage,
does not trigger and does not recover an episode.

The first crossing while armed creates one durable run intent. With reanalysis
disabled, continued high cycles are recorded as `suppressed_episode` and link
to the earlier analysis. This remains the default and the behavior of old
sessions without the additive policy fields. A valid cycle at or
below the threshold records recovery. If cooldown has already expired, this
re-arms immediately. Recovery during cooldown enters `recovered_cooldown`. A
new high episode during that period is suppressed and enters
`cooldown_episode`; persistent high values never cause a delayed run. A later
valid cycle at or below the threshold after cooldown re-arms the session. This
conservative policy prevents trigger storms.

## Optional bounded same-episode reanalysis

For a new session, an operator may explicitly enable bounded same-episode
reanalysis. A later committed high cycle is eligible only when all gates pass:

1. its percentage is strictly above threshold and it has new predictions;
2. no RCA run for the session is queued or running;
3. the interval since the last durable execution intent has elapsed;
4. the current analysis window contains enough unique timestamps after the
   previous RCA analysis end;
5. the selected analysis content differs from the previous analysis;
6. accepted attempts remain in the episode budget;
7. recipe readiness and existing runtime input/window validation pass.

The interval starts at the durable server timestamp of the accepted execution
intent. It does not use browser polling, dispatch, worker start or completion
time. Dispatch retries keep the same run ID, intent time and attempt number. A
failed accepted run consumes one attempt and remains rate-limited.
Each accepted reanalysis also moves `cooldown_until` from that intent's cycle
time, so recovery shortly after the latest attempt still respects the separate
between-episode cooldown.

New observations are timestamp-based: only unique selected timestamps strictly
after the previous analysis end count. Buffer length, overlapping collection,
warm-up, repeated collection and late samples at or before that end cannot
inflate the count. Analysis windows may overlap. A content checksum excluding
the cycle-artifact reference prevents identical matrices and timestamps from
being reanalysed through another reference.

An input-selection failure is a **preflight skip**. It creates one idempotent
per-cycle history record but no execution intent, consumes no budget, starts no
interval and does not mark an armed episode analysed. A later high cycle is
evaluated again even under the default one-run policy. Accepted intent
reservation and episode-state publication share the session file lock, so two
concurrent eligible cycles cannot consume the same attempt. At most one
queued/running run is allowed per session.

Stopping monitoring prevents newly evaluated trigger intents and new dispatch
attempts. A task submitted before the stop may finish and persist its result;
it cannot resume monitoring.

## Compatibility and naming boundary

AD monitoring and RCA readiness are separate. A live-ready model can always be
monitored with RCA disabled.

The readiness fix removes an incorrect dependency on `identity_labels=["pod"]`.
Catalogue collection already freezes the actual Prometheus series labels in
`series_mapping.metric_labels`; `identity_labels` is an optional query-shaping
field and is not proof that every resulting feature must have pod semantics.
No deployed user model was available in this checkout for inspection. The bug
and fix are verified against the catalogue builder and representative stored
recipe fixtures with the same contract.

The RCA gate accepts only recipes where:

- at least two feature IDs exist and are unique;
- every feature has exactly one `series_mapping` entry;
- each mapping references one known, unique query execution;
- every query execution has an explicit `feature_name` used as `metric_name`;
- stored column order and feature metadata match exactly;
- the selected matrix has at least two observations and two features and all
  values are finite.

The immutable feature ID is both the unique matrix-column name and the primary
ranking/graph label. Optional entity metadata is derived only from an explicit
`series_mapping.metric_labels.pod`, or from a `resolved_target.pod` when that
query is explicitly `execution_mode=per_target` and `target_type=pod`. If both
exist they must agree. Entity metadata may be unavailable; that does not make a
uniquely mapped feature incompatible and the UI then displays `Unavailable`.
No pod, service or metric is parsed from a feature string or PromQL.

New RCA inputs identify this derived mapping as
`cloudsentinel.crca-feature-metadata/v1`. Each column records
`column_index`, exact `feature_id`, identical `feature_name`, explicit query
`metric_name`, and optional `{type, id, source}` entity metadata. That contract
and its checksum are included in new run provenance. The CRCA boundary accepts
these explicit feature names; already queued inputs with the previous explicit
flat `entity_type`/`entity_id` fields remain readable. Historical records are
not rewritten.

Safe readiness reason codes include `missing_recipe`,
`unsupported_recipe_contract`, `invalid_sampling`, `invalid_alignment`,
`invalid_missing_data_policy`, `insufficient_features`, `invalid_feature_id`,
`duplicate_feature_id`, `missing_series_mapping`, `invalid_series_mapping`,
`invalid_query_execution`, `unknown_execution_id`,
`missing_query_feature_name`, `conflicting_entity_metadata`, and
`invalid_recipe_integrity`. Insufficient rows or incomplete window coverage is
a runtime `skipped` condition rather than static recipe incompatibility. New
skipped runs store a safe `skip_reason_code`, including
`insufficient_runtime_context`, `incomplete_analysis_window`,
`invalid_matrix_shape`, `feature_metadata_mismatch`, or
`non_finite_runtime_input`.

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
preflight record. It does not require recovery before a later high cycle can
try again and does not consume the execution budget.

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

New sessions use policy `cloudsentinel.monitoring-rca-reanalysis/v1`. Each
accepted run records first analysis versus reanalysis, episode and attempt
number, budget, previous run, durable intent timestamp, previous/current
analysis end, timestamp-based new-observation count, content identity and its
own stored input/result checksums. Existing JSON records are read without
migration and are never rewritten merely by deploying this version.

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
threshold and windows, selected reanalysis policy, attempts used versus budget
and conditional earliest eligibility time. It distinguishes first analysis,
reanalysis, preflight skip, same-episode, interval, new-data, active-run and
budget decisions. Suppressed decisions link to earlier analyses where present.
The run list labels each attempt and its own analysis window. The dashboard also
shows a threshold line, trigger markers, run status, stored
ranking, stored graph and safe checksum provenance. It distinguishes started,
suppressed, skipped and incompatible states. Rankings are described as
diagnostic evidence and never as a proven root cause.

### Polling and rendering

The dashboard still polls while a monitoring session or RCA run is active, but
polling no longer implies a full visual refresh. `FIX-POLLING-ISSUE-1` compares
the safe list and detail payloads with the last rendered payload. Unchanged
responses leave the session list, selected detail, chart and RCA ranking in
place. A newly committed cycle updates the existing Chart.js instance with
animation disabled, so existing points do not replay their entrance animation.
Terminal RCA detail responses are cached in the browser by run identity and
result identity; the cache is bounded to 50 entries. The browser pauses its
poll timer while the page is hidden and immediately refreshes after it becomes
visible again. A temporary polling error shows the safe error message without
erasing the last successfully rendered dashboard.

## Build, load and apply (do not run from this document automatically)

`EXPAND-RCA-2` changes Data Ingestion API/worker, CRCA API/worker and the Django
monitoring UI. Run these commands from the repository root on the EC2 host:

```bash
docker build \
  -t jojojochem/data_ingestion:expand-rca-2 \
  ./data_ingestion
docker build \
  -f anomaly_detection/crca/Dockerfile \
  -t jojojochem/anomaly_detection_crca:expand-rca-2 \
  .
docker build \
  -t jojojochem/monitoring_project:expand-rca-2 \
  ./user_interface/monitoring_project

minikube image load jojojochem/data_ingestion:expand-rca-2
minikube image load jojojochem/anomaly_detection_crca:expand-rca-2
minikube image load jojojochem/monitoring_project:expand-rca-2

kubectl apply -f k8s/data_ingestion-deployment.yml
kubectl apply -f k8s/data_ingestion_celery-deployment.yml
kubectl apply -f k8s/crca_anomaly_detection-deployment.yml
kubectl apply -f k8s/crca_anomaly_detection-celery-deployment.yml
kubectl apply -f k8s/monitoring_project-deployment.yml
kubectl rollout status deployment/data-ingestion-deployment -n cloudsentinel
kubectl rollout status deployment/celery-worker-deployment -n cloudsentinel
kubectl rollout status deployment/crca-anomaly-detection-deployment -n cloudsentinel
kubectl rollout status deployment/crca-anomaly-detection-celery-deployment -n cloudsentinel
kubectl rollout status deployment/monitoring-project-deployment -n cloudsentinel
```

The manifest uses `imagePullPolicy: IfNotPresent`, which is required for the
locally loaded images. Generic Detection Runtime, Learning API/worker,
data-processing and legacy CGNN images remain unchanged. The BARO wheel is
installed with `--no-deps` after the CRCA image's explicit dependency pins, so
the package cannot silently replace pandas, NumPy or scikit-learn versions.

Check the exact deployment names and the unchanged CRCA worker:

```bash
kubectl rollout status deployment/data-ingestion-deployment -n cloudsentinel
kubectl rollout status deployment/celery-worker-deployment -n cloudsentinel
kubectl rollout status deployment/monitoring-project-deployment -n cloudsentinel

kubectl get pods -n cloudsentinel
kubectl get deployment -n cloudsentinel \
  data-ingestion-deployment celery-worker-deployment \
  monitoring-project-deployment crca-anomaly-detection-deployment \
  crca-anomaly-detection-celery-deployment \
  -o custom-columns=NAME:.metadata.name,IMAGE:.spec.template.spec.containers[0].image,READY:.status.readyReplicas

kubectl logs -n cloudsentinel deployment/celery-worker-deployment --tail=100
kubectl logs -n cloudsentinel deployment/crca-anomaly-detection-celery-deployment --tail=100
```

Optional port-forwards, each in its own terminal:

```bash
kubectl port-forward -n cloudsentinel service/data-ingestion-service 5001:80
kubectl port-forward -n cloudsentinel service/crca-anomaly-detection-service 5023:80
kubectl port-forward -n cloudsentinel service/monitoring-project-service 8000:80
```

Then verify health/readiness without starting monitoring:

```bash
curl --show-error --fail http://127.0.0.1:5001/healthz | jq
curl --show-error --fail http://127.0.0.1:5023/healthz | jq
curl --show-error --fail http://127.0.0.1:5023/internal/monitoring-capability | jq

export MODEL_ID='replace-with-an-existing-live-ready-model-id'
curl --show-error --fail \
  "http://127.0.0.1:5001/monitoring-rca/readiness/${MODEL_ID}" | jq
```

For a non-destructive rollback, keep all PVCs and undo only the changed
Deployments:

```bash
kubectl rollout undo deployment/data-ingestion-deployment -n cloudsentinel
kubectl rollout undo deployment/celery-worker-deployment -n cloudsentinel
kubectl rollout undo deployment/monitoring-project-deployment -n cloudsentinel
```

Do not delete `data-catalogue-pvc`. New policy fields and run records are
additive; older images ignore them, although a session created with reanalysis
should be stopped before rollback to avoid applying mixed policy code to one
active session. Existing models need no fetch, training or recipe change.

## Live acceptance procedure

1. Open **Live Monitoring**, select an existing live-ready model and leave
   automatic RCA disabled. Start it and verify committed AD cycles continue and
   no RCA run appears.
2. Start a session with automatic RCA enabled and same-episode reanalysis
   disabled. Verify the setup and detail state both say that only one analysis
   is allowed per continuous episode.
3. Cause or wait for one committed cycle strictly above threshold. Verify one
   first-analysis marker/run. Keep later cycles above threshold and verify
   `episode_already_analyzed`, a link to the earlier run, and no second run.
4. Start a separate session with reanalysis enabled. The safe temporary test
   values are the defaults: 900-second interval, 20 new observations and budget
   3, provided the recipe sampling/window can supply at least 20 new rows in 15
   minutes. At 15-second sampling, about 60 timestamps may become available in
   that interval. Do not shorten the interval merely to force heavy RCA runs.
5. Verify attempt 1 starts at the first high eligible cycle. While it is queued
   or running, verify later high cycles say another run is active and no
   parallel run appears.
6. After attempt 1 is terminal, verify high cycles remain suppressed until both
   900 seconds since its durable intent and 20 unique timestamps after its
   analysis end are available. The UI's “not earlier than” time is conditional,
   not a scheduling guarantee.
7. Verify attempt 2 is labelled **Reanalysis**, links to attempt 1, has its own
   cycle/window/input checksums and shows its timestamp-based new-row count.
8. Keep the episode high through accepted attempt 3. Verify later high cycles
   report that the episode budget is reached. If an accepted run fails, confirm
   it still consumes one attempt and no immediate replacement appears.
9. For preflight behavior, use an analysis window for which the first high cycle
   lacks stored context. Verify a skipped record uses attempt 0; a later high
   cycle may start attempt 1 once enough context exists, without a recovery
   cycle or Prometheus refetch.
10. Produce a valid cycle at or below threshold. Verify episode recovery. Test
    cooldown separately before expecting the first run of a new episode.
11. Stop monitoring while a run is submitted. Verify no new intent is created;
    the submitted run may finish and remain visible in history.
12. Reopen the stopped session after Redis/Celery result expiry or service
    restart without deleting the PVC. Verify every decision, attempt, independent
    input provenance and completed result still loads.
13. For a completed result, verify ranking labels are exact feature IDs and treat
    the ranking/graph as diagnostic evidence, never proof of a root cause.

These are operator acceptance checks. No real monitoring, inference or RCA run
was executed by this code change. Repeated 100% IF results are useful only for
checking rate limits and budget; their detection correctness remains unknown.

## Troubleshooting and limits

- **RCA incompatible:** inspect the safe reason code. The recipe must have
  unique feature IDs, a one-to-one series mapping, known execution references,
  explicit query feature names and valid recipe integrity. Pod/entity metadata
  is optional. Continue AD monitoring with RCA off.
- **RCA runtime readiness unavailable:** verify the CRCA API and `/healthz`, then
  its internal capability endpoint. No monitoring input is sent by that check.
- **BARO unavailable:** verify that the capability response reports
  `cloudsentinel.baro-robust-scorer/v1` as available and that the runtime has
  exactly `fse-baro==0.2.2`. An absent or different version is rejected.
- **BARO skipped:** inspect the safe reason code. A stored positive detector
  prediction must exist inside the selected immutable analysis window, with at
  least two baseline and two post-change rows. No data is refetched.
- **Queued/pending:** inspect API/worker connectivity and use the reconcile
  endpoint only while the monitoring session is active.
- **Skipped:** the committed context did not contain enough valid rows for the
  requested analysis window. No refetch or imputation occurs. The skip consumes
  no attempt; a later high cycle is checked again.
- **Reanalysis did not start:** inspect the durable cycle decision. Interval,
  timestamp-based new data, unchanged input, an active run or budget may each
  suppress it. “Not earlier than” is conditional and not a scheduled start.
- **Running after worker crash:** no automatic lease recovery exists yet.
  A hanging running run blocks reanalysis; this tranche deliberately does not
  add a lease/reaper or allow a second parallel run.
- **100% anomaly cycles:** this tranche does not diagnose or alter IF/LOF/CGNN
  scores, fitting or thresholds.
- CausalRCA and BARO RobustScorer are explicit registered methods, not a new RCA
  plugin framework. BARO BOCPD, automatic remediation and monitoring auto-resume
  remain out of scope.
