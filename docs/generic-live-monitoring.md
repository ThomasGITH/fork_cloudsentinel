# Generic live monitoring

CloudSentinel's existing Live Monitoring screens now select an explicit Saved
Model artefact. They do not select a detector type and they do not infer live
queries from feature names.

## Provenance contract

A successful fetch of a new Prometheus-backed catalogue version writes an
immutable `live_input_recipe` using `cloudsentinel.live-input/v1`. The recipe
contains the configured Prometheus source identifier, resolved query
executions, target bindings, identity labels, exact series-to-feature mapping,
sampling interval, timestamp alignment and missing-data policy. Its canonical
JSON SHA-256 and feature-order SHA-256 are verified when the training bundle is
downloaded and are copied into the immutable snapshot and Saved Model record.

No recipe is generated for an old version or model. Such models remain valid
for offline inference and Comparison and are exposed as
`live_monitoring.status=not_ready`.

The first policy is deliberately strict: off-grid samples are ignored with a
warning, while a live window containing a missing value is rejected. No
forward fill, backward fill, zero fill or threshold adaptation occurs.

## Session lifecycle

`POST /start_monitoring` accepts JSON containing an explicit `model_id`,
`window_seconds` and `poll_interval_seconds`. Before dispatch, data ingestion
resolves the model through the Learning service's internal recipe endpoint and
pins the model artifact-manifest hash, recipe hash and feature-order hash in a
Redis-backed session.

The existing `monitoring_task` queries only the server-configured Prometheus
source, assembles features through the same canonical function used by offline
fetch, verifies the exact stored series mapping and calls only the Generic
Detection Runtime. Per-session buffers are deduplicated and bounded. The
runtime adapter remains responsible for model-specific windowing and reports
warm-up observations.

Public session responses omit PromQL, matrices, task IDs, service URLs and
internal storage details. They show collection state, buffered observations,
warm-up, warnings, missing-value failures and the latest compact generic result.

Routes:

- `GET /get_active_tasks` lists safe current/recent session projections.
- `GET /monitoring-sessions/<session_id>` returns one safe projection.
- `DELETE /stop_monitoring/<session_id>` stops that exact session.

## Remaining legacy boundary

The active continuous-monitoring setup, task list and result display no longer
read models or results from the CGNN detection service. Legacy CGNN resources
are intentionally retained. The manual historical CGNN anomaly-detection and
configuration views still reference `API_CGNN_ANOMALY_DETECTION_URL`; those
callers require a separate parity decision before the legacy deployment can be
removed. The compatibility `/preprocess_cgnn_data` route also remains, but the
new live task does not call it.

## Local verification

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/cloudsentinel-pycache \
  env/bin/python -m unittest tests.test_generic_live_monitoring
PYTHONPYCACHEPREFIX=/tmp/cloudsentinel-pycache \
  env/bin/python -m unittest discover -s tests -p 'test_*.py'
git diff --check
```
