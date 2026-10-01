# Generic training runs

`POST /training_runs` starts independent children of the single generic Celery
task `train_detector_plugin_task` for one dataset. `GET /training_runs/{run_id}` returns the durable run record, reconciled
with current Celery state when that state is still available. The legacy CGNN
and Isolation Forest endpoints remain available.

`GET /training_runs` is the compact history endpoint. It accepts `page`
(default `1`), `page_size` (default `20`, maximum `100`), `status`,
`detector_id`, `dataset_id`, `source`, and `sort=newest|oldest`. It omits task
IDs, filesystem paths, broker information, stack traces, and large results. A
damaged run record is counted in `skipped_corrupt_records` and does not make the
whole history unavailable.

Generic TrainingRun children persist `queued`, `running`, `completed`,
`failed`, `dispatch_failed`, or `validation_failed`, including lifecycle
timestamps and compact failure/result metadata. Parent state is recalculated
from all children as `queued`, `running`, `completed`, `partial_success`, or
`failed`. Updates use a per-run filesystem lock and atomic replacement. Stored
terminal states take precedence over ambiguous Celery `PENDING` results, which
also protects history after Redis result expiry. A task failure is written when
Celery reaches terminal failure; no scheduler or status sweeper is introduced.

The local implementation stores immutable source snapshots below
`DATASET_SNAPSHOT_STORAGE_ROOT` and run metadata below
`TRAINING_RUN_STORAGE_ROOT`. Generic children read the shared immutable
snapshot directly; they do not create detector-specific input copies.
Existing datasets are read from
`EXISTING_DATASETS_ROOT`. Defaults point at directories beside the
learning-adaptation application.

## Saved models

Successful generic TrainingRun children create an artifact-independent model
record below `MODEL_CATALOGUE_STORAGE_ROOT`. Its default is a `models`
directory beside `TRAINING_RUN_STORAGE_ROOT`. The record retains detector,
dataset/version/partition, snapshot, feature-order, parameter, compact
evaluation, and promotion provenance. It never stores model bytes, worker
paths, temporary directory names, service URLs, or Celery task IDs.

`GET /models` supports `page`, `page_size` (maximum `100`), `detector_id`,
`status`, `dataset_id`, `training_run_id`, and `sort=newest|oldest`.
`GET /models/{model_id}` returns the complete safe metadata projection and
returns `404` for an unknown model. Corrupt records are isolated from list
responses in the same way as corrupt run records.

For new inference-enabled TrainingRuns, the generic executor validates and
atomically publishes the adapter's complete artifact before it writes the
Saved Model record. CGNN, Isolation Forest, and LOF records use
`promotion.status=not_applicable` and `inference.status=ready`; no
detector-specific `/save_model` call is part of successful training. The
catalogue record survives subsequent cleanup of temporary artifacts. Failed,
dispatch-failed, and validation-failed children do not create available model
records. Repeated lifecycle delivery for the same model and provenance reuses
the first record rather than changing its identity or timestamps.

The API validates manifest capabilities, manifest parameters, and the shared
`metrics-partition-v1` envelope before dispatch. Each generic child lazily loads
its adapter and performs detector-specific compatibility validation.
Compatibility, execution, promotion, or dispatch failure for one detector does
not cancel its siblings. The response field
`physical_parallelism_guaranteed` is `false`; actual parallel execution needs
separate Celery queues or workers.

This filesystem backend is intended for local development and tests. A
Kubernetes deployment needs storage shared by the learning-adaptation API and
workers, or an object-storage implementation of the snapshot store.

CGNN still uses process-global `config.json`. IF-D2 permits only one CGNN child
per run and does not make concurrent CGNN runs safe. True CGNN parallelism
requires process isolation, request-scoped configuration, and dedicated worker
queues.

DC-5 additionally accepts an exact catalogue source:

```json
{
  "dataset": {
    "source": "catalogue",
    "dataset_id": "ds_...",
    "version": 3,
    "partition_id": "partition-..."
  },
  "detectors": [{"detector_id": "isolation-forest", "parameters": {}}]
}
```

The learning service downloads the binary bundle from `API_DATA_CATALOGUE_URL`,
verifies its manifest and file hashes, and atomically creates the same immutable
snapshot shape used by existing datasets. Catalogue identity, partition and
feature-order hashes, label provenance, source hashes, and observation counts
are stored in both snapshot and run metadata.

Catalogue, snapshot, or bundle-integrity failures stop the run before dispatch.
A detector-specific compatibility failure is stored as `validation_failed` on
that child while compatible siblings continue. CGNN feature importance
is unavailable for catalogue data because arbitrary PromQL feature names do
not provide a trustworthy container-by-metric mapping; legacy CGNN behavior is
unchanged.

First-version limitations: records created before this lifecycle was deployed
remain readable but are not retroactively turned into Saved Model records.
The detector-specific legacy training routes remain backward compatible but do
not create model-catalogue records because they have no generic TrainingRun or
snapshot provenance.
Without a periodic reconciler, a worker that is forcibly terminated without a
Celery terminal failure event may remain `running` until a later detail/history
request can reconcile it with a retained Celery result. The file locks protect
processes sharing one filesystem; multi-host correctness still depends on the
shared filesystem's locking semantics.


`POST /training_runs` also accepts an optional non-empty `run_name` of at most
200 characters. It is persisted and included in list and detail responses;
existing requests and old records remain valid without it.
