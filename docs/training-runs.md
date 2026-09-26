# Generic training runs

`POST /training_runs` starts independent detector-specific Celery tasks for one
existing dataset. `GET /training_runs/{run_id}` returns the stored run metadata
and current child task states. The legacy CGNN and Isolation Forest endpoints
remain available.

The local implementation stores immutable source snapshots below
`DATASET_SNAPSHOT_STORAGE_ROOT` and run metadata and child inputs below
`TRAINING_RUN_STORAGE_ROOT`. Existing datasets are read from
`EXISTING_DATASETS_ROOT`. Defaults point at directories beside the
learning-adaptation application.

The API validates and prepares every selected detector before dispatching the
first task. Dispatches remain independent: a dispatch or runtime failure for
one detector does not cancel its siblings. The response field
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

Catalogue or bundle-integrity failures stop the run before dispatch. A
detector-specific compatibility failure is stored as `validation_failed` on
that child while compatible siblings are dispatched. CGNN feature importance
is unavailable for catalogue data because arbitrary PromQL feature names do
not provide a trustworthy container-by-metric mapping; legacy CGNN behavior is
unchanged.
