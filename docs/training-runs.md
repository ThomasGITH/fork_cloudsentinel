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
