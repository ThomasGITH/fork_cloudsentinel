# Data Catalogue backend

DC-2 adds a local file-backed catalogue to the data-ingestion service. It
provides `POST /datasets`, `GET /datasets`, and `GET /datasets/{dataset_id}`.
New records start as version 1 with status `draft`; DC-2 does not execute their
stored PromQL metadata.

Catalogue metadata is stored below `CATALOGUE_STORAGE_ROOT`. Existing datasets
are projected read-only from `LEGACY_DATASETS_ROOT`. Both paths are
configurable; local defaults point beside the data-ingestion application and at
the repository's `learning_adaptation/datasets` directory respectively.

Dataset versions contain immutable source/provenance metadata. Partitions are
separate, checksummed definitions with mode `none`, `predefined`, or
`time_range`. No implicit split is created. Legacy files remain in place and
are represented through safe relative artifact references and SHA-256 hashes.

DC-3 adds a one-shot Celery fetch through
`POST /datasets/{dataset_id}/versions/{version}/fetch`. The caller supplies a
server-side `prometheus_source_id`; it cannot supply a Prometheus URL or
credentials. `GET /datasets/{dataset_id}/fetch-status?version=...` reports the
attempt phase and compact error state. Available versions can be sampled with
`GET /datasets/{dataset_id}/preview?version=...&limit=...`.

Fetch artifacts are staged below the dataset directory and promoted to
`artifacts/{version}` only after query execution, canonical UTC-grid assembly,
validation, and checksumming succeed. Missing values remain empty. No fill,
scaling, detector preprocessing, or automatic train/test split is performed.

The source map and fetch limits use Flask configuration. The default source is
`cluster-default`, configured by `CATALOGUE_CLUSTER_DEFAULT_PROMETHEUS_URL`.
Limits cover query count and length, time window, sampling interval,
theoretical samples, returned series, response bytes, retries, timeouts, and
preview rows.

The file backend remains intended for local development and tests. API and
worker processes must share `CATALOGUE_STORAGE_ROOT`; multi-replica locking and
object storage remain future work.
