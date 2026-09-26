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

The file backend is intended for local development and tests. API and worker
pods will need shared storage or object storage before DC-3 adds background
Prometheus fetch-and-save tasks. DC-3 will also add the fetch endpoint and
status transitions; no fetch endpoint exists in DC-2.
