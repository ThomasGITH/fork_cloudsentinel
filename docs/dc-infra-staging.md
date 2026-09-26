# Data Catalogue staging infrastructure

DC-INFRA-1 prepares the Data Catalogue to TrainingRun flow for a controlled,
single-node Minikube or EC2 Kubernetes staging environment. It does not deploy
the manifests or provision their volumes.

## Images

Build both images from the same source revision and do not overwrite their
staging tags:

```sh
docker build -t jojojochem/data_ingestion:dc-infra-1 data_ingestion
docker build -f learning_adaptation/Dockerfile \
  -t jojojochem/learning_adaptation:dc-infra-1 .
```

DC-INFRA-2 must test these images locally, push or load them into the selected
cluster runtime, and keep the API and worker of each service on the exact same
tag or image digest.

## Persistent storage

`data-catalogue-pvc` is mounted by the data ingestion API and its Celery
worker. Both use `/app/storage/catalogue` through
`CATALOGUE_STORAGE_ROOT`. This includes catalogue records, immutable versions,
fetch attempts, staging data, raw responses, canonical artifacts, labels,
partitions, and materialised training bundles.

`learning-adaptation-pvc` is mounted by the learning API and worker. The API
stores immutable snapshots and TrainingRun metadata below `/app/storage`; both
pods share `/app/storage/trained_models_temp` for the existing CGNN lifecycle
and Isolation Forest promotion retries.

Both claims use `ReadWriteOnce`. This is suitable for the intended single-node,
single-replica staging deployment. A multi-node or multi-replica deployment
requires RWX or object storage plus cross-process locking and coordination.

## Runtime constraints

The data ingestion worker and learning worker both run with concurrency one.
For catalogue storage this limits concurrent writers to the file-backed
repository. For learning it avoids concurrent use of CGNN's global mutable
configuration. It does not provide distributed locking.

The containers run as UID/GID 10001 and the pods use `fsGroup: 10001` with
`fsGroupChangePolicy: OnRootMismatch`. The selected volume provisioner must
honour this ownership configuration.

Only the learning worker has repository-backed resource settings: 2 CPU and
4 GiB memory requested, with 4 CPU and 8 GiB limits. Data ingestion and API
resource values must be established from DC-INFRA-2 staging measurements.

The API health endpoints only report process health. They do not access Redis,
Prometheus, Celery, Kubernetes, or persistent artifacts.

## DC-INFRA-2 prerequisites

Before applying manifests, verify the cluster is single-node, the default
StorageClass supports the two RWO claims, the Prometheus service exists as
`prometheus-server.monitoring.svc.cluster.local`, and the tagged images are
available to the cluster. After deployment, verify task registration, service
DNS, PVC visibility between each API/worker pair, a bounded catalogue fetch,
bundle materialisation, a combined TrainingRun, Isolation Forest promotion,
and persistence after pod restarts.
