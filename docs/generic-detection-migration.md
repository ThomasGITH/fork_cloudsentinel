# Generic detection migration

## New training lifecycle

CGNN, Isolation Forest, and external LOF now use the same required publication
boundary:

```text
training and evaluation
→ validate complete ArtifactResult
→ atomically publish immutable artifact
→ write Saved Model with inference.status=ready
→ complete TrainingRun child
```

The built-in adapters return `promotion.status=not_applicable`. The generic
executor reports a `PUBLISHING` phase and is the only component that publishes
new TrainingRun artifacts. A legacy detector-service outage cannot turn an
otherwise successful generic TrainingRun into a failure.

Old Saved Models without a verified generic artifact remain readable with
`inference.status=legacy_external`. They must be retrained or migrated through
a separately validated migration; the runtime does not guess artifact paths.

## Caller migration and remaining boundary

The manual CSV detection flow in `data_processing` now uses two server-side
settings:

- `API_LEARNING_ADAPTATION_URL` to resolve the exact Saved Model and its
  feature identity;
- `API_GENERIC_ANOMALY_DETECTION_URL` to activate that model and call
  `POST /detect`.

The request supplies a `model_id`. The caller sends the authoritative feature
order/hash from the Saved Model record and passes the raw numeric matrix so the
loaded adapter uses its training-fitted scaler. A request cannot provide or
override either upstream URL.

The continuous monitoring flow is not yet safe to migrate. Its collector uses
a fixed metric-key configuration, while catalogue-trained models may have
arbitrary PromQL-derived features. A new server-side live input binding must
map a model's feature identity to ordered Prometheus queries before this caller
can use the generic runtime. Until then these legacy CGNN callers remain:

- monitoring model selection and continuous monitoring;
- legacy CGNN configuration and historical result-management screens.

Consequently the legacy CGNN Deployment and Service are still required. The
legacy Isolation Forest Deployment and Service have no required new
TrainingRun caller after this migration, but should be removed only after the
live verification below succeeds.

## Build and rollout

From the repository root, build these immutable staging images:

```bash
docker build -f learning_adaptation/Dockerfile \
  -t jojojochem/learning_adaptation:generic-detection-migration-1 .
docker build -f anomaly_detection/generic/Dockerfile \
  -t jojojochem/anomaly_detection_generic:generic-detection-migration-1 .
docker build -f data_processing/Dockerfile \
  -t jojojochem/data_processing:generic-detection-migration-1 data_processing
docker build -f user_interface/monitoring_project/Dockerfile \
  -t jojojochem/monitoring_project:generic-detection-migration-1 \
  user_interface/monitoring_project
```

Load them into Minikube:

```bash
minikube image load jojojochem/learning_adaptation:generic-detection-migration-1
minikube image load jojojochem/anomaly_detection_generic:generic-detection-migration-1
minikube image load jojojochem/data_processing:generic-detection-migration-1
minikube image load jojojochem/monitoring_project:generic-detection-migration-1
```

Apply only the affected manifests, then wait for rollouts:

```bash
kubectl apply -f k8s/learning_adaptation-deployment.yml
kubectl apply -f k8s/learning_adaptation-celery-deployment.yml
kubectl apply -f k8s/data_processing-deployment.yml
kubectl apply -f k8s/generic_anomaly_detection-deployment.yml
kubectl apply -f k8s/generic_anomaly_detection-service.yml
kubectl apply -f k8s/monitoring_project-deployment.yml

kubectl rollout status -n cloudsentinel deployment/learning-adaptation-deployment
kubectl rollout status -n cloudsentinel deployment/learning-adaptation-celery-deployment
kubectl rollout status -n cloudsentinel deployment/data-processing-deployment
kubectl rollout status -n cloudsentinel deployment/generic-anomaly-detection-deployment
kubectl rollout status -n cloudsentinel deployment/monitoring-project-deployment
```

## Live verification

1. Confirm `/healthz` through a temporary port-forward to the generic service.
2. Run a fresh combined TrainingRun with CGNN and Isolation Forest. Publish and
   select external LOF in a separate fresh run if it is not already active.
3. Confirm every successful child is `completed`, every Saved Model has
   `inference.status=ready`, and no learning-worker log contains `/save_model`
   or either legacy detection-service DNS name.
4. For each new model, call `POST /models/<model_id>/activate`, then `POST
   /detect` with a bounded matrix and the exact feature order.
5. Compare IF and CGNN results against known fixtures or the legacy runtime
   before retiring any resource.
6. Exercise the manual CSV detection page and confirm it uses a Saved Model ID
   and succeeds while the legacy IF service is unavailable.

After successful IF verification, remove the resources contained in:

```text
k8s/isolation_forest_anomaly_detection-deployment.yml
  Deployment/isolation-forest-anomaly-detection-deployment
  Service/isolation-forest-anomaly-detection-service
```

Do not yet remove the resources contained in
`k8s/cgnn_anomaly_detection-deployment.yml`; continuous monitoring and legacy
result/configuration pages remain blockers. This repository stores each legacy
Service in the same YAML file as its Deployment, so separate
`*-service.yml` files do not currently exist.
