# Retired Isolation Forest detection deployment

IF-C4 originally packaged a standalone Isolation Forest detection service.
That Kubernetes Deployment, Service, and dedicated PVC have now been retired
after the generic runtime was verified with a freshly trained IF model.

New Isolation Forest TrainingRuns publish an immutable artifact to
`model-artifact-pvc`, produce a Saved Model with `inference.status=ready`, and
are activated and executed by `generic-anomaly-detection-service`. They do not
call the former `/save_model` endpoint.

The legacy Python service and Dockerfile remain in the source tree for
backward-compatible local/reference use, but no active Kubernetes manifest
deploys them. Do not use
`API_ISOLATION_FOREST_ANOMALY_DETECTION_URL` for new training or detection.

Current deployment and smoke-test instructions are in:

- `docs/generic-detection-deployment.md`
- `docs/generic-detection-migration.md`

Deleting the manifest from the repository does not delete objects that may
still exist in an older cluster. Remove those explicitly only after completing
the documented parity checks.
