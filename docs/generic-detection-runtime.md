# Generic anomaly-detection runtime

The generic runtime provides one detector-independent batch-inference host. It
does not replace the existing CGNN and Isolation Forest services yet. Those
legacy routes, promotion calls, result formats, and callers remain available
during the migration period.

## Inference contract

A detector opts in through its manifest:

```yaml
capabilities:
  inference:
    enabled: true
    protocol: cloudsentinel.inference/v1
    input_profile: metrics-matrix/v1
    artifact_format: sklearn-pipeline-joblib-v1
```

The adapter implements `InferenceDetectorAdapter.load_model()` and
`InferenceDetectorAdapter.predict_inference()`. The loaded model is opaque to
the host. `PredictionResult` always uses `0 = normal`, `1 = anomaly`, and an
anomaly score where a larger value means more anomalous. The host validates
the output shape, finite scores, binary values, warm-up count, bounded
diagnostics, and JSON serializability.

Discovery remains manifest-only. Adapter code is imported only when a model is
activated. An external model retains the immutable `PluginReference` from its
TrainingRun, so activation resolves the exact plugin release used for
training.

## Artifact lifecycle

For inference-enabled plugins, the generic training executor publishes the
validated `ArtifactResult` to `MODEL_ARTIFACT_STORAGE_ROOT` before creating the
available Saved Model record:

```text
MODEL_ARTIFACT_STORAGE_ROOT/
└── model_<id>/
    ├── artifact_manifest.json
    └── files/
        └── detector-owned required files
```

Publication copies to staging, records each safe filename, byte size, and
SHA-256 checksum, and then installs the directory atomically. Published
artifacts are immutable. Symlinks, traversal, missing required files, and
content conflicts are rejected. The Saved Model receives a logical artifact
ID and manifest checksum, never a filesystem path.

New records use one of these inference statuses: `ready`, `not_ready`,
`legacy_external`, `artifact_unavailable`, `incompatible`, or `not_supported`.
Existing records without generic artifact evidence remain readable and are
projected as `legacy_external`; they are not silently treated as ready.

CGNN publication includes its trained state dict, model configuration,
evaluation metadata, model parameters, and the training-fitted scaler.
Isolation Forest and Local Outlier Factor publish their complete persisted
scikit-learn pipelines plus metadata and evaluation summaries.

## Local HTTP API

The Flask application is `anomaly_detection.generic.app:app` and listens on
port 5015 when run directly.

- `GET /healthz` reports process health without loading the catalogue,
  artifacts, adapters, or ML libraries.
- `POST /models/<model_id>/activate` verifies the Saved Model record and all
  artifact checksums, resolves the pinned adapter, loads the model, and caches
  it. Repeated activation is idempotent.
- `GET /models/<model_id>/runtime-status` returns a path-free status and whether
  the model is currently cached.
- `POST /detect` accepts multipart fields `matrix` (headerless numeric CSV) and
  `metadata` (JSON). Metadata requires `model_id` plus `feature_order` or
  `feature_order_sha256`; timestamps and a small context object are optional.

Detection requires prior activation. The compact response contains counts,
percentage, warm-up observations, detector identity, and a logical result
reference. Full inputs, predictions, and scores are neither returned nor
stored by default. Request bytes, observations, features, cache size, and
result storage are bounded through `GENERIC_DETECTION_*` environment settings.
The optional context object accepts only bounded monitoring/RCA context fields
and never an upstream URL. Generic RCA dispatch is deliberately deferred until
caller migration; when added, its endpoint must come only from server-side
configuration and remain a host concern rather than an adapter concern.

## Local verification

```bash
env/bin/python -m unittest tests.test_generic_detection
env/bin/python -m unittest discover -s tests -p 'test_*.py'
git diff --check
```

The focused tests create genuine persisted IF and novelty-enabled LOF
pipelines, install LOF as an external immutable plugin release, reconstruct a
small CGNN state dict, and exercise activation, cache behavior, input
validation, checksum rejection, and compact result storage.

## Migration boundary

This tranche does not add Kubernetes resources or PVC mounts, migrate live
callers, remove detector-specific services, add streaming sequence state, or
make old artifacts generic-inference-ready. Deployment must later mount the
Saved Model catalogue and generic artifact store read-only in the runtime, and
mount the external Plugin PVC read-only at the same configured root used by
the learning worker. Parity tests must run against legacy CGNN and IF services
before their callers or manifests can be retired.
