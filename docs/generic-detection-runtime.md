# Generic anomaly-detection runtime

The generic runtime provides one detector-independent batch-inference host.
New generic TrainingRuns publish their artifacts only through the generic
artifact store; CGNN, Isolation Forest, and LOF adapters do not call a legacy
detection service. Existing detector-specific services and routes remain
temporarily available for old records and callers during live parity testing.

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

The manual CSV detection caller now resolves an exact Saved Model by
`model_id`, obtains its authoritative feature identity from learning
adaptation, activates it, and sends the raw matrix to the generic runtime.
Both upstream URLs are server configured. The caller never accepts an upstream
URL from request metadata and does not dispatch on detector ID.

Continuous Prometheus monitoring still depends on the legacy CGNN service.
Its current model metadata does not contain a safe generic mapping from
arbitrary catalogue feature identity to the monitoring collector's concrete
Prometheus queries. Guessing that mapping from feature names would be unsafe.
The CGNN deployment therefore cannot be removed until a server-side live input
profile/query binding is implemented and verified. The Isolation Forest
service has no remaining new-TrainingRun caller and can be retired after the
live IF generic parity checks in `generic-detection-migration.md` pass.

The deployment uses three storage boundaries:

- `learning-adaptation-pvc` remains the owner of Saved Model catalogue
  records. The generic runtime mounts it read-only.
- `model-artifact-pvc` is written only by the learning worker and mounted
  read-only by the generic runtime.
- `detection-results-pvc` is writable only by the generic runtime.

The external `detector-plugin-repository-pvc` is also mounted read-only by the
runtime. Existing Saved Models without generic artifact evidence remain
`legacy_external` and are never activated implicitly.
