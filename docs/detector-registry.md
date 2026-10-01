# Detector registry

Phase A introduced manifest discovery. Phase B added a thin CGNN adapter while preserving the existing CGNN implementation and service contracts. Phase C resolves adapters generically from each manifest entry point. A detector is discovered when a direct child of the `detectors` directory contains a `manifest.yaml` file.

The registry uses `yaml.safe_load`, limits each manifest to 1 MB, rejects paths that resolve outside the configured detector directory, and validates parameter defaults and constraints. `scan_detectors()` validates every plugin in isolation. Invalid manifests and every participant in a duplicate detector ID are quarantined in `discovery_errors`; valid plugins remain available. Discovery validates the `entry_point` string but deliberately does not import it.

`get_adapter(detector_id)` retains the legacy `train`, `evaluate`, `predict` contract used by existing detector-specific routes. `get_training_adapter()` is the lazy resolver for generic TrainingRuns and checks the separate `TrainableDetectorAdapter` protocol only when its worker child starts.

Schema version 1 requires:

- `schema_version`
- `id`
- `name`
- `version`
- `description`
- `supported_modalities`
- `capabilities`
- `input_requirements`
- `training_parameters`
- `entry_point`

Each training parameter has `type`, `default`, and `description`. Nullable defaults must explicitly set `nullable: true`.
Optional UI-facing constraints are `minimum`, `exclusive_minimum`, `maximum`,
`exclusive_maximum`, `allowed_values`, `excluded_values`, and `advanced`.
Discovery validates these values without importing the adapter. Isolation
Forest advertises only `auto` for `max_samples` and `contamination` and excludes
`0` for `n_jobs`, matching runtime validation.

A generically trainable plugin declares:

```yaml
capabilities:
  training:
    enabled: true
    protocol: cloudsentinel.training/v1
    input_profile: metrics-partition-v1
```

The v1 training adapter implements `validate_training(context)` and
`run_training(context, progress)`. It returns a `TrainingResult` containing a
validated `ArtifactResult`, compact evaluation and model metadata, and a
`PromotionResult`. The executor owns TrainingRun lifecycle persistence and
Saved Model creation.

`metrics-partition-v1` supplies immutable, checksummed train, test, and label
CSV files plus dataset and feature provenance. Labels are required by this
profile because both current integrations evaluate during training. This is a
profile constraint rather than a universal adapter assumption. A future
`metrics-partition-v2` can make labels optional once the Data Catalogue bundle
schema and evaluation lifecycle support that distinction.

`GET /detectors` returns:

```json
{
  "detectors": [
    {
      "schema_version": 1,
      "id": "cgnn"
    }
  ],
  "discovery_errors": []
}
```

The abbreviated object above is only illustrative; the endpoint returns full validated manifests and safe discovery errors without absolute paths or tracebacks.

Plugins are trusted server-side code installed by a developer or operator. There is no browser Python upload, runtime dependency installation, or code execution during discovery. Dependencies must already be present in the worker image.

Built-in manifests are discovered from the installed `detectors` package.
Additional immutable releases can be mounted through the additive
`DETECTOR_PLUGIN_ROOTS` setting. Public discovery identifies entries as
`source: builtin` or `source: external` and removes private entry points and
storage information. See [External detector plugin repository](external-detector-plugins.md)
for package installation, activation, rollback, integrity checks and exact
TrainingRun version pinning.

For local development, install the service requirements and shared package from the repository root:

```shell
python -m pip install -r learning_adaptation/requirements.txt
python -m pip install -e .
```

Build the learning-adaptation image from the repository root so the shared package is available in the build context:

```shell
docker build -f learning_adaptation/Dockerfile -t learning_adaptation .
```

Build the CGNN detection image from the same repository-root context:

```shell
docker build -f anomaly_detection/cgnn/Dockerfile -t anomaly_detection_cgnn .
```

Run the focused tests with:

```shell
python -m unittest tests.test_detectors tests.test_detector_resolver tests.test_cgnn_adapter
```

Generic TrainingRuns publish inference artifacts atomically to
`MODEL_ARTIFACT_STORAGE_ROOT`. A successful publication creates an
inference-ready Saved Model; detector adapters do not contact a
detector-specific service. The old binary Isolation Forest promotion helper is
retained only for the legacy route/test surface and is not imported or invoked
by the manifest-driven TrainingRun adapter.
