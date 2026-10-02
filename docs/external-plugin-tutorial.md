# Publish an external detector plugin

This tutorial is the normal operator workflow for adding a trusted detector to
CloudSentinel. You create a plugin folder and run one repository script. The
script handles the temporary Kubernetes administration pod, validation,
immutable installation, optional activation and cleanup.

External plugins are trusted server-side Python code. They are stored on the
Detector Plugin PVC instead of in the CloudSentinel application image. The
Learning API and Learning Worker mount that repository read-only. CloudSentinel
does not offer browser upload of Python code and never installs plugin
dependencies at runtime.

## 1. Create the package

The minimum layout is:

```text
my_detector/
├── manifest.yaml
├── adapter.py
└── requirements.lock
```

You may include additional Python modules and import them relatively, for
example `from .implementation import fit_model`. An external manifest uses a
plugin-relative entry point:

```yaml
schema_version: 1
id: my-detector
name: My detector
version: 1.0.0
description: Detects anomalies in multivariate metric observations.
supported_modalities: [metrics]
runtime:
  profile: python-ml-cpu/v1
capabilities:
  training:
    enabled: true
    protocol: cloudsentinel.training/v1
    input_profile: metrics-partition-v1
input_requirements:
  training:
    format: Numeric two-dimensional matrix
    finite_values_required: true
training_parameters: {}
entry_point: adapter:MyDetectorAdapter
```

To start from a commented skeleton, copy the repository template:

```bash
cp -R examples/external_plugins/template_detector \
  /home/ubuntu/plugins/my_detector
```

The template is intentionally not publishable until all `<replace-...>` values
are replaced and its adapter methods are implemented.

### Required manifest fields

The current schema requires:

- `schema_version`: currently exactly `1`;
- `id`: unique lowercase identifier using letters, digits, `-` or `_`;
- `name`: user-facing Detector Library name;
- `version`: immutable release version;
- `description`: user-facing explanation;
- `supported_modalities`: non-empty list, currently normally `[metrics]`;
- `capabilities.training.enabled`: required boolean;
- `capabilities.training.protocol` and `input_profile` when training is enabled;
- `input_requirements`: required mapping describing accepted data;
- `training_parameters`: required mapping, which may be empty;
- `entry_point`: plugin-relative `module:ClassName`, normally
  `adapter:MyDetectorAdapter`.

`input_requirements` is descriptive compatibility metadata. The adapter must
still enforce dataset-dependent conditions in `validate_training()`.

### Optional manifest fields

- `runtime.profile`; omission currently selects `python-ml-cpu/v1`;
- `capabilities.inference`; when enabled, `protocol`, `input_profile` and
  `artifact_format` become required;
- individual parameter constraints such as bounds, allowed values and UI
  grouping;
- additional descriptive nested input-requirement fields.

The currently supported training contract is
`cloudsentinel.training/v1` with `metrics-partition-v1`. The current generic
inference contract is `cloudsentinel.inference/v1` with
`metrics-matrix/v1`.

### Training parameters

Every entry under `training_parameters` requires:

```yaml
parameter_name:
  type: integer
  default: 10
  description: User-facing explanation of the parameter.
```

Supported types are `boolean`, `integer`, `number`, and `string`. Optional
metadata includes:

- `minimum` or `exclusive_minimum`;
- `maximum` or `exclusive_maximum`;
- `allowed_values`;
- `excluded_values`;
- `nullable: true`, required when the default is `null`;
- `advanced: true|false` for UI grouping.

Defaults and allowed values must match the declared type and bounds. A default
must be included in `allowed_values` and may not appear in `excluded_values`.
Unknown submitted parameter names are rejected and never forwarded to the
adapter. Relationships that cannot be expressed as a simple bound, such as
`n_neighbors < training observations`, belong in `validate_training()`.

### Adapter contract

An enabled training plugin must expose a class that can be constructed without
arguments and implements:

```python
def validate_training(context: TrainingContext) -> None: ...
def run_training(
    context: TrainingContext,
    progress: ProgressReporter,
) -> TrainingResult: ...
```

The adapter owns detector-specific validation, preprocessing, fitting,
evaluation and artifact creation. It may write only inside the supplied
artifact workspace. The generic executor owns run lifecycle, artifact
validation/publication and Saved Model creation.

When inference is enabled, the same adapter must additionally implement:

```python
def load_model(context: ModelLoadContext) -> Any: ...
def predict_inference(
    model: Any,
    context: InferenceContext,
) -> PredictionResult: ...
```

Predictions use `0 = normal`, `1 = anomaly`, and larger scores must mean more
anomalous.

Use a unique detector ID. External plugins cannot replace built-in detector
IDs. The version identifies an immutable release, so publish changed content
under a new version.

`requirements.lock` contains exact pins such as:

```text
numpy==1.26.4
scikit-learn==1.5.0
```

It is a compatibility declaration, not an installation request. Every listed
package and exact version must already exist in the learning runtime image.
Adding a dependency to this file does not install it. Runtime profiles state
which prebuilt environment the plugin expects. Training capability makes the
plugin eligible for the generic TrainingRun flow. Inference capability makes a
correctly published model eligible for the Generic Detection Runtime when its
artifact contract is supported by the adapter.

## 2. Publish and activate

Run the wrapper from a CloudSentinel repository checkout on the EC2 instance.
The plugin path must be absolute:

```shell
./scripts/external_plugin.sh publish \
  /home/ubuntu/plugins/my_detector \
  --id my-detector \
  --version 1.0.0 \
  --activate
```

The `--id` and `--version` values are assertions: the script rejects the
package if they do not match its manifest. `--activate` is explicit because
installation and activation are deliberately separate operations.

Internally, the wrapper performs:

```text
temporary plugin-admin pod
→ package copy to private staging
→ manifest, dependency and adapter validation
→ immutable release installation
→ explicit activation
→ staging and pod cleanup
```

You do not normally need to run `kubectl apply`, `kubectl cp`, `kubectl exec`
or `cloudsentinel-plugin` yourself. The lower-level CLI remains available as a
technical recovery tool. Pass `--keep-admin-pod` only while troubleshooting;
otherwise the wrapper deletes the temporary pod after success and failure.

No detector-specific image, Deployment or Service is created. Compatible
adapter code therefore needs no application image rebuild.

## 3. Confirm the result

Refresh **Models → Detector Library**. The detector should appear with source
`external` and can then be selected in **Train models** when its modality and
input profile match the selected dataset.

For a technical API check, access the Learning API through the existing safe
port-forward or application route and inspect:

```text
GET /detectors
```

Do not expose the internal Kubernetes service publicly merely to perform this
check.

## 4. Add another detector

Create another package folder with a different detector ID and run the same
command:

```shell
./scripts/external_plugin.sh publish \
  /home/ubuntu/plugins/another_detector \
  --id another-detector \
  --version 1.0.0 \
  --activate
```

The preceding publication normally removed the temporary admin pod. That is
expected: the script automatically recreates it for every later publication.

## 5. Upgrade safely

First give changed plugin content a new manifest version. To install it without
changing the currently active release:

```shell
./scripts/external_plugin.sh publish \
  /home/ubuntu/plugins/my_detector_1_1 \
  --id my-detector \
  --version 1.1.0
```

The immutable 1.1.0 release is then installed while 1.0.0 remains active. To
make the new release active for newly created TrainingRuns, publish with the
explicit switch:

```shell
./scripts/external_plugin.sh publish \
  /home/ubuntu/plugins/my_detector_1_1 \
  --id my-detector \
  --version 1.1.0 \
  --activate
```

Queued and running TrainingRuns retain their pinned plugin version and digest.
Activation changes which immutable release future runs select.

List installed and active releases with:

```shell
./scripts/external_plugin.sh list
```

## 6. Roll back

Rollback changes only the active reference. It does not rewrite or reinstall
the old immutable release:

```shell
./scripts/external_plugin.sh rollback my-detector 1.0.0
```

Future TrainingRuns select 1.0.0 after the switch. Existing pinned runs remain
unchanged.

## Common errors

### `plugin directory is missing manifest.yaml`

Check the package layout and pass the directory containing `manifest.yaml`,
not its parent. `adapter.py` and `requirements.lock` are required too.

### Reserved or duplicate detector ID

Built-in IDs cannot be overridden. Choose a unique external detector ID. Two
different active plugins cannot use the same ID.

### Runtime dependency is unavailable or incompatible

`requirements.lock` may only pin dependencies already present at the exact
compatible version in the runtime image. The publication process never runs
`pip install`.

### Adapter import or contract validation failed

Check the plugin-relative entry point, relative imports and the required
training adapter methods. Validate imports against the same dependency set as
the deployed learning image. CloudSentinel keeps the preceding active release
unchanged.

### Version/content conflict

Published releases are immutable. If code changes, increment the manifest
version and publish that version. Do not reuse a version for different bytes.

### Plugin is installed but not visible

Only an active release appears in Detector Library. Publish with `--activate`
or activate an already installed release through the technical operator CLI.
Then refresh the page. Also confirm that the Learning API and worker mount the
same Plugin PVC read-only and use the configured external active root.

### Kubernetes permission failure

The operator account needs permission to apply, wait for, exec into and delete
the `plugin-admin` pod, and to copy files to it in the `cloudsentinel`
namespace. It does not need to modify the Learning deployments for each
plugin.

### Admin pod remains after an error

The wrapper uses an exit trap to delete the temporary pod on success and
failure. A terminated SSH session or unavailable Kubernetes API can prevent
cleanup. Delete the `plugin-admin` pod manually when connectivity returns, or
rerun the wrapper; it safely recreates or reuses the administration pod.

Use `--keep-admin-pod` only when inspecting a publication problem. Remember to
remove that pod afterward because it has read-write access to the Plugin PVC.
