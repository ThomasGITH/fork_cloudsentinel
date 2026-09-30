# External detector plugin repository

CloudSentinel loads immutable built-in detector plugins from the installed
`detectors` Python package. Trusted operators can additionally install plugins
on a persistent repository mounted at `/opt/cloudsentinel/detectors`. This is
an operator workflow: there is no browser upload, HTTP installer, runtime
`pip install`, or per-plugin environment.

## Package layout

An external package contains at least:

```text
my_detector/
├── manifest.yaml
├── adapter.py
└── requirements.lock
```

Additional Python modules are allowed. Use plugin-relative imports such as
`from .implementation import train_model`. The manifest entry point is also
plugin-relative, for example `adapter:MyDetectorAdapter`. External code may
only use dependencies already present in the selected runtime image.

`requirements.lock` is declarative and accepts pinned `name==version` lines.
The installer compares these pins with the current runtime and never installs
packages. A missing or incompatible dependency prevents installation.

## Repository lifecycle

The operator CLI copies a source package to staging, validates its files,
manifest, dependency pins and adapter contract, calculates deterministic
SHA-256 hashes, and atomically publishes an immutable release:

```text
/opt/cloudsentinel/detectors/
├── staging/
├── releases/<detector-id>/<version>/sha256_<digest>/
├── active/<detector-id> -> ../releases/...
└── repository.json
```

Installation does not activate a release. Activation atomically switches the
`active` symlink after integrity and runtime-readiness checks. Rollback uses
the same operation to select an older installed version. Releases are not
garbage-collected automatically because queued runs and saved models retain an
exact plugin reference.

```shell
cloudsentinel-plugin validate /staging/my_detector
cloudsentinel-plugin install /staging/my_detector
cloudsentinel-plugin activate my-detector 1.0.0
cloudsentinel-plugin list
cloudsentinel-plugin rollback my-detector 0.9.0
```

The CLI output contains logical IDs, versions, package checksums and safe
status values. It does not return repository paths or entry points.

## Discovery and version pinning

`DETECTOR_PLUGIN_ROOTS` contains additional absolute discovery roots separated
by the platform path separator. Kubernetes uses:

```text
DETECTOR_PLUGIN_ROOTS=/opt/cloudsentinel/detectors/active
```

The built-in package root is always scanned first. External plugins cannot
replace built-in IDs. Invalid external manifests and duplicate external IDs
are quarantined independently, while valid built-ins remain visible.
Discovery reads manifests only and never imports adapter code.

When a TrainingRun is created, each child stores the exact source, detector
version, package checksum, manifest checksum and runtime profile. The worker
verifies and loads that immutable release even if an operator activates a
newer version before the queued task starts. Saved Model records retain the
same reference. Old built-in run records without a plugin reference remain
supported.

## Recommended operator procedure

For normal trusted developer and administrator use, follow
[Publish an external detector plugin](external-plugin-tutorial.md). The
repository wrapper reduces publication to one command and automatically
creates, uses and removes the temporary admin pod.

The commands below document the lower-level technical fallback.

## Manual Minikube fallback

After a compatible application image containing the repository code has been
built as `jojojochem/learning_adaptation:external-plugins-1` and deployed,
create the PVC and temporary admin pod. This is the one application-image
upgrade needed to introduce the repository mechanism; later compatible plugin
code does not require rebuilding that image.

```shell
kubectl apply -f k8s/detector-plugin-repository-pvc.yml
kubectl apply -f k8s/plugin-admin-pod.yml
kubectl wait -n cloudsentinel --for=condition=Ready pod/plugin-admin --timeout=120s
```

Copy and install a local trusted package:

```shell
kubectl cp ./local_outlier_factor \
  cloudsentinel/plugin-admin:/opt/cloudsentinel/detectors/staging/incoming/local_outlier_factor

kubectl exec -n cloudsentinel plugin-admin -- \
  cloudsentinel-plugin validate \
  /opt/cloudsentinel/detectors/staging/incoming/local_outlier_factor

kubectl exec -n cloudsentinel plugin-admin -- \
  cloudsentinel-plugin install \
  /opt/cloudsentinel/detectors/staging/incoming/local_outlier_factor

kubectl exec -n cloudsentinel plugin-admin -- \
  cloudsentinel-plugin activate local-outlier-factor 1.0.0
```

Verify through the Learning API and remove the privileged temporary pod:

```shell
kubectl port-forward -n cloudsentinel service/learning-adaptation-service 5005:80
curl -fsS http://127.0.0.1:5005/detectors
kubectl delete pod -n cloudsentinel plugin-admin
```

The Learning API and worker mount the repository read-only. The admin pod is
temporary, has no Service, mounts the PVC read-write, runs non-root, and exits
after one hour if it is not deleted earlier.

## Current limits

- The first deployment assumes one Kubernetes node and an RWO volume.
- Multi-node use needs RWX storage or an immutable object-storage/GitOps flow.
- Plugins are trusted server-side code; import isolation is a namespace and
  integrity boundary, not an operating-system sandbox.
- Inference validation remains `unvalidated` until a future Generic Detection
  Runtime validates the same release against its runtime profile.
- Dependencies cannot be added through a plugin package.
- Release signing, automatic garbage collection and browser administration are
  outside the first implementation.
