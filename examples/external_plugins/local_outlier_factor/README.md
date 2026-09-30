# Local Outlier Factor external detector

This package is CloudSentinel's first real external detector example. Local
Outlier Factor (LOF) compares the local density around each observation with
the density around its nearest neighbors. It is useful when anomalies occupy
sparser local regions than normal metric observations.

The package is not under `detectors/` and built-in discovery does not load it.
It becomes visible only after a trusted operator publishes it to the external
Detector Plugin PVC.

## Runtime requirements

The `python-ml-cpu/v1` runtime must already contain the exact versions pinned
in `requirements.lock`. Publishing validates those versions and never runs
`pip install`.

The model is persisted as a joblib-serialized scikit-learn pipeline:

```text
StandardScaler → LocalOutlierFactor(novelty=True)
```

`novelty=True` is required because CloudSentinel evaluates on observations
that were not used for fitting and a future generic inference runtime will do
the same for live observations.

## Parameters

- `n_neighbors`: local neighborhood size; it must be smaller than the number
  of training observations.
- `metric`: allowed distance metric (`minkowski`, `euclidean`, `manhattan` or
  `chebyshev`).
- `contamination`: currently fixed to scikit-learn's `auto` behavior.
- `algorithm`: advanced nearest-neighbor search strategy.
- `leaf_size`: advanced BallTree/KDTree leaf size.
- `p`: advanced Minkowski power parameter.

## Data and evaluation

The plugin uses `metrics-partition-v1`:

```text
train.csv  → fit StandardScaler and LOF
test.csv   → predict and evaluate
labels.csv → evaluation only
```

Labels always mean `0 = normal` and `1 = anomaly`. They are never passed to
pipeline fitting, scaling, threshold selection or feature transformation. The
current v1 bundle nevertheless requires labels so the training flow can write
its evaluation summary.

The artifact directory contains:

```text
model.joblib
model_metadata.json
model_evaluation.json
```

Scores use `-pipeline.decision_function(X)`, so a higher score means more
anomalous. Scikit-learn prediction `-1` becomes CloudSentinel anomaly label
`1`, while prediction `+1` becomes normal label `0`.

## Publish on EC2

From a CloudSentinel repository checkout, use the absolute package path:

```shell
./scripts/external_plugin.sh publish \
  /absolute/path/to/local_outlier_factor \
  --id local-outlier-factor \
  --version 1.0.0 \
  --activate
```

After publication, refresh **Models → Detector Library**. The detector should
appear with source `external` and can use the generic **Train models** flow.

## Current limitations

- Training uses only the labelled `metrics-partition-v1` profile.
- `contamination` supports only `auto` in this first version.
- Input must be finite numeric metrics with stable positional feature order.
- No LOF-specific inference service or promotion call exists.
- Live prediction awaits the future Generic Detection Runtime.
- Plugin dependencies must already exist in the selected runtime image.
