# Historical anomaly-detection robustness

COMPARISON-ROBUSTNESS-1 defines robustness as variation in the performance of
the same immutable Saved Model artifact across stored, independent evaluation
contexts. It does not produce a composite robustness score and does not
compare different artifact versions as though they were one model.

## Identity and eligible records

An artifact series is pinned by all three values:

```text
model_id
artifact_id
artifact_manifest_sha256
```

Only `completed` and `partial` ComparisonRuns participate. Within those runs,
only individual model results with `status=completed` participate. Failed
runs and failed model results remain visible in their original ComparisonRun
but cannot influence robustness statistics.

The evaluation context retains dataset ID, dataset version, partition ID,
partition checksum, modality, workload context, anomaly scenario, ground-truth
availability and safe incident-window metadata. Repeated results for the same
artifact identity and partition checksum are deduplicated. The most recent
terminal ComparisonRun is canonical. The UI states this rule explicitly and
links every retained context to its source ComparisonRun.

## Metrics

For each exact artifact, the API reports distinct context count, labelled
context count, median precision, median recall, median F1, minimum/maximum F1,
F1 range, population standard deviation when at least two F1 values exist,
runtime median/minimum/maximum, and lead-time availability plus its
median/minimum/maximum.

Accuracy and lead-time statistics use only contexts where those values are
actually available. Unlabelled contexts still contribute runtime and context
coverage, but never receive fabricated precision, recall, F1 or lead time.
At least two distinct contexts are required before the UI describes the data
as sufficient for robustness assessment.

## API

```text
GET /api/comparisons/<comparison_id>/robustness
```

Optional filters are:

```text
workload=Low|Normal|High|Variable
scenario=Normal operation|CPU stress|Memory stress|Network delay|Unknown
labelled_only=true|false
shared_only=true|false
```

The response uses aggregation contract
`cloudsentinel.comparison-robustness/v1`. It contains safe artifact identity,
per-model summaries, canonical context rows, a matrix for contexts shared by
at least two currently selected artifacts, coverage warnings, deduplication
counts and source ComparisonRun IDs. It never contains raw matrices,
predictions, scores, filesystem paths, internal URLs, task IDs, entry points or
tracebacks. Responses are capped at 200 contexts per model by default through
`COMPARISON_ROBUSTNESS_MAX_CONTEXTS_PER_MODEL`.

The Comparison list endpoint also accepts the complete set of `model_id`,
`artifact_id` and `artifact_manifest_sha256` to show related runs for one exact
artifact. Partial identity filters are rejected.

## UI behavior

The Robustness tab focuses on artifacts selected by the current ComparisonRun.
It shows aggregate coverage, accuracy and runtime summaries, a shared-context
matrix for fair side-by-side evaluation, and historical context rows. Missing
values are shown as `Unavailable`. Coverage differences are stated explicitly;
the UI does not declare a winner when artifacts were evaluated on different
contexts.

The implementation is read-only and computes from immutable ComparisonRun
files on request. It does not rerun inference or rewrite historical records.

## Staging verification

1. Build and load `jojojochem/learning_adaptation:comparison-robustness-1`
   and `jojojochem/monitoring_project:comparison-robustness-1` using the build
   contexts in `docs/comparison-mvp.md`.
2. Apply both Learning manifests and the monitoring-project manifest, then
   wait for all three rollouts.
3. Produce completed comparisons that use the same exact Saved Model artifact
   on at least two distinct partition checksums and workload contexts. Reusing
   only the detector name is intentionally insufficient.
4. Open either ComparisonRun and select **Robustness**. Verify the context
   count, labelled coverage, aggregate metrics and historical links.
5. Repeat one comparison on the same partition. Verify that the newest result
   replaces the earlier result and that the deduplication notice appears.
6. Add an unlabelled context. Verify that runtime and coverage appear while
   precision, recall, F1 and lead time remain `Unavailable`.
7. Exercise workload, scenario, labelled-only and shared-only filters. Confirm
   that every context link opens its immutable source ComparisonRun.

## Known limits

- Aggregation is a bounded on-demand scan of file-backed ComparisonRuns; no
  cache or database index is introduced.
- Old records without a partition checksum cannot participate because their
  evaluation identity is not strong enough for safe deduplication.
- Detection lead time is aggregated only when the stored result carries the
  existing non-negative, before-or-at-incident semantic.
- Robustness remains anomaly-detection-only. RCA, export and automatic actions
  are outside scope.
