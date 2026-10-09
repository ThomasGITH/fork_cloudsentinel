# Generic live monitoring: provenance gap and migration boundary

This audit covers the existing monitoring path and the minimum safe path to
replace its CGNN-specific behavior with explicit Saved Model selection. It does
not introduce a second monitoring flow.

## Current path

The Django monitoring form gets pods from Kubernetes and gets models and metric
names from the legacy CGNN detection service. It sends the chosen containers,
metrics and timing settings to `data_ingestion`.

`data_ingestion.monitoring_task` repeatedly:

1. builds a sliding time range;
2. looks up PromQL templates in the legacy data-ingestion `config.json`;
3. expands every selected container × metric combination;
4. queries Prometheus;
5. aligns samples to a time grid and applies `ffill`, then `bfill`, then zero;
6. sends a headerless matrix to `/preprocess_cgnn_data`.

`data_processing.cgnn_preprocess` already requires an explicit Saved Model ID,
loads its feature identity from the Learning API, activates that model in the
Generic Detection Runtime, and invokes generic `/detect`. This final inference
hop is generic, but collection, model discovery, result retrieval and the UI
are still coupled to the legacy CGNN service and its configuration.

## What is stored today

The Data Catalogue version retains the original query templates, resolved
queries, requested/resolved targets, sampling interval and canonical feature
order. Its canonical assembly uses a fixed timestamp grid and represents
missing samples as empty values.

The training bundle and immutable snapshot retain:

- dataset ID, version and partition ID/checksum;
- feature order and feature-order hash;
- sampling step in `dataset_details.step_size`;
- source artifact checksums and a provenance artifact reference;
- label source and train/test counts.

The Saved Model retains the dataset/partition identity, snapshot identity,
feature order/hash, parameters, artifact identity and stored evaluation. It
does **not** retain or expose a complete executable live-input recipe.

## Blocking contract gap

The model/snapshot boundary does not contain all of the following as one
versioned, immutable and verified contract:

- Prometheus source identifier;
- PromQL templates and their execution modes;
- target type, target binding and identity-label rules;
- the mapping from every resolved series to every feature ID;
- sampling/alignment tolerance;
- missing-data policy;
- live range/window size required before inference;
- detector windowing/warm-up requirements in a host-consumable form.

The stored provenance reference is not enough: the learning and monitoring
services cannot safely turn that reference into queries, and old versions or
legacy datasets may not have equivalent provenance. The legacy monitoring
collector's container × metric names also cannot be inferred from arbitrary
PromQL-derived catalogue feature names. Its fill policy differs from catalogue
materialization and is not model provenance.

For these reasons the existing monitoring UI/task must not yet be switched to
arbitrary Saved Models. Doing so would require a guessed query mapping or a
silent preprocessing change.

## Smallest safe in-place migration

1. Add a versioned `live_input_recipe` to available catalogue versions. Freeze
   query templates, execution/target binding, sampling, series-to-feature
   identity, alignment and missing-data policy. Give the recipe a SHA-256.
2. Include that recipe and digest in the training bundle and immutable snapshot,
   then pin a safe recipe projection in each new Saved Model record. Old models
   remain explicitly `live_monitoring: not_ready`.
3. Add server-side validation that the recipe produces exactly the model's
   stored feature order/hash. Do not accept a browser-provided Prometheus URL or
   query override.
4. Adapt the existing `/start_monitoring`, `/stop_monitoring` and monitoring
   task in place. Require an explicit inference-ready `model_id`; resolve its
   pinned recipe; collect the bounded matrix; apply only the pinned alignment
   and missing-data policy; call the existing Generic Detection Runtime.
5. Maintain a per-monitoring-session observation buffer. The adapter/runtime
   remains responsible for detector-specific windowing; the session must retain
   enough rows for declared warm-up/lookback and must display warm-up honestly.
6. Replace legacy CGNN model/config/result lookups in the existing Django
   monitoring pages with Learning model metadata and generic result/status
   endpoints. Show the selected model ID/name, recipe readiness, collection
   status, warm-up and safe errors. Never auto-switch models.
7. After parity tests, remove the remaining CGNN service URL, fixed metric
   configuration and legacy result caller from the active monitoring path.

This sequence supports multiple Saved Models without detector-ID dispatch. A
model becomes selectable only when both generic inference and its pinned live
input recipe are ready.

## Remaining legacy CGNN dependencies

- Django monitoring setup calls the CGNN service for available models and
  configuration.
- Monitoring result pages read and delete results through CGNN endpoints.
- The monitoring manifest still configures `API_CGNN_ANOMALY_DETECTION_URL`.
- Data ingestion expands fixed container × metric configuration and posts to a
  route named `/preprocess_cgnn_data`.
- The monitoring form and result language are CGNN-specific.
- The data-processing compatibility route is generic internally, but retains a
  CGNN name and accepts a caller-assembled matrix rather than a pinned recipe.

These dependencies should be removed by adapting the current monitoring flow,
not by building a parallel live-monitoring module.
