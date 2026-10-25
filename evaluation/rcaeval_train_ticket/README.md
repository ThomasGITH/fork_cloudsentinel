# RCAEval Train Ticket external validation

This directory contains the frozen protocol and repeatable preparation code for
the second-environment evaluation. It does not modify CloudSentinel's product
code.

The evaluation uses 25 RCAEval RE1 Train Ticket cases: one case for each pairing
of five root services and five fault types. The chosen repetitions follow the
Latin-square rule in `protocol.json`, so neither the service, fault type, nor
repetition determines the outcome sample by itself.

The source cases contain roughly 198--238 metrics, while CloudSentinel's current
CSV catalogue contract accepts at most 100. The preparation step therefore
constructs one fixed profile from metrics shared by all selected cases and with
at least one finite observation in every case. It prioritises workload and
latency coverage across services and CPU/memory
coverage for all five possible root services. This is a declared compatibility
adaptation and an evaluation result for portability; it is not presented as a
property of RCAEval.

Run with a Python environment containing pandas, pyarrow, NumPy and
scikit-learn:

```text
python prepare_and_evaluate.py --output-dir prepared
```

The output contains:

- source files and SHA-256 hashes;
- canonical CSV files accepted by CloudSentinel (`timestamp`, 100 features,
  `label`);
- the exact common feature profile;
- per-case preparation evidence;
- separate local Isolation Forest and LOF diagnostic results.

The local detector diagnostics are a reproducible check of whether the selected
historical cases contain usable anomaly signal. They are not labelled as
CloudSentinel execution results. Application-level import, training, comparison,
and replay evidence is retained separately after running the deployed workflow.

## Calibrated held-out evaluation

`prepare_calibrated_pairs.py` prepares five service/fault pairs. Each pair uses
one repetition for parameter selection and a different repetition for the final
test. `run_calibrated_evaluation.py` trains Isolation Forest, Local Outlier
Factor, and CGNN through CloudSentinel. The final model is fitted only on the
normal baseline at the start of the held-out run. Labels from the later period
are used only to calculate the final evaluation metrics.

`rerun_calibrated_comparisons.py` can repeat the comparison stage without
retraining, and `summarize_calibrated_results.py` produces the compact aggregate
and per-case tables used in the thesis. Generated datasets, run state, logs, and
result files are stored below `prepared/` and are intentionally not committed.

BARO ranking can be evaluated with the exact pinned `fse-baro==0.2.2` package:

```text
python evaluate_baro.py --prepared-dir prepared
```

This uses the published injection time as the change point. It therefore tests
whether the known root service is prioritised by the ranking, not whether an
anomaly or change point can be detected without that ground truth.

`run_cloudsentinel_evaluation.py` imports all 25 canonical datasets into a
deployed CloudSentinel instance, trains Isolation Forest and Local Outlier
Factor once on the pre-injection part of the frozen base case, and compares both
unchanged Saved Models on every service/fault context. Its state file retains
the returned dataset, partition, training, model, comparison, metric, timing,
and robustness records and allows an interrupted run to resume.
