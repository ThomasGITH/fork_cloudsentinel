# NaaVRE read-only operational evaluation

This directory contains the bounded evaluation-only connection from CloudSentinel
to the Prometheus datasource exposed through NaaVRE's Grafana instance. It is kept
on the dedicated `codex/naavre-grafana-proxy` branch and is not part of the main
CloudSentinel implementation.

The collector uses aggregate queries only. It never writes authentication material
or raw Prometheus labels to its result. Until an environment owner supplies or
confirms incident timestamps, these results support only connection, portability,
provenance, freshness, and query-overhead claims. They do not support claims about
anomaly-detection or RCA accuracy.

`run_cloudsentinel_catalogue_case.py` additionally runs a bounded, multivariate
dataset fetch through CloudSentinel's normal catalogue assembler and then reuses
the resulting immutable input recipe for a live-window smoke test. Raw operational
telemetry is held in a temporary directory and deleted after the summary and
artifact hashes have been produced.

Example:

```shell
python evaluation/naavre/collect_read_only_evidence.py \
  --token-file /secure/path/to/token.txt \
  --output evaluation/naavre/results/read_only_evidence.json
```
