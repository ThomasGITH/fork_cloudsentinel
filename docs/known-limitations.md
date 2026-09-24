# Existing technical limitations

These limitations predate the detector registry and are intentionally unchanged in Phase A:

- Training and evaluation do not consistently interpret boolean configuration values: some paths use booleans while others compare values with the string `"True"`.
- Training configuration is stored in a shared relative `config.json`, so concurrent runs can affect one another.
- CGNN imports and model paths depend on each service's current working directory.
- Training and online detection use different normalization behaviour. Training fits `MinMaxScaler` on training data and applies it to test data; online detection fits it on the submitted detection data.
- Training labels have inconsistent boolean/string handling for the anomaly-sequence option.
- The manual detection upload supplies less request metadata than the monitoring-oriented `/detect_anomalies` route reads.
- The Kubernetes learning-adaptation API and Celery worker do not declare shared persistent model storage.
- The manifest entry point is metadata only in Phase A. The adapter and execution-path integration are deferred to Phase B.
