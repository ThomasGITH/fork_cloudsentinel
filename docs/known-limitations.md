# Existing technical limitations

These limitations predate the detector plugin work and remain intentionally unchanged through Phase C:

- Training and evaluation do not consistently interpret boolean configuration values: some paths use booleans while others compare values with the string `"True"`.
- Training configuration is stored in a shared relative `config.json`, so concurrent runs can affect one another.
- CGNN imports and model paths depend on each service's current working directory.
- Training and online detection use different normalization behaviour. Training fits `MinMaxScaler` on training data and applies it to test data; online detection fits it on the submitted detection data.
- Training labels have inconsistent boolean/string handling for the anomaly-sequence option.
- The manual detection upload supplies less request metadata than the monitoring-oriented `/detect_anomalies` route reads.
- The Kubernetes learning-adaptation API and Celery worker do not declare shared persistent model storage.
- Only the bundled CGNN detector is present. The registry can resolve manifest entry points, but plugin trust, isolation, dependency management, and installation are not implemented.
