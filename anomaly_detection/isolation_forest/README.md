# Isolation Forest detection results

`POST /detect_anomalies` stores compact results at
`{IF_RESULTS_STORAGE_ROOT}/{task_id}/isolation_forest_results.json`. Writes use
a temporary file in the task directory followed by an atomic replacement.

Iterations are immutable request identities: submitting an existing
`task_id` and `iteration` returns HTTP 409 before prediction or RCA is repeated.
The service does not coordinate writes between multiple service replicas.

When `percentage > crca_threshold`, the service first stores the prediction and
then calls the existing data-ingestion `/anomaly_rca` endpoint. A successful RCA
task ID is added atomically to the iteration. An RCA failure is stored as
`crca_error`; the detection endpoint returns HTTP 502 with `status: rca_failed`
while preserving the prediction result.
