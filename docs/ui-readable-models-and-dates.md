# Readable model names and dates

CloudSentinel keeps immutable Saved Model IDs in APIs, links and provenance, but
uses a readable model label as the primary text in Models, Training History,
Comparison and monitoring setup. If no explicit display name was stored, the UI
derives a label from detector name, detector version and training time. This is
presentation-only and does not change model selection or identity.

Visible timestamps use the unambiguous format `10 Oct 2026, 17:20 UTC`.
API values, signed wizard state and `datetime-local` form values remain ISO 8601.
Server-rendered templates use the shared `ui_datetime` filter; dynamically
updated pages use `CloudSentinel.formatDateTime` with the same UTC semantics.

Only the Django monitoring image changes:

```bash
docker build \
  -t jojojochem/monitoring_project:ui-readable-models-dates-1 \
  ./user_interface/monitoring_project

minikube image load --overwrite=true \
  jojojochem/monitoring_project:ui-readable-models-dates-1

kubectl apply -f k8s/monitoring_project-deployment.yml
kubectl rollout status deployment/monitoring-project-deployment \
  -n cloudsentinel --timeout=180s
```

No Learning, data-ingestion, generic detection or CausalRCA image is required
for this presentation update.
