# Kubernetes topology-scaling evaluation

This directory contains the reproducible protocol and raw evidence for the
topology-scaling evaluation executed on 26 October 2026. The evaluation used a
second Online Boutique installation in the isolated `boutique-scale-test`
namespace. The existing CloudSentinel and original Online Boutique
installation were left in place.

## Question

Can CloudSentinel still collect a stable, traceable metric dataset when the
monitored application grows from single replicas to a larger Deployment-based
topology whose pod names change?

This is a topology and integration test. It is not a throughput benchmark:
the temporary load generator remained paused, apart from HTTP smoke requests.

## Configurations

| Configuration | Online Boutique topology | Result |
|---|---|---|
| A | 11 application pods: one pod per service, with the temporary load generator paused | Healthy |
| B | Five stateless services scaled from one to three replicas; 21 application pods in total | Healthy |
| C | All scalable stateless Deployments at three replicas; Redis and the chart's standalone checkout pod at one; 29 application pods in total | Healthy |
| D | Attempted second Minikube worker on the same EC2 host | Worker could not start; treated as an environment limitation |

CPU scheduler reservations in the chart were lowered only for this isolated
installation. CPU limits were not raised. Two Python services also received a
longer startup-probe delay because the chart started liveness checks before
they could boot on the shared host.

## Results

Each healthy configuration was captured three times. All nine frontend smoke
requests returned HTTP 200. The figures below are the three observed values,
not a claim about maximum capacity.

| Configuration | Node CPU | Node memory | Host memory available | Frontend response time |
|---|---|---|---|---|
| A | 35%, 35%, 31% | 43%, 43%, 43% | 8,564, 8,559, 8,558 MiB | 0.201, 0.185, 0.087 s |
| B | 31%, 31%, 31% | 42%, 44%, 44% | 8,361, 8,327, 8,315 MiB | 0.555, 0.046, 0.038 s |
| C | 23%, 33%, 33% | 44%, 46%, 46% | 7,963, 7,950, 7,935 MiB | 0.019, 0.032, 0.038 s |

The first Data Catalogue fetch covered the transition from A to B. It stored
17 observations with 10 service-level CPU and memory features. All 10 query
executions succeeded, validation passed, and the stored matrix contained zero
missing values. The fetch also stored checksums, resolved-query provenance,
and a live-input recipe.

The frontend Deployment was then restarted. All three original pod names were
replaced. During the rolling restart, 44 of 45 one-second HTTP probes succeeded
and one timed out. A second Data Catalogue fetch after this name change stored
the same 10 features with zero missing values. Its live-input recipe checksum
was identical to the first fetch:

`4d1c616d02d17bc93c60942deaa7216c53b34e61a9d0bfdb7c22943c8c7d1178`

This worked because the evaluation queries selected and aggregated pods by a
stable service-name pattern such as `frontend(-.*)?`. CloudSentinel's older
default metric configuration instead substitutes an exact pod name. That
legacy route does not retain identity across Deployment pod replacement.

Configuration D did not reach Kubernetes. The second Minikube container
stopped in `systemd` with `Too many open files` while creating its cgroup
inotify object. Raising `fs.inotify.max_user_instances` from 128 to 512 and
then 1024 did not resolve the error, so the original value of 128 was restored.
CloudSentinel and configuration C remained healthy afterward. Because both
Minikube containers would share one four-vCPU EC2 host, D would in any case
only test multi-node scheduling topology, not added physical capacity.

## Interpretation

The results show that the redesigned Data Catalogue can retain stable feature
identity while a monitored Deployment grows and its pod names change, provided
that the dataset definition uses stable service-level queries. It handled the
tested increase from 11 to 29 Online Boutique pods without fetch errors,
missing feature values, or CloudSentinel failures. The evaluation also exposes
two boundaries: the exact-pod legacy path is not safe for changing pod names,
and this single-host Minikube setup could not provide a second worker node.

## Evidence

The `evidence/` directory contains the unabridged snapshots, Prometheus query
results, Data Catalogue API responses, checksums, rolling-restart probe log,
and worker failure log. `capture_snapshot.sh`, `run_catalogue_fetch.py`, and
`low_request_values.yaml` document the test procedure.

## Temporary resources still present

No cleanup was performed. The following items should be reviewed with the
operator before removal:

- the `boutique-scale-test` namespace and its `onlineboutique-scale` Helm release;
- the stopped partial Minikube worker container `minikube-m02`;
- the two Data Catalogue records created by this evaluation, if they are not
  retained as thesis evidence.

