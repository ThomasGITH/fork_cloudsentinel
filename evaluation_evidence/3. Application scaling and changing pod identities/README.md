# Application scaling and changing pod names
This file explains the protocols/observations/results of the application scaling/pod identity change evaluation test. 

## Setup — 26 October 2026

- Second Online Boutique installation in namespace `boutique-scale-test`.
- Existing CloudSentinel and original Online Boutique installation kept in place.
- Main question: does data collection still work when services get more replicas and replacement pods get new names?
- Also checked whether the collected data still had consistent feature definitions and records of the queries and settings used.
- Temporary load generator paused. Only HTTP requests to check frontend availability.
- No heavy-traffic or maximum-capacity testing.

## Configurations

| Configuration | Setup | Result |
|---|---|---|
| A | One pod per service; 11 application pods | Healthy |
| B | Five stateless services increased from one to three replicas; 21 application pods | Healthy |
| C | All scalable stateless Deployments at three replicas. Redis and the chart's standalone checkout pod kept at one; 29 application pods | Healthy |
| D | Attempt to add a second Minikube worker on the same EC2 host | Worker failed to start |

Setup adjustments for the separate test installation:

- Lowered CPU requests so Kubernetes could schedule the extra pods on the shared host.
- CPU limits were not increased.
- Increased the startup-probe delay for two Python services. The original checks started before the services had enough time to boot.

## Health and resource measurements

- Three measurements each for A, B and C.
- All nine frontend checks returned HTTP 200.
- Values below are individual measurements from this test, not capacity limits.
- Node = Minikube node; host = EC2 machine running Minikube. Memory shown in MiB.

| Configuration | Node CPU | Node memory | Available host memory | Frontend response time |
|---|---|---|---|---|
| A | 35%, 35%, 31% | 43%, 43%, 43% | 8,564, 8,559, 8,558 MiB | 0.201, 0.185, 0.087 s |
| B | 31%, 31%, 31% | 42%, 44%, 44% | 8,361, 8,327, 8,315 MiB | 0.555, 0.046, 0.038 s |
| C | 23%, 33%, 33% | 44%, 46%, 46% | 7,963, 7,950, 7,935 MiB | 0.019, 0.032, 0.038 s |

## Data Catalogue: adding replicas

- First collection covered the change from A to B.
- Stored 17 observations with 10 service-level CPU and memory features.
- Service-level queries combine matching pods into one measurement per feature.
- All 10 Prometheus queries succeeded.
- Validation passed; no missing values.
- Also stored file checksums, the exact queries used, and the recipe for collecting compatible Live Monitoring input later.

## Data Catalogue: replacing pods

- Rolling restart of the frontend Deployment.
- All three frontend pods replaced; all three names changed.
- HTTP checks sent once per second during the restart: 44/45 succeeded, one timed out.
- Second Data Catalogue collection after the restart:
  - Same 10 features.
  - No missing values.
  - Same live-input recipe checksum as before:

`4d1c616d02d17bc93c60942deaa7216c53b34e61a9d0bfdb7c22943c8c7d1178`

- Queries used stable service-name patterns, e.g. `frontend(-.*)?`, rather than one exact pod name.
- Matching replicas were combined into service-level measurements, so new pod suffixes did not change the feature definitions.
- The older default metric configuration still uses exact pod names. After a pod is replaced, that old name no longer identifies the replacement.

## Second worker attempt

- Configuration D failed before the worker joined Kubernetes.
- Second Minikube container stopped during `systemd` startup.
- Error: `Too many open files` while creating a cgroup inotify object.
- Tried increasing `fs.inotify.max_user_instances`: 128 → 512 → 1024.
- Neither increase fixed the problem. Restored the original value of 128.
- CloudSentinel and configuration C remained healthy afterward.
- Both workers would have shared the same four-vCPU EC2 host anyway. This would have tested scheduling across two nodes, without adding physical CPU or memory.

## Main findings and limits

- Tested increase from 11 to 29 application pods.
- Data Catalogue collections completed without query errors or missing feature values. No CloudSentinel failures observed.
- Feature definitions stayed consistent after frontend pod replacement because the queries used stable service-name patterns.
- This depends on how the queries are written. The older exact-pod-name route still has the replacement problem.
- No working second node, so no multi-node results.
- Load generator paused, so no conclusions about performance under heavy traffic or maximum capacity.

## Evidence

Original records include:

- Kubernetes state and resource snapshots.
- Prometheus query results.
- Data Catalogue API responses and checksums.
- HTTP check results during the rolling restart.
- Worker startup failure log.

Files used for the test:

| File | Purpose |
|---|---|
| `capture_snapshot.sh` | Capture application state and resource measurements |
| `run_catalogue_fetch.py` | Run Data Catalogue collection |
| `low_request_values.yaml` | Record adjusted resource settings |
