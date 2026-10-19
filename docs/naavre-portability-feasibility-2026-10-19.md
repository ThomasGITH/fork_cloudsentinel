# NaaVRE portability — read-only haalbaarheidsanalyse (2026-10-19)

## 1. Samenvatting

- De service-accountauthenticatie, Grafana-metadata-API en datasource-query-API zijn getest en werken (HTTP 200).
- Datasource `prometheus` bestaat, is standaard en read-only; productie versus staging blijft onbevestigd.
- Gabriels `up{node="k8s-staging-1"}`-voorbeeld werkt nu en gaf drie series terug.
- Eén recente disk-I/O-range-query werkte: 56 punten/serie, 65 s interval, geen gaten, nulls of duplicaten.
- Een beperkte `count(up)`-retentieprobe toont in deze datasource historie vanaf circa 2026-09-17 13:00 UTC; eerdere punten ontbreken.
- De genoemde juli-, augustus- en 9-septemberincidenten vallen daarbuiten, maar een recenter incident of externe historische export kan wel bruikbaar zijn.
- CloudSentinel bevraagt nu een directe Prometheus `query_range`-API; Grafana POST, bearer-auth en dataframeframes worden niet ondersteund.
- Offline catalogusverwerking op de eigen installatie is waarschijnlijk mogelijk met een kleine Grafana-exportadapter; training vereist nu wel rijlabels.
- Portability met recente telemetry werkt qua toegang/export; bekende-incidentanalyse is nog niet aangetoond en hangt af van een recenter incident of export.
- Ja: mail Gabriel vandaag over omgeving, tijdzone/duur/scope, `scruffy`, retentie/normaalinterval en toestemming voor beperkte datahandling.

## 2. Uitgevoerde tests

### Feiten

- Token uitsluitend gelezen uit `/tmp/cloudsentinel-naavre-token` (mode `0600`); token niet weergegeven, gelogd of in Git opgeslagen.
- Eerste netwerkpoging in de sandbox: DNS-fout `[Errno 8] nodename nor servname provided, or not known`. Dezelfde verzoeken buiten de netwerksandbox werkten met normale TLS-validatie.
- `GET /grafana/api/datasources`: HTTP 200, JSON-lijst met 3 datasources. Relevant: naam `Prometheus`, UID `prometheus`, type `prometheus`, default `true`, read-only `true`, HTTP-methode `POST`, minimale intervalhint `65s`.
- `POST /grafana/api/ds/query` met Gabriels gecorrigeerde expressie `up{node="k8s-staging-1"}`: HTTP 200; `results.A.status=200`; drie Grafana dataframeframes, elk één tijd- en één numeriek veld. De node bestaat dus momenteel. Instance-adressen zijn hier niet gerapporteerd.
- `GET /grafana/api/search?type=dash-db&limit=50`: HTTP 200; 35 dashboardmetadatarecords, waaronder Argo Workflows, CoreDNS, Jupyter Hub, Kubernetes compute/network/PV, ingress, MinIO, Node Exporter en Prometheus.
- Read-only dashboarddefinities voor Node Exporter USE, Argo Workflows, Jupyter Hub en MinIO: HTTP 200. De metricnamen in §3 komen uitsluitend uit deze definities of uit uitgevoerde queries.
- Recente selectiequery: `topk(3, instance_device:node_disk_io_time_seconds:rate5m{job="node-exporter"})`; HTTP 200, drie actuele series. Geselecteerd: `instance="k8s-staging-1", device="sda"`.
- Recente range-query via datasource `prometheus`:
  - PromQL: `instance_device:node_disk_io_time_seconds:rate5m{job="node-exporter",instance="k8s-staging-1",device="sda"}`
  - aangevraagd: `2026-10-19T08:59:32Z` t/m `2026-10-19T09:59:32Z`; `intervalMs=65000`, `maxDataPoints=60`;
  - Grafana rapporteerde `Step: 1m5s`; ontvangen timestamps `2026-10-19T08:59:10Z` t/m `09:58:45Z`;
  - 56 unieke punten, alle intervallen 65 s, 0 nulls, 0 niet-eindige waarden, 0 duplicaten en geen intervalgaten;
  - labels koppelen de serie aan Node Exporter, namespace `monitoring`, node `k8s-staging-1`, device `sda`;
  - waarden 0,2412–0,3946. Het dashboard noemt dit “Disk IO Utilisation”; de recording-rule-naam duidt op een vijfminuten-rate van disk-I/O-tijd (afgeleid resultaat, geen ruwe counter). Exacte recording-ruledefinitie is niet geïnspecteerd. De 65 s query-step is expliciete resolutie/downsampling; de bron kan fijner zijn.
- Voorlopige incident-retentiecheck, omdat de incidenttijdzone onbekend is: `2026-09-09T07:00:00Z` t/m `13:00:00Z`, step 5 min, één geselecteerde diskserie. HTTP 200 en een leeg dataframeframe (`fields=[]`, `values=[]`).
- Controle met Gabriels selector `up{node="k8s-staging-1"}` over hetzelfde venster: HTTP 200 en eveneens een leeg frame.
- Beperkte retentieprobe: `count(up)` van `2026-07-26T00:00:00Z` t/m `2026-10-19T10:00:00Z`, één punt per dag (maximaal 86). HTTP 200; 32 punten, uitsluitend 18 september t/m 19 oktober, steeds 62 targets. Er waren geen punten in juli, augustus of vóór 18 september.
- Verfijning rond de grens: `count(up)` van 17 t/m 19 september, één punt per uur. Het eerste zichtbare punt was `2026-09-17T13:00:00Z`; vervolgens 36 aaneengesloten uurpunten t/m 19 september 00:00 UTC. Dit is een waargenomen beschikbaarheidsgrens, niet noodzakelijk de formeel geconfigureerde retentie.
- Exacte augustuscontrole voor het incident van 24 augustus 09:10 (tijdzone nog onbekend): voorlopig UTC-venster `2026-08-24T05:30:00Z` t/m `11:30:00Z`, step 15 min. Zowel `count(up)` als `count(instance_device:node_disk_io_time_seconds:rate5m)` gaf HTTP 200 met lege frames. De query was labelonafhankelijk en telde alle destijds aanwezige series van beide metricfamilies.
- Lokale codeverificatie in de git-genegeerde `.venv-analysis`: 69 geselecteerde catalogus-, fetch-, lifecycle-, trainingsintegratie-, live-monitoring- en comparisontests; **69 geslaagd**. De test gebruikte alleen tijdelijke lokale data en een loopback fake-Prometheus.
- Geen EC2-, Kubernetes- of NaaVRE-configuratie gewijzigd; geen modeltraining, deployment, fault injection, gebruikers-/datasource-/dashboardwijziging of brede metricexport uitgevoerd.
- Kleine ruwe API-responses staan alleen onder `/tmp/cloudsentinel-naavre/` en zijn niet voor publicatie bestemd.

#### Diagnose van de lege historische resultaten

| Mogelijke verklaring | Bevinding | Conclusie |
|---|---|---|
| Ontbrekende historie in de huidige datasource | `count(up)` zonder labelselectors is leeg vóór circa 17 september 13:00 UTC en daarna aaneengesloten gevuld | **Sterk ondersteund** als verklaring voor juli/augustus/9 september; formele retentieconfiguratie nog onbevestigd |
| Huidige serie of label bestond toen niet | Augustus is ook leeg voor labelonafhankelijk `count(up)` en voor de telling van alle disk-I/O-recording-rule-series | **Niet voldoende als hoofdverklaring**; het bestaan van de specifieke disk-recording rule vóór de grens blijft afzonderlijk onbekend |
| Query- of accountrechtenfout | Dezelfde endpoint, token en queryfamilies geven HTTP 200 en recente data; historische requests geven HTTP 200 met lege frames, niet 401/403/400 | **Niet waargenomen**; een backend/datasourcewissel of andere historische bron blijft onbekend |
| Andere Grafana-datasource | Het account ziet drie datasources maar slechts één van type Prometheus (`uid=prometheus`) | **Geen alternatieve Prometheusbron zichtbaar**; een niet-zichtbaar archief/export kan nog bestaan |

### Aannames/interpretaties

- `staging` in URL en labels is geen afdoende bewijs voor de werkelijk gemonitorde omgeving; in dit rapport heet dit een externe operationele omgeving.
- Het zesuurs UTC-venster is alleen een voorlopige offset-afdekking rond 10:37:16; er wordt geen exacte incidentuitlijning geclaimd.
- De dagelijkse en uurlijkse `count(up)`-probes zijn labelonafhankelijk en tonen een scherpe beschikbaarheidsgrens rond 17 september 13:00 UTC. Daardoor is “huidige node/serielabel bestond toen niet” geen afdoende verklaring voor de ontbrekende juli-/augustusdata. Niet uitgesloten zijn een wijziging van datasource/backend, een oudere omgeving waarin zelfs `up` niet in deze bron stond, of een extern archief.
- Voor de disk-I/O-recording rule afzonderlijk blijft onbekend of zij vóór de retentiegrens al bestond; de gelijktijdig lege `count(up)` maakt dat onderscheid voor augustus niet relevant voor bruikbaarheid via deze datasource.
- Incidentbeschrijvingen zijn gerapporteerde context en geen geverifieerde oorzaken of puntlabels.

### Open vragen

- Welke omgeving monitort datasource `prometheus` werkelijk: productie, staging of een combinatie?
- Welke tijdzone, duur en getroffen workflows/nodes/namespaces horen bij de incidenten?
- Wat is `scruffy` precies en met welke Prometheuslabels (node, instance, job, MinIO-server) correspondeert het?
- Bevestigt Gabriel dat de waargenomen grens rond 17 september de bedoelde retentie is, en bestaat oudere data in een andere datasource of read-only export/archief?
- Welk behouden interval is representatief voor normale belasting?
- Is beperkte export naar de eigen CloudSentinel-installatie, tijdelijke lokale opslag en geanonimiseerde thesisrapportage toegestaan, en onder welke bewaartermijn/voorwaarden?

## 3. Vastgestelde metrics en beperkingen

| Metric/query | Component en semantiek | Beschikbaarheid/historie | Belangrijkste beperking |
|---|---|---|---|
| `up{node="k8s-staging-1"}` / `count(up)` | Prometheus target-availability; actuele frames betroffen kubelet endpoints. `count(up)` is labelonafhankelijk | Actueel: 3 node-series met waarde 1. Retentieprobe: data zichtbaar vanaf circa 2026-09-17 13:00 UTC; 9 september en 24 augustus leeg | Geeft bereikbaarheid van scrape targets, niet workflow-I/O of oorzaak; oudere bron/archief onbekend |
| `instance_device:node_disk_io_time_seconds:rate5m` | Node Exporter recording rule; dashboard: disk-I/O-utilisatie; vijfminuten-rate, eenheid effectief seconden I/O-tijd per seconde | Actueel getest op `k8s-staging-1/sda`: 56 punten. 9 september leeg; labelonafhankelijke telling op 24 augustus leeg | Node/device is bekend, maar geen workflow-, image-pull- of volume-attributie; bestaan van deze recording rule vóór retentiegrens onbekend |
| `instance:node_network_receive_bytes_excluding_lo:rate5m`, `...transmit...` | Node Exporter netwerkdoorvoer, vijfminuten-rate | Vastgesteld in read-only dashboarddefinitie; data niet bevraagd | Kan image-downloadtraffic ondersteunen, maar niet aan pulls toeschrijven zonder container/runtime/Kubernetescontext |
| `instance_device:node_disk_io_time_weighted_seconds:rate5m` | Node Exporter disk-I/O-saturatie | Vastgesteld in dashboarddefinitie; data niet bevraagd | Vereist node/device-mapping en behouden historie |
| `argo_workflows_count`, `argo_workflows_operation_duration_seconds_bucket`, queue depth/latency-metrics | Argo workflowstatus, operation duration en wachtrijen; dashboard gebruikt gauges en `rate`/`delta`/histogramquantielen | Vastgesteld in Argo-dashboarddefinitie; data niet bevraagd | Geen bewezen koppeling met de genoemde workflows of image downloads |
| `jupyterhub_server_spawn_duration_seconds_{bucket,sum,count}` | Jupyter Hub spawnduur en status; dashboard gebruikt `rate`, `increase` en histogramquantielen; seconden | Vastgesteld in Jupyter-dashboarddefinitie; niet afzonderlijk bevraagd. De algemene datasourcehistorie begint na alle genoemde Jupyter-incidenten | Relevant voor een recenter spawnincident of externe historische export; instance/job en incidentduur ontbreken |
| `minio_node_io_rchar_bytes`, `minio_node_io_wchar_bytes`, `minio_s3_traffic_{received,sent}_bytes`, `minio_s3_requests_total` | MinIO proces-I/O, S3-verkeer en requests | Vastgesteld in MinIO-dashboarddefinitie; data niet bevraagd | Niet vastgesteld welke MinIO/server/volume bij `scruffy` of getroffen Jupyter/Argo-workloads hoort |

Voor image-downloadslowdowns zijn node-diskutilisatie/-saturatie, netwerkdoorvoer en Kubernetes pod/containercontext plausibel. Voor Jupyter-spawnproblemen zijn de spawnhistogrammen directer. Voor `scruffy`-saturatie zijn node- en MinIO-I/O/trafficmetrics kandidaten. Zonder container-runtime/image-pullmetrics, workflow/pod/node-mapping, volume/PVC-mapping en incidentduur is betrouwbare componentattributie niet mogelijk.

## 4. Huidige CloudSentinel-contracten en kleinste vervolgstap

### Baseline

- `data_ingestion/data_collector.py::query_prometheus` (regels 83–107) doet een directe unauthenticated `GET {url}/api/v1/query_range`. `fetch_metrics` (38–80) neemt alleen de eerste serie per query, bouwt lokale naïeve timestamps, vult gaten met forward-fill, backward-fill en daarna nul, en verwijdert de timestampkolom.
- `data_processing/cgnn_preprocess.py::load_data/get_data/normalize_data` leest headerloze numerieke CSV, verwacht train/test en doorgaans binaire labels, en normaliseert met `MinMaxScaler` (fit op train bij training).
- `data_processing/crca_preprocess.py::normalize_data` accepteert CSV/TXT/PKL en fit een eigen `MinMaxScaler` op het aangeboden bestand. CRCA verwerkt een lokale matrix, maar de collector blijft directe Prometheus verwachten.

### Verbeterde workflow

- `data_ingestion/catalogue/models.py::validate_source` (32–73) accepteert alleen een server-side `prometheus_source_id`; URL/token in datasetpayloads is verboden. ISO-8601-timestamps moeten een zone bevatten en worden UTC (`validation.py::validate_timestamp`, 69–79).
- `data_ingestion/catalogue/fetching.py::PrometheusRangeClient.query_range` (209–270) is nog steeds specifiek voor directe Prometheus GET en verwacht een Prometheus matrixresultaat. Er is geen bearer-header- of Grafana-dataframepad.
- `build_query_executions` (121–200) begrenst queryaantal, window, step en samples; `assemble_time_series` (307–421) maakt een inclusieve UTC-grid, bewaart stabiele featurevolgorde/serieslabels, laat off-grid samples met waarschuwing vallen en telt ontbrekende waarden.
- Canonieke output is `observations.csv` (`timestamp` + benoemde features), `feature_matrix.csv`, `timestamps.csv`, schema/provenance/checksums en een immutable live-inputrecipe (`fetching.py`, 666–709; `live_input.py::build_live_input_recipe`, 32–80). Catalogusnormalisatie en imputatie zijn `none`.
- Partities zijn predefined of niet-overlappende tijdvakken met optionele gap (`validation.py::validate_partition`, 168–222). Trainingbundles zijn checksummed ZIPs met headerloze `train.csv`, `test.csv`, `labels.csv` en manifest; huidige trainingmaterialisatie vereist binaire rijlabels die de testset volledig dekken (`materialization.py`, 168–321).
- Evaluatiebundles kunnen zonder labels worden gemaakt en bevatten bij time-range-data timestamps (`materialization.py::materialize_evaluation_bundle`, 401–575). Dan kan wel inference/comparison plaatsvinden met een reeds bestaand, feature-compatibel model, maar geen gelabelde accuracy-score.
- De huidige CGNN-adapter vereist labels, meer observaties dan `lookback`, zet NaN naar nul en fit `MinMaxScaler` alleen op train (`detectors/cgnn/adapter.py`, 69–132). De Isolation Forest-adapter vereist in de huidige trainingsflow eveneens labels voor evaluatie, eindige waarden en gebruikt een op train gefitte, persistente `StandardScaler`-pipeline (`detectors/isolation_forest/adapter.py`, 62–90; `implementation.py`, 47–96 en 181–219).
- Live monitoring herhaalt dezelfde directe Prometheus-client, vereist exact dezelfde featurevolgorde en serieslabels en weigert een window met ontbrekende waarden (`data_ingestion/live_monitoring.py::collect_recipe_window`, 73–133).

### Oordeel A–E

| Niveau | Oordeel | Onderbouwing |
|---|---|---|
| A. Toegang en kleine telemetry-export | **getest en werkt** | Auth, datasource, dashboards, instant-query en één recente eenuursrange zijn echt uitgevoerd met HTTP 200 en bruikbare data. |
| B. Datasetimport en workflow op eigen installatie | **waarschijnlijk mogelijk met genoemde aanpassing** | Geen NaaVRE-deployment nodig; de verbeterde catalogus kan canonieke metricmatrices verwerken. Nodig: Grafana-adapter plus expliciete feature/partitionmetadata. Training blijft zonder rijlabels geblokkeerd; labelvrije evaluatie vereist een compatibel bestaand model. |
| C. Exploratieve analyse rond een bekend incident | **onbekend** | Geen van de genoemde incidenten valt binnen de zichtbaar beschikbare historie van deze datasource. Dit blokkeert die specifieke vensters, niet incidentanalyse in het algemeen: een recenter bekend incident of historische export kan de case alsnog mogelijk maken. Tijdzone, duur en componentmapping blijven nodig. |
| D. Live remote monitoring via Grafana | **waarschijnlijk mogelijk met genoemde aanpassing** | De live recipeflow bestaat, maar ondersteunt alleen directe Prometheus GET. Een Grafana POST/dataframe/auth-adapter is vereist; directe Prometheus-toegang is niet technisch noodzakelijk als deze adapter wordt toegevoegd. |
| E. Deployment in NaaVRE/gecontroleerde experimenten | **onbekend** | Buiten scope en niet aangetoond; huidige read-only toegang bewijst geen deployment- of fault-injectionmogelijkheid. |

Kleinste vervolgstap na Gabriels antwoord: implementeer **niet nu**, maar specificeer één server-side `GrafanaRangeClient` naast `PrometheusRangeClient`. Deze moet POST `/grafana/api/ds/query`, token uit een niet-geversioneerde serversecret lezen, expliciete epoch-ms `from/to`, `range=true`, `intervalMs` en datasource UID sturen, Grafanaframes begrensd omzetten naar het bestaande `{metric, values}`-contract en errors onderscheiden. Daarna kunnen `execute_catalogue_fetch` en `collect_recipe_window` de bestaande assembler, feature-identiteit, checksums en missing-dataregels hergebruiken. Token/URL horen niet in datasetmetadata. Een alternatieve eenmalige bestandsimport zou een nieuwe ondersteunde canonical-importroute vereisen; die bestaat nu niet als publiek pad.

## 5. Conceptmail aan Gabriel (English)

Subject: NaaVRE telemetry portability check — remaining metadata and retention questions

Hi Gabriel,

Thank you — the read-only Grafana service account works. I confirmed the `prometheus` datasource and successfully ran your `up{node="k8s-staging-1"}` example. I also retrieved one recent, small Node Exporter disk-I/O range (one series for one hour) through Grafana.

Before I proceed with a limited CloudSentinel portability case, could you please clarify the remaining points?

1. Does the `prometheus` datasource monitor production, staging, or a combination? The URL and current labels contain “staging”, but I do not want to infer the monitored environment from naming alone.
2. What timezone applies to the incident timestamps, approximately how long did each symptom last, and which workflows/components/nodes/namespaces were affected?
3. What exactly is “scruffy” (node, host, MinIO server, storage system, or something else), and which Prometheus labels identify it?
4. Could you confirm the retention or archive situation? A low-cardinality `count(up)` query, independent of node labels, returned data only from approximately 2026-09-17 13:00 UTC onward. A six-hour check around the 24 August incident returned HTTP 200 but empty results for both `count(up)` and the count of all available Node Exporter disk-I/O recording-rule series. The listed July/August/9 September incidents therefore appear to be outside the history currently visible through this datasource, rather than merely using a node label that has since changed.
5. If those incidents are not available here, could we either use a more recent known incident within the retained period, or receive a small historical export from another existing source? A CSV/JSON export with timestamps, series labels, query/metric semantics and sampling interval would be sufficient; no additional cluster privileges are needed.
6. Could you identify one retained interval representative of normal operation for comparison?
7. May I export a small, scoped set of telemetry through the existing read-only Grafana API to my own CloudSentinel installation, store it temporarily outside version control, and report only aggregated/anonymised results in my thesis? Please let me know any scope, retention, deletion, or attribution requirements.

I am not requesting Kubernetes/admin access or direct Prometheus access at this stage; the existing Grafana route appears sufficient if the needed history is available.

Best regards,
Sue

## 6. Grens van de bewijsvoering

Deze case kan aantonen dat een externe operationele telemetrybron via read-only Grafana bereikbaar is, dat beperkte series met labels/timestamps exporteerbaar zijn, en hoeveel adapter-, featuremapping- en datacuratie nodig is om ze op de eigen CloudSentinel-installatie te gebruiken. Zij kan na een kleine adapter ook portability van de ingestie-/preprocessingketen onderzoeken.

Deze check toont **niet** aan dat CloudSentinel in NaaVRE draait, dat alle relevante historie bestaat, dat incidenttijden puntlabels zijn, dat gemelde oorzaken zijn geverifieerd, dat modeltraining verantwoord is, of dat detectie/RCA accuraat is. Er zijn geen gecontroleerde faults, geen causale attributie en geen NaaVRE-deployment uitgevoerd. Eén geslaagde recente metric bewijst evenmin multivariate featuredekking, modelcompatibiliteit of live betrouwbaarheid.
