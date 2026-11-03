# Overview of all evaluations
During my thesis, we performed a total of 7 different evaluations over the final implementation. This evidence folder is meant to make it clear how the evidence was produced. Each evaluation section has a corresponding evidence folder. Each evidence folder contains outputs of commands, shellscripts that were used in the evaluation, yaml configurations and info about the settings and protocol that was used in the evaluation. Note that in some places, generative AI was used to help explain the explanations and outputs.

## 1. Dataset management, training, comparison, and detector integration
Contains the shared test setup for evaluations 1–4, automated test results, and a repeat check of the workflow's main functions. It also includes an independent check of the calculated scores, saved files from a workflow run, training and comparison records, anomaly labels, and checks showing whether results can be traced back to their data, models, and settings. Copies of the detector adapters—the code connecting each detector to CloudSentinel—are included.

## 2. Performance and resource use
Contains measurements of how long the selected model comparisons took and how much CPU and memory they used.

## 3. Application scaling and changing pod identities
Contains records of the Kubernetes setup and health, checks of whether the frontend responded, Prometheus query results, and data collected through the Data Catalogue. It also documents the attempt to start a second worker node and why that attempt failed.

## 4. Known-fault tests in Online Boutique
Contains the start and end times of deliberately introduced faults, checks using Prometheus and Kubernetes to confirm what happened, Live Monitoring results, detector responses, and root-cause analysis (RCA) results.

## 5. RCAEval Train Ticket benchmark
Contains the fixed test procedure, selected test cases, the common set of metrics used across cases, detector results, and BARO service rankings. It also includes file hashes, which identify the source files and can be used to check whether they have changed.

## 6. NaaVRE external environment
Contains the procedure and scripts used to collect monitoring data with read-only access, together with the retained summary results. Login details and raw operational monitoring data are not included.

## 7. User evaluation
Contains the procedure used to guide participants through the application, anonymised participant notes, and a summary of common themes across their feedback. Audio recordings, full transcripts, and information identifying participants are not included.

The evidence collection does not include raw trained-model files, private server paths, login details, or protected operational monitoring data.