"use strict";

load("user_interface/monitoring_project/config_app/static/js/monitoring_rca_presentation.js");

const presentation = globalThis.CloudSentinelRcaPresentation;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const historical = presentation.summary({run: {}, result: {ranking: []}});
assert(historical.method_id === presentation.CAUSAL, "historical results must default to CausalRCA");
assert(historical.top_suspects.length === 0, "an empty ranking must remain empty");

const causal = presentation.summary({
  run: {
    method: presentation.CAUSAL,
    analysis_window: {start: "2026-10-14T10:00:00Z", end: "2026-10-14T10:10:00Z"},
    observation_count: 40,
    feature_count: 2,
  },
  result: {ranking: [
    {rank: 1, feature_id: "cpu_usage|pod=checkout", metric_name: "cpu_usage", entity: {type: "pod", id: "checkout"}, score: 8},
    {rank: 2, feature_id: "mem_usage|pod=checkout", metric_name: "mem_usage", entity: null, score: 4},
  ]},
});
assert(causal.method_name === "CausalRCA", "CausalRCA summary method is wrong");
assert(causal.strongest_entity.label === "pod: checkout", "entity summary is wrong");
assert(causal.strongest_metric === "CPU usage", "human metric label is wrong");
assert(presentation.relativeScore(causal.top_suspects[1], causal.top_suspects) === 50, "relative score is wrong");
assert(presentation.entity(causal.top_suspects[1]).available === false, "missing entity must remain unavailable");

const changePoint = {
  source: "stored_monitoring_cycle_result",
  detected_change_point: "2026-10-14T10:05:00Z",
  index: 20,
  baseline_observation_count: 20,
  anomalous_observation_count: 20,
};
const baro = presentation.summary({
  run: {method: presentation.BARO, observation_count: 40, feature_count: 2},
  result: {
    method: presentation.BARO,
    change_point_evidence: changePoint,
    ranking: [{rank: 1, feature_id: "latency|pod=api", metric_name: "latency", score: 2.5}],
  },
});
assert(baro.method_name === "BARO RobustScorer", "BARO summary method is wrong");
assert(baro.change_point === changePoint, "BARO change-point evidence is not preserved");

const strange = '<img src=x onerror="alert(1)">';
assert(presentation.humanMetric(strange) === strange, "strange metric names must remain plain text values");
assert(presentation.entity({entity: {type: strange, id: strange}}).label.includes(strange), "strange entity labels must remain data");

print("monitoring RCA presentation tests passed");
