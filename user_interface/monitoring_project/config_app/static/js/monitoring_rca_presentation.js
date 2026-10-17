(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.CloudSentinelRcaPresentation = api;
})(typeof window !== "undefined" ? window : globalThis, function () {
  "use strict";

  const CAUSAL = "cloudsentinel.causal-rca/legacy-v1";
  const BARO = "cloudsentinel.baro-robust-scorer/v1";
  const METHODS = {
    [CAUSAL]: "CausalRCA",
    [BARO]: "BARO RobustScorer",
  };

  function methodId(run, result) {
    const explicit = (result && result.method) || (run && run.method);
    if (typeof explicit === "string" && explicit) return explicit;
    const algorithm = run && run.algorithm;
    if (algorithm && typeof algorithm.id === "string" && algorithm.id) return algorithm.id;
    return CAUSAL;
  }

  function methodName(id) {
    return METHODS[id] || "RCA method";
  }

  function ranking(result) {
    return result && Array.isArray(result.ranking)
      ? result.ranking.filter((item) => item && typeof item === "object")
      : [];
  }

  function entity(item) {
    const structured = item && item.entity;
    if (structured && structured.id !== undefined && structured.id !== null && String(structured.id)) {
      return {
        available: true,
        type: String(structured.type || item.entity_type || "entity"),
        id: String(structured.id),
        label: `${String(structured.type || item.entity_type || "entity")}: ${String(structured.id)}`,
      };
    }
    if (item && item.entity_id !== undefined && item.entity_id !== null && String(item.entity_id)) {
      return {
        available: true,
        type: String(item.entity_type || "entity"),
        id: String(item.entity_id),
        label: `${String(item.entity_type || "entity")}: ${String(item.entity_id)}`,
      };
    }
    return {available: false, type: null, id: null, label: "Unavailable — no entity metadata"};
  }

  function humanMetric(value) {
    if (value === undefined || value === null || String(value) === "") return "Unavailable";
    const raw = String(value);
    const known = {
      cpu_usage: "CPU usage",
      mem_usage: "Memory usage",
      memory_usage: "Memory usage",
      cpu_throttling: "CPU throttling",
      pod_restarts: "Pod restarts",
    };
    if (known[raw]) return known[raw];
    const spaced = raw.replaceAll("_", " ").replaceAll("-", " ").replace(/\s+/g, " ").trim();
    return spaced ? spaced.charAt(0).toUpperCase() + spaced.slice(1) : raw;
  }

  function score(item) {
    if (!item || item.score === null || item.score === undefined || item.score === "") return null;
    const value = Number(item && item.score);
    return Number.isFinite(value) ? value : null;
  }

  function relativeScore(item, items) {
    const value = score(item);
    if (value === null) return 0;
    const maximum = Math.max(0, ...ranking({ranking: items}).map((entry) => {
      const candidate = score(entry);
      return candidate === null ? 0 : Math.abs(candidate);
    }));
    return maximum > 0 ? Math.min(100, Math.abs(value) / maximum * 100) : 0;
  }

  function summary(payload) {
    const run = (payload && payload.run) || {};
    const result = (payload && payload.result) || {};
    const rows = ranking(result);
    const top = rows[0] || null;
    return {
      method_id: methodId(run, result),
      method_name: methodName(methodId(run, result)),
      analysis_window: run.analysis_window || null,
      strongest_entity: top ? entity(top) : {available:false, label:"Unavailable"},
      strongest_metric: top ? humanMetric(top.metric_name || top.feature_name || top.feature_id) : "Unavailable",
      top_suspects: rows.slice(0, 3),
      observation_count: Number.isInteger(run.observation_count) ? run.observation_count : null,
      feature_count: Number.isInteger(run.feature_count) ? run.feature_count : null,
      change_point: result.change_point_evidence || run.change_point_evidence || null,
    };
  }

  return {
    CAUSAL,
    BARO,
    methodId,
    methodName,
    ranking,
    entity,
    humanMetric,
    score,
    relativeScore,
    summary,
  };
});
