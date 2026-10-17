(function () {
  "use strict";

  const root = document.getElementById("monitoring-history");
  if (!root) return;
  const rcaPresentation = window.CloudSentinelRcaPresentation;
  const listNode = document.getElementById("monitoring-sessions");
  const detailNode = document.getElementById("monitoring-detail");
  const errorNode = document.getElementById("monitoring-error");
  const pageNode = document.getElementById("sessions-page");
  const newer = document.getElementById("sessions-newer");
  const older = document.getElementById("sessions-older");
  const state = {
    page: 1,
    pages: 0,
    selected: new URL(window.location.href).searchParams.get("session"),
    cyclePage: 1,
    rcaPage: 1,
    chart: null,
    timer: null,
    selectedRca: new URL(window.location.href).searchParams.get("rca"),
    sessionsRenderKey: null,
    detailRenderKey: null,
    rcaResultCache: new Map(),
    shouldPoll: false,
  };

  function node(tag, text, className) {
    const element = document.createElement(tag);
    if (text !== undefined && text !== null) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  }

  function badge(status) {
    const styles = {active:"success", warmup:"warning", collecting:"info", queued:"secondary", running:"info", completed:"success", skipped:"warning", stopped:"secondary", failed:"danger", error:"danger", interrupted:"warning"};
    return node("span", status || "Unavailable", "badge bg-" + (styles[status] || "secondary"));
  }

  function time(value) {
    return window.CloudSentinel.formatDateTime(value, "Unavailable");
  }

  function rcaDecision(event) {
    const labels = {
      started: "First RCA analysis started",
      reanalysis_started: "RCA reanalysis started",
      suppressed_episode: "RCA not repeated: this episode was already analysed",
      suppressed_interval: "RCA reanalysis interval has not elapsed",
      suppressed_new_observations: "Not enough new observations for RCA reanalysis",
      suppressed_identical_input: "RCA reanalysis input is unchanged",
      suppressed_running: "Another RCA run is active",
      suppressed_concurrent: "Another RCA intent was reserved first",
      suppressed_budget: "RCA episode budget reached",
      suppressed_cooldown: "Episode cooldown is active",
      skipped: "RCA preflight skipped; a later high cycle can try again",
      recovered: "Episode recovered",
      below_or_equal: "Threshold not exceeded",
      no_new_data: "No newly evaluated observations",
    };
    return labels[event.status] || String(event.status || "Unavailable").replaceAll("_", " ");
  }

  function select(sessionId) {
    if (state.selected !== sessionId) {
      state.sessionsRenderKey = null;
      state.detailRenderKey = null;
    }
    state.selected = sessionId;
    const url = new URL(window.location.href);
    if (sessionId) url.searchParams.set("session", sessionId); else url.searchParams.delete("session");
    window.history.replaceState({}, "", url);
  }

  function selectRca(runId) {
    state.selectedRca = runId;
    const url = new URL(window.location.href);
    if (runId) url.searchParams.set("rca", runId); else url.searchParams.delete("rca");
    window.history.replaceState({}, "", url);
  }

  function showError(message) { errorNode.textContent = message; errorNode.classList.remove("d-none"); }
  function clearError() { errorNode.textContent = ""; errorNode.classList.add("d-none"); }

  function renderKey(value) {
    return JSON.stringify(value);
  }

  async function json(url, options) {
    const response = await fetch(url, Object.assign({cache:"no-store", headers:{Accept:"application/json"}}, options || {}));
    const body = await response.text();
    let payload;
    try { payload = JSON.parse(body); } catch (_error) { throw new Error("Monitoring History returned an invalid response."); }
    if (!response.ok) throw new Error(payload.message || payload.error || "Monitoring History is temporarily unavailable.");
    return payload;
  }

  function renderSessions(payload) {
    const items = payload.items || [];
    if (!state.selected && items.length) select(items[0].session_id);
    const key = renderKey({selected:state.selected, payload});
    if (state.sessionsRenderKey === key) return false;
    state.sessionsRenderKey = key;
    listNode.replaceChildren();
    if (!items.length) {
      listNode.append(node("div", "No monitoring sessions found.", "p-3 text-muted"));
      detailNode.replaceChildren(node("div", "Start a monitoring session to create durable history.", "alert alert-secondary"));
      select(null);
    }
    items.forEach((session) => {
      const button = node("button", null, "list-group-item list-group-item-action session-button");
      button.type = "button";
      if (session.session_id === state.selected) button.classList.add("active");
      const top = node("div", null, "d-flex justify-content-between gap-2");
      top.append(node("strong", session.display_name || session.model_id || "Monitoring session"));
      top.append(badge(session.status));
      button.append(top);
      button.append(node("small", (session.detector_id || "Saved model") + " · " + time(session.started_at), "d-block text-muted mt-1"));
      button.addEventListener("click", () => { state.cyclePage=1; state.rcaPage=1; selectRca(null); select(session.session_id); loadAll(false); });
      listNode.append(button);
    });
    state.pages = payload.pages || 0;
    pageNode.textContent = state.pages ? `Page ${payload.page} of ${state.pages}` : "";
    newer.disabled = state.page <= 1;
    older.disabled = !state.pages || state.page >= state.pages;
    return true;
  }

  function metric(label, value) {
    const wrapper = node("div", null, "col-sm-6 col-xl-3");
    const card = node("div", null, "border rounded p-3 h-100");
    card.append(node("div", label, "small text-muted"));
    card.append(node("strong", value));
    wrapper.append(card);
    return wrapper;
  }

  function chartData(cycles, rca) {
    const ordered = cycles.slice().reverse();
    const threshold = rca && rca.enabled ? Number(rca.threshold_percentage) : null;
    const datasets = [{
      label: "Anomalous newly evaluated observations (%)",
      data: ordered.map((cycle) => cycle.anomaly_percentage === null || cycle.anomaly_percentage === undefined ? null : cycle.anomaly_percentage),
      borderColor: "#0d6efd", backgroundColor: "rgba(13,110,253,.12)",
      pointRadius: ordered.map((cycle) => cycle.rca_trigger && ["started","reanalysis_started","skipped"].includes(cycle.rca_trigger.status) ? 7 : 4),
      pointBackgroundColor: ordered.map((cycle) => {
        const status = cycle.rca_trigger && cycle.rca_trigger.status;
        if (["started","reanalysis_started"].includes(status)) return "#dc3545";
        if (status === "skipped") return "#fd7e14";
        if (status && status.startsWith("suppressed_")) return "#ffc107";
        return "#0d6efd";
      }),
      tension: .2, spanGaps: false,
    }];
    if (Number.isFinite(threshold)) datasets.push({
      label: "Automatic RCA threshold (%)", data: ordered.map(() => threshold),
      borderColor: "#dc3545", borderDash: [7, 5], pointRadius: 0, tension: 0,
    });
    return {
      labels: ordered.map((cycle) => time((cycle.prediction_window || {}).end || (cycle.query_window || {}).end)),
      datasets,
    };
  }

  function renderChart(cycles, rca) {
    const canvas = document.getElementById("monitoring-cycle-chart");
    if (!canvas || typeof Chart === "undefined") return;
    const data = chartData(cycles, rca);
    if (state.chart && state.chart.canvas === canvas) {
      state.chart.data.labels = data.labels;
      state.chart.data.datasets = data.datasets;
      state.chart.update("none");
      return;
    }
    if (state.chart) state.chart.destroy();
    state.chart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data,
      options: {responsive:true, maintainAspectRatio:false, scales:{y:{min:0,max:100,title:{display:true,text:"Anomaly percentage (%)"}},x:{title:{display:true,text:"Latest newly evaluated prediction time"}}}},
    });
  }

  function summaryValue(label, value) {
    const wrapper = node("div", null, "col-sm-6 col-xl-4");
    const card = node("div", null, "border rounded p-3 h-100");
    card.append(node("div", label, "small text-muted"));
    card.append(node("strong", value));
    wrapper.append(card);
    return wrapper;
  }

  function rankingScore(item) {
    const value = rcaPresentation.score(item);
    return value === null ? "Unavailable" : value.toFixed(6);
  }

  function renderSuspects(rows, target) {
    const section = node("section", null, "mb-3");
    section.append(node("h4", "Top suspects", "h6"));
    section.append(node("p", "Bars show relative score magnitude within this run only. Scores are not probabilities and must not be compared between RCA methods.", "small text-muted"));
    const list = node("ol", null, "list-group list-group-numbered");
    rows.forEach((item) => {
      const entry = node("li", null, "list-group-item");
      const identity = rcaPresentation.entity(item);
      const top = node("div", null, "d-flex justify-content-between gap-3 flex-wrap");
      const labels = node("div");
      labels.append(node("strong", rcaPresentation.humanMetric(item.metric_name || item.feature_name || item.feature_id)));
      labels.append(node("div", identity.available ? identity.label : "Entity unavailable", "small text-muted"));
      top.append(labels);
      top.append(node("span", rankingScore(item), "font-monospace small"));
      entry.append(top);
      const track = node("div", null, "rca-score-track mt-2");
      const bar = node("div", null, "rca-score-bar");
      const percentage = rcaPresentation.relativeScore(item, rows);
      bar.style.width = `${percentage.toFixed(2)}%`;
      bar.setAttribute("role", "img");
      bar.setAttribute("aria-label", `Relative score magnitude ${percentage.toFixed(1)} percent within this run`);
      track.append(bar); entry.append(track); list.append(entry);
    });
    section.append(list); target.append(section);
  }

  function renderBaroEvidence(summary, target) {
    const evidence = summary.change_point || {};
    const card = node("section", null, "alert alert-light border mb-3");
    card.append(node("h4", "BARO change evidence", "h6"));
    if (!evidence.detected_change_point) {
      card.append(node("p", "Change-point evidence is unavailable for this stored result.", "mb-0 text-muted"));
      target.append(card); return;
    }
    const window = summary.analysis_window || {};
    const list = node("dl", null, "row mb-2");
    [
      ["Detector change point", time(evidence.detected_change_point)],
      ["Evidence source", evidence.source === "stored_monitoring_cycle_result" ? "Stored monitoring-cycle detector result" : "Unavailable"],
      ["Change-point index", Number.isInteger(evidence.index) ? evidence.index : "Unavailable"],
      ["Baseline period", `${time(window.start)} – ${time(evidence.detected_change_point)}`],
      ["Anomalous period", `${time(evidence.detected_change_point)} – ${time(window.end)}`],
      ["Baseline observations", evidence.baseline_observation_count ?? "Unavailable"],
      ["Post-change observations", evidence.anomalous_observation_count ?? "Unavailable"],
    ].forEach(([label, value]) => { list.append(node("dt", label, "col-sm-4")); list.append(node("dd", value, "col-sm-8")); });
    card.append(list);
    card.append(node("p", "The detector supplied this change point. BARO compares robust metric behaviour before and from that point; the first metric is shown because it received the largest score in this run.", "small mb-0"));
    target.append(card);
  }

  function renderTechnicalRanking(result, rows, target) {
    const details = node("details", null, "mb-3");
    details.append(node("summary", "Full technical ranking"));
    if (!rows.length) {
      details.append(node("p", "No bounded ranking rows are available.", "text-muted mt-2"));
      target.append(details); return;
    }
    const responsive = node("div", null, "table-responsive mt-2");
    const table = node("table", null, "table table-sm align-middle");
    const head = node("thead"); const row = node("tr");
    ["Rank","Feature","Entity","Metric","Score"].forEach((label)=>row.append(node("th",label))); head.append(row); table.append(head);
    const body = node("tbody");
    rows.forEach((item)=>{
      const tr=node("tr");
      const identity=rcaPresentation.entity(item);
      [item.rank ?? "Unavailable",item.feature_id||item.feature_name||"Unavailable",identity.label,item.metric_name||"Unavailable",rankingScore(item)].forEach((value)=>tr.append(node("td",value)));
      body.append(tr);
    });
    table.append(body); responsive.append(table); details.append(responsive);
    if (result && result.truncated) details.append(node("p", "The ranking is truncated to the first 100 rows.", "small text-muted"));
    target.append(details);
  }

  function renderRcaDetail(payload, target) {
    const run = payload.run || {};
    const result = payload.result || null;
    const summary = rcaPresentation.summary(payload);
    const rows = rcaPresentation.ranking(result);
    target.replaceChildren();
    const heading = node("div", null, "d-flex justify-content-between align-items-center mb-3");
    const title = node("div");
    title.append(node("strong", summary.method_name));
    title.append(node("div", run.run_id || "Stored RCA run", "small text-muted provenance-value"));
    heading.append(title); heading.append(badge(run.status)); target.append(heading);
    if (run.failure) target.append(node("p", run.failure.message || "RCA execution failed.", "alert alert-danger"));
    if (run.skip_reason) target.append(node("p", run.skip_reason, "alert alert-warning"));

    const overview = node("section", null, "mb-3");
    overview.append(node("h4", "Operator summary", "h6"));
    const grid = node("div", null, "row g-2");
    const window = summary.analysis_window || {};
    grid.append(summaryValue("RCA method", summary.method_name));
    grid.append(summaryValue("Analysis window", window.start || window.end ? `${time(window.start)} – ${time(window.end)}` : "Unavailable"));
    grid.append(summaryValue("Most strongly represented entity", summary.strongest_entity.label));
    grid.append(summaryValue("Strongest metric", summary.strongest_metric));
    grid.append(summaryValue("Analysed observations", summary.observation_count ?? "Unavailable"));
    grid.append(summaryValue("Analysed features", summary.feature_count ?? "Unavailable"));
    overview.append(grid); target.append(overview);

    if (rows.length) renderSuspects(rows.slice(0, 3), target);
    else if (run.status === "completed") target.append(node("p", "This completed run contains no ranking rows.", "alert alert-secondary"));
    else if (["failed","skipped"].includes(run.status)) target.append(node("p", "No ranking is available for this run.", "alert alert-secondary"));

    if (summary.method_id === rcaPresentation.BARO) renderBaroEvidence(summary, target);
    if (result && Array.isArray(result.graphs)) result.graphs.forEach((graph)=>{ if(graph && graph.media_type==="image/png" && /^[A-Za-z0-9+/=]+$/.test(graph.base64||"")){ const figure=node("figure",null,"mt-3"); const img=node("img"); img.className="img-fluid border rounded"; img.alt="Stored CausalRCA dependency graph"; img.src="data:image/png;base64,"+graph.base64; figure.append(img); figure.append(node("figcaption","Dependency graph produced by CausalRCA for this stored run.","small text-muted mt-1")); target.append(figure); } });
    renderTechnicalRanking(result, rows, target);
    const details=node("details",null,"mt-3"); details.append(node("summary","RCA provenance")); const pre=node("pre",JSON.stringify({method:summary.method_id,cycle_id:run.cycle_id,analysis_kind:run.analysis_kind,execution_attempt_number:run.execution_attempt_number,episode_attempt_budget:run.episode_attempt_budget,previous_run_id:run.previous_run_id,trigger_percentage:run.trigger_percentage,threshold_percentage:run.threshold_percentage,analysis_window:run.analysis_window,observation_count:run.observation_count,feature_count:run.feature_count,new_observation_count:run.new_observation_count,input_sha256:run.input_sha256,analysis_input_sha256:run.analysis_input_sha256,result_sha256:run.result_sha256,policy_version:run.policy_version,preprocessing:run.preprocessing,algorithm:run.algorithm,change_point_evidence:summary.change_point},null,2),"small mt-2 provenance-value"); details.append(pre); target.append(details);
    target.append(node("p", "This result is diagnostic evidence, not proof of the actual root cause.", "alert alert-warning mt-3 mb-0"));
  }

  function rcaCacheKey(sessionId, run) {
    return `${sessionId}:${run.run_id}:${run.result_sha256 || run.updated_at || run.status || "unknown"}`;
  }

  function cacheRcaResult(key, payload) {
    state.rcaResultCache.set(key, payload);
    while (state.rcaResultCache.size > 50) {
      state.rcaResultCache.delete(state.rcaResultCache.keys().next().value);
    }
  }

  async function loadRcaDetail(sessionId, run, target) {
    const runId = run.run_id;
    selectRca(runId);
    const cacheKey = rcaCacheKey(sessionId, run);
    const cached = state.rcaResultCache.get(cacheKey);
    if (cached) {
      renderRcaDetail(cached, target);
      return;
    }
    target.replaceChildren(node("p", "Loading stored RCA result…", "text-muted"));
    try {
      const url = root.dataset.rcaUrlTemplate.replace("SESSION_ID", encodeURIComponent(sessionId)).replace("RUN_ID", encodeURIComponent(runId));
      const payload = await json(url);
      const loadedRun = payload.run || {};
      if (["completed","failed","skipped"].includes(loadedRun.status)) {
        cacheRcaResult(cacheKey, payload);
        cacheRcaResult(rcaCacheKey(sessionId, loadedRun), payload);
      }
      if (!target.isConnected || state.selected !== sessionId || state.selectedRca !== runId) return;
      renderRcaDetail(payload, target);
    } catch(error) {
      if (target.isConnected && state.selected === sessionId && state.selectedRca === runId) {
        target.replaceChildren(node("p", error.message, "alert alert-danger"));
      }
    }
  }

  function renderRca(session, payload, body) {
    const config = session.rca || {enabled:false, readiness:{status:"unavailable"}};
    const section = node("section", null, "border rounded p-3 mb-4");
    section.append(node("h3", "Automatic root-cause analysis", "h5"));
    const readiness = (config.readiness || {}).status || "unavailable";
    const summary = config.enabled
      ? `Enabled · ${rcaPresentation.methodName(config.method || rcaPresentation.CAUSAL)} · trigger strictly above ${config.threshold_percentage}% · ${Math.round((config.analysis_window_seconds||0)/60)} minute analysis window · ${Math.round((config.cooldown_seconds||0)/60)} minute cooldown`
      : "Disabled for this session.";
    section.append(node("p", summary));
    if (config.enabled && config.reanalyze_within_episode) {
      section.append(node("p", `Bounded same-episode reanalysis enabled · not more often than every ${config.reanalysis_min_interval_seconds} seconds · at least ${config.reanalysis_min_new_observations} new observations · ${config.attempt_count||0} / ${config.max_attempts_per_episode} accepted attempts used in this episode.`, "small text-muted"));
      if (config.last_intent_at) {
        const next = new Date(new Date(config.last_intent_at).getTime() + Number(config.reanalysis_min_interval_seconds||0)*1000);
        section.append(node("p", `Reanalysis is not eligible before ${time(next.toISOString())}, and only if the threshold remains exceeded, enough new data is available, no run is active, and budget remains.`, "small text-muted"));
      }
    } else if (config.enabled) {
      section.append(node("p", "Same-episode reanalysis is disabled. RCA is not run again after this episode has been analysed; select the earlier analysis below.", "small text-muted"));
    }
    section.append(node("p", `Input readiness: ${readiness}${config.readiness && config.readiness.reason ? " — "+config.readiness.reason : ""}`, "small text-muted"));
    section.append(node("p", "A threshold crossing starts at most one run per episode. No-new-data cycles do not recover or re-arm an episode.", "small text-muted"));
    section.append(node("p", "A persistently high anomaly percentage is not by itself proof of a reliable incident detection.", "small text-muted"));
    const runs = (payload.rca_runs && payload.rca_runs.items) || [];
    const triggerEvents = (((payload.cycles||{}).items)||[]).filter((cycle)=>cycle.rca_trigger);
    if (triggerEvents.length) {
      const events=node("div",null,"mb-3"); events.append(node("strong","Recent trigger decisions","d-block mb-1"));
      triggerEvents.slice(0,10).forEach((cycle)=>{ const event=cycle.rca_trigger; const line=node("div",null,"small text-muted mb-1"); line.append(document.createTextNode(`${time((cycle.prediction_window||{}).end)} · ${rcaDecision(event)}${event.reason ? " — "+event.reason : ""}`)); if(event.related_run_id||event.previous_run_id){ const link=node("a","View earlier analysis","ms-2"); const url=new URL(window.location.href); url.searchParams.set("session",session.session_id); url.searchParams.set("rca",event.related_run_id||event.previous_run_id); link.href=url.toString(); line.append(link); } events.append(line); });
      section.append(events);
    }
    if (!runs.length) section.append(node("p", "No automatic RCA run has been stored for this session.", "alert alert-secondary"));
    else {
      const list=node("div",null,"list-group mb-3"); const detail=node("div",null,"border rounded p-3");
      runs.forEach((run)=>{ const button=node("button",null,"list-group-item list-group-item-action"); button.type="button"; const top=node("div",null,"d-flex justify-content-between gap-2"); const kind=run.analysis_kind==="reanalysis" ? `Reanalysis ${run.execution_attempt_number||""}` : run.analysis_kind==="preflight_skip" ? "Preflight skip" : "First analysis"; top.append(node("span",`${kind} · ${time((run.analysis_window||{}).start)} – ${time((run.analysis_window||{}).end)}`)); top.append(badge(run.status)); button.append(top); button.append(node("small",`Triggered at ${run.trigger_percentage}% (threshold ${run.threshold_percentage}%)${run.new_observation_count!==undefined ? ` · ${run.new_observation_count} new observations` : ""}`,"text-muted")); button.addEventListener("click",()=>loadRcaDetail(session.session_id,run,detail)); list.append(button); });
      section.append(list); section.append(detail); if(!state.selectedRca) loadRcaDetail(session.session_id,runs[0],detail); else { const selected=runs.find((run)=>run.run_id===state.selectedRca)||runs[0]; loadRcaDetail(session.session_id,selected,detail); }
      if ((payload.rca_runs.pages || 0) > 1) {
        const navigation=node("div",null,"d-flex justify-content-between align-items-center mt-3");
        const recent=node("button","More recent RCA runs","btn btn-sm btn-outline-secondary"); recent.type="button"; recent.disabled=state.rcaPage<=1;
        const old=node("button","Older RCA runs","btn btn-sm btn-outline-secondary"); old.type="button"; old.disabled=state.rcaPage>=payload.rca_runs.pages;
        recent.addEventListener("click",()=>{state.rcaPage-=1;selectRca(null);loadAll(false);}); old.addEventListener("click",()=>{state.rcaPage+=1;selectRca(null);loadAll(false);});
        navigation.append(recent); navigation.append(node("span",`RCA page ${payload.rca_runs.page} of ${payload.rca_runs.pages}`,"small text-muted")); navigation.append(old); section.append(navigation);
      }
    }
    body.append(section);
  }

  function chartCard(payload, cycles, existingCard) {
    let card = existingCard;
    if (!card) {
      card = node("section", null, "border rounded p-3 mb-4");
      card.id = "monitoring-cycle-chart-card";
      card.append(node("h3", "Anomaly percentage by committed cycle", "h5"));
      const wrap = node("div", null, "chart-wrap");
      wrap.id = "monitoring-cycle-chart-wrap";
      const canvas = node("canvas");
      canvas.id = "monitoring-cycle-chart";
      wrap.append(canvas);
      card.append(wrap);
      const empty = node("p", "No cycle with newly evaluated predictions is available yet.", "text-muted");
      empty.id = "monitoring-cycle-chart-empty";
      card.append(empty);
      const note = node("p", "The chart shows the current history page. Older cycles remain available on older history pages.", "small text-muted");
      note.id = "monitoring-cycle-chart-note";
      card.append(note);
      const navigation = node("div", null, "d-flex justify-content-between align-items-center mt-3");
      navigation.id = "monitoring-cycle-chart-navigation";
      card.append(navigation);
    }
    const hasData = cycles.some((cycle) => cycle.anomaly_percentage !== null && cycle.anomaly_percentage !== undefined);
    card.querySelector("#monitoring-cycle-chart-wrap").classList.toggle("d-none", !hasData);
    card.querySelector("#monitoring-cycle-chart-empty").classList.toggle("d-none", hasData);
    card.querySelector("#monitoring-cycle-chart-note").classList.toggle("d-none", !payload.cycles.truncated);
    const navigation = card.querySelector("#monitoring-cycle-chart-navigation");
    navigation.replaceChildren();
    navigation.classList.toggle("d-none", (payload.cycles.pages || 0) <= 1);
    if ((payload.cycles.pages || 0) > 1) {
      const recent = node("button", "More recent cycles", "btn btn-sm btn-outline-secondary"); recent.type="button"; recent.disabled=state.cyclePage<=1;
      const old = node("button", "Older cycles", "btn btn-sm btn-outline-secondary"); old.type="button"; old.disabled=state.cyclePage>=payload.cycles.pages;
      recent.addEventListener("click",()=>{state.cyclePage-=1;state.detailRenderKey=null;loadAll(false);}); old.addEventListener("click",()=>{state.cyclePage+=1;state.detailRenderKey=null;loadAll(false);});
      navigation.append(recent); navigation.append(node("span",`Cycle page ${payload.cycles.page} of ${payload.cycles.pages}`,"small text-muted")); navigation.append(old);
    }
    return card;
  }

  function renderDetail(payload) {
    const session = payload.session;
    const cycles = (payload.cycles && payload.cycles.items) || [];
    const key = renderKey({selected:state.selected, cyclePage:state.cyclePage, rcaPage:state.rcaPage, payload});
    if (state.detailRenderKey === key) return false;
    state.detailRenderKey = key;
    const preservedChartCard = document.getElementById("monitoring-cycle-chart-card");
    detailNode.replaceChildren();
    const card = node("article", null, "card");
    const body = node("div", null, "card-body");
    const header = node("div", null, "d-flex justify-content-between align-items-start flex-wrap gap-3 mb-3");
    const title = node("div");
    title.append(node("h2", session.display_name || "Monitoring session", "h4 mb-1"));
    title.append(node("div", session.model_id, "small text-muted provenance-value"));
    header.append(title); header.append(badge(session.status)); body.append(header);
    const latest = session.latest_cycle || {};
    const metrics = node("div", null, "row g-3 mb-4");
    metrics.append(metric("Last successful update", time(session.last_successful_update)));
    metrics.append(metric("Newly evaluated", latest.prediction_count === undefined ? "Unavailable" : latest.prediction_count));
    metrics.append(metric("Latest anomalies", latest.anomaly_count === undefined ? "Unavailable" : latest.anomaly_count));
    metrics.append(metric("Warm-up observations", latest.warmup_observations === undefined ? (session.warmup_observations || 0) : latest.warmup_observations));
    metrics.append(metric("Buffered observations", session.buffered_observations === undefined ? "Unavailable" : session.buffered_observations));
    body.append(metrics);
    if (latest.prediction_count > 0) {
      body.append(node("p", `${latest.anomaly_count} / ${latest.prediction_count} evaluated observations anomalous (${latest.anomaly_percentage.toFixed(2)}%). This is the latest newly evaluated cycle, not a cumulative incident count.`, "alert alert-info"));
    } else if (session.status === "warmup") {
      body.append(node("p", "The model is warming up; no prediction timestamp is eligible yet.", "alert alert-warning"));
    } else {
      body.append(node("p", "No new observations were eligible in the latest cycle. Missing percentages are not plotted as 0%.", "alert alert-secondary"));
    }
    if (session.status === "interrupted") body.append(node("p", "Execution state is unavailable. This historical session will not resume automatically.", "alert alert-warning"));
    const updatedAt = session.last_successful_update ? new Date(session.last_successful_update).getTime() : NaN;
    if (["active","collecting","warmup"].includes(session.status) && Number.isFinite(updatedAt) && Date.now() - updatedAt > Math.max(30000, (session.poll_interval_seconds || 0) * 2000)) {
      body.append(node("p", "This session is stale: no successful cycle was committed within two poll intervals.", "alert alert-warning"));
    }
    if (session.error) body.append(node("p", session.error.message || session.error, "alert alert-danger"));
    (session.warnings || []).forEach((warning) => body.append(node("p", warning, "alert alert-warning py-2")));
    body.append(chartCard(payload, cycles, preservedChartCard));
    renderRca(session, payload, body);
    const details = node("details", null, "mb-3"); details.append(node("summary", "Provenance details"));
    const list = node("dl", null, "row mt-3");
    Object.entries(payload.provenance || {}).forEach(([key,value]) => { list.append(node("dt", key.replaceAll("_", " "), "col-sm-4")); list.append(node("dd", value || "Unavailable", "col-sm-8 provenance-value")); });
    details.append(list); body.append(details);
    if (["queued","collecting","warmup","active"].includes(session.status) && session.execution_state === "available") {
      const stop = node("button", "Stop monitoring", "btn btn-outline-danger"); stop.type="button";
      stop.addEventListener("click", async () => {
        stop.disabled=true;
        try {
          await json(root.dataset.stopUrlTemplate.replace("SESSION_ID", encodeURIComponent(session.session_id)), {method:"POST",headers:{Accept:"application/json","X-CSRFToken":root.dataset.csrfToken}});
          await loadAll(false);
        } catch(error) { showError(error.message); stop.disabled=false; }
      });
      body.append(stop);
    }
    card.append(body); detailNode.append(card); renderChart(cycles, session.rca);
    return true;
  }

  function schedulePoll() {
    clearTimeout(state.timer);
    if (state.shouldPoll && !document.hidden) state.timer=setTimeout(() => loadAll(true), 4000);
  }

  async function loadAll(_schedule) {
    clearTimeout(state.timer);
    let shouldPoll = false;
    try {
      const list = await json(`${root.dataset.listUrl}?page=${state.page}&page_size=20`);
      renderSessions(list);
      if (state.selected) {
        const detailUrl = root.dataset.detailUrlTemplate.replace("SESSION_ID", encodeURIComponent(state.selected));
        const detail = await json(`${detailUrl}?cycle_page=${state.cyclePage}&cycle_page_size=100&rca_page=${state.rcaPage}&rca_page_size=20`);
        renderDetail(detail);
        shouldPoll = ["queued","collecting","warmup","active"].includes((detail.session||{}).status)
          || (((detail.rca_runs||{}).items)||[]).some((run)=>["queued","running"].includes(run.status));
      }
      clearError();
    } catch (error) { showError(error.message); shouldPoll = true; }
    state.shouldPoll = shouldPoll;
    schedulePoll();
  }

  newer.addEventListener("click", () => { if (state.page > 1) { state.page -= 1; select(null); loadAll(false); } });
  older.addEventListener("click", () => { if (state.page < state.pages) { state.page += 1; select(null); loadAll(false); } });
  window.addEventListener("pagehide", () => clearTimeout(state.timer));
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) clearTimeout(state.timer);
    else if (state.shouldPoll) loadAll(true);
  });
  loadAll(true);
})();
