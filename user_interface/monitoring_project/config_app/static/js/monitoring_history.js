(function () {
  "use strict";

  const root = document.getElementById("monitoring-history");
  if (!root) return;
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
    if (!value) return "Unavailable";
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? "Unavailable" : parsed.toLocaleString();
  }

  function select(sessionId) {
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
    listNode.replaceChildren();
    if (!items.length) {
      listNode.append(node("div", "No monitoring sessions found.", "p-3 text-muted"));
      detailNode.replaceChildren(node("div", "Start a monitoring session to create durable history.", "alert alert-secondary"));
      select(null);
    }
    if (!state.selected && items.length) select(items[0].session_id);
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
  }

  function metric(label, value) {
    const wrapper = node("div", null, "col-sm-6 col-xl-3");
    const card = node("div", null, "border rounded p-3 h-100");
    card.append(node("div", label, "small text-muted"));
    card.append(node("strong", value));
    wrapper.append(card);
    return wrapper;
  }

  function renderChart(cycles, rca) {
    if (state.chart) { state.chart.destroy(); state.chart = null; }
    const canvas = document.getElementById("monitoring-cycle-chart");
    if (!canvas || typeof Chart === "undefined") return;
    const ordered = cycles.slice().reverse();
    const threshold = rca && rca.enabled ? Number(rca.threshold_percentage) : null;
    const datasets = [{
      label: "Anomalous newly evaluated observations (%)",
      data: ordered.map((cycle) => cycle.anomaly_percentage === null || cycle.anomaly_percentage === undefined ? null : cycle.anomaly_percentage),
      borderColor: "#0d6efd", backgroundColor: "rgba(13,110,253,.12)",
      pointRadius: ordered.map((cycle) => cycle.rca_trigger && ["started","skipped"].includes(cycle.rca_trigger.status) ? 7 : 4),
      pointBackgroundColor: ordered.map((cycle) => cycle.rca_trigger && cycle.rca_trigger.status === "started" ? "#dc3545" : "#0d6efd"),
      tension: .2, spanGaps: false,
    }];
    if (Number.isFinite(threshold)) datasets.push({
      label: "Automatic RCA threshold (%)", data: ordered.map(() => threshold),
      borderColor: "#dc3545", borderDash: [7, 5], pointRadius: 0, tension: 0,
    });
    state.chart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels: ordered.map((cycle) => time((cycle.prediction_window || {}).end || (cycle.query_window || {}).end)),
        datasets,
      },
      options: {responsive:true, maintainAspectRatio:false, scales:{y:{min:0,max:100,title:{display:true,text:"Anomaly percentage (%)"}},x:{title:{display:true,text:"Latest newly evaluated prediction time"}}}},
    });
  }

  async function loadRcaDetail(sessionId, runId, target) {
    selectRca(runId);
    target.replaceChildren(node("p", "Loading stored CausalRCA result…", "text-muted"));
    try {
      const url = root.dataset.rcaUrlTemplate.replace("SESSION_ID", encodeURIComponent(sessionId)).replace("RUN_ID", encodeURIComponent(runId));
      const payload = await json(url);
      const run = payload.run || {};
      target.replaceChildren();
      const heading = node("div", null, "d-flex justify-content-between align-items-center mb-2");
      heading.append(node("strong", run.run_id || "CausalRCA run")); heading.append(badge(run.status)); target.append(heading);
      if (run.failure) target.append(node("p", run.failure.message || "CausalRCA failed.", "alert alert-danger"));
      if (run.skip_reason) target.append(node("p", run.skip_reason, "alert alert-warning"));
      const result = payload.result;
      if (result && Array.isArray(result.ranking) && result.ranking.length) {
        const table = node("table", null, "table table-sm table-responsive align-middle");
        const head = node("thead"); const row = node("tr"); ["Rank","Feature","Pod","Metric","Score"].forEach((label)=>row.append(node("th",label))); head.append(row); table.append(head);
        const body = node("tbody"); result.ranking.forEach((item)=>{ const tr=node("tr"); [item.rank,item.feature_name,item.entity_id,item.metric_name,Number(item.score).toFixed(6)].forEach((value)=>tr.append(node("td",value))); body.append(tr); }); table.append(body); target.append(table);
        if (result.truncated) target.append(node("p", "The ranking is truncated to the first 100 rows.", "small text-muted"));
      } else if (run.status === "completed") target.append(node("p", "No bounded ranking rows are available.", "text-muted"));
      if (result && Array.isArray(result.graphs)) result.graphs.forEach((graph)=>{ if(graph && graph.media_type==="image/png" && /^[A-Za-z0-9+/=]+$/.test(graph.base64||"")){ const img=node("img"); img.className="img-fluid border rounded mt-3"; img.alt="Stored CausalRCA dependency graph"; img.src="data:image/png;base64,"+graph.base64; target.append(img); } });
      const details=node("details",null,"mt-3"); details.append(node("summary","RCA provenance")); const pre=node("pre",JSON.stringify({cycle_id:run.cycle_id,trigger_percentage:run.trigger_percentage,threshold_percentage:run.threshold_percentage,analysis_window:run.analysis_window,input_sha256:run.input_sha256,result_sha256:run.result_sha256,preprocessing:run.preprocessing,algorithm:run.algorithm},null,2),"small mt-2 provenance-value"); details.append(pre); target.append(details);
      target.append(node("p", "This ranking is diagnostic evidence from the stored analysis window; it is not proof of the actual root cause.", "small text-muted mt-3"));
    } catch(error) { target.replaceChildren(node("p", error.message, "alert alert-danger")); }
  }

  function renderRca(session, payload, body) {
    const config = session.rca || {enabled:false, readiness:{status:"unavailable"}};
    const section = node("section", null, "border rounded p-3 mb-4");
    section.append(node("h3", "Automatic CausalRCA", "h5"));
    const readiness = (config.readiness || {}).status || "unavailable";
    const summary = config.enabled
      ? `Enabled · trigger strictly above ${config.threshold_percentage}% · ${Math.round((config.analysis_window_seconds||0)/60)} minute analysis window · ${Math.round((config.cooldown_seconds||0)/60)} minute cooldown`
      : "Disabled for this session.";
    section.append(node("p", summary));
    section.append(node("p", `Input readiness: ${readiness}${config.readiness && config.readiness.reason ? " — "+config.readiness.reason : ""}`, "small text-muted"));
    section.append(node("p", "A threshold crossing starts at most one run per episode. No-new-data cycles do not recover or re-arm an episode.", "small text-muted"));
    section.append(node("p", "A persistently high anomaly percentage is not by itself proof of a reliable incident detection.", "small text-muted"));
    const triggerEvents = (((payload.cycles||{}).items)||[]).filter((cycle)=>cycle.rca_trigger);
    if (triggerEvents.length) {
      const events=node("div",null,"mb-3"); events.append(node("strong","Recent trigger decisions","d-block mb-1"));
      triggerEvents.slice(0,10).forEach((cycle)=>{ const event=cycle.rca_trigger; events.append(node("div",`${time((cycle.prediction_window||{}).end)} · ${event.status}${event.reason ? " — "+event.reason : ""}`,"small text-muted")); });
      section.append(events);
    }
    const runs = (payload.rca_runs && payload.rca_runs.items) || [];
    if (!runs.length) section.append(node("p", "No automatic CausalRCA run has been stored for this session.", "alert alert-secondary"));
    else {
      const list=node("div",null,"list-group mb-3"); const detail=node("div",null,"border rounded p-3");
      runs.forEach((run)=>{ const button=node("button",null,"list-group-item list-group-item-action"); button.type="button"; const top=node("div",null,"d-flex justify-content-between gap-2"); top.append(node("span",`${time((run.analysis_window||{}).start)} – ${time((run.analysis_window||{}).end)}`)); top.append(badge(run.status)); button.append(top); button.append(node("small",`Triggered at ${run.trigger_percentage}% (threshold ${run.threshold_percentage}%)`,"text-muted")); button.addEventListener("click",()=>loadRcaDetail(session.session_id,run.run_id,detail)); list.append(button); });
      section.append(list); section.append(detail); if(!state.selectedRca) loadRcaDetail(session.session_id,runs[0].run_id,detail); else { const selected=runs.find((run)=>run.run_id===state.selectedRca)||runs[0]; loadRcaDetail(session.session_id,selected.run_id,detail); }
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

  function renderDetail(payload) {
    const session = payload.session;
    const cycles = (payload.cycles && payload.cycles.items) || [];
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
    const chartCard = node("section", null, "border rounded p-3 mb-4");
    chartCard.append(node("h3", "Anomaly percentage by committed cycle", "h5"));
    if (cycles.some((cycle) => cycle.anomaly_percentage !== null && cycle.anomaly_percentage !== undefined)) {
      const wrap = node("div", null, "chart-wrap"); const canvas = node("canvas"); canvas.id = "monitoring-cycle-chart"; wrap.append(canvas); chartCard.append(wrap);
    } else chartCard.append(node("p", "No cycle with newly evaluated predictions is available yet.", "text-muted"));
    if (payload.cycles.truncated) chartCard.append(node("p", "The chart shows the current history page. Older cycles remain available on older history pages.", "small text-muted"));
    if ((payload.cycles.pages || 0) > 1) {
      const navigation = node("div", null, "d-flex justify-content-between align-items-center mt-3");
      const recent = node("button", "More recent cycles", "btn btn-sm btn-outline-secondary"); recent.type="button"; recent.disabled=state.cyclePage<=1;
      const old = node("button", "Older cycles", "btn btn-sm btn-outline-secondary"); old.type="button"; old.disabled=state.cyclePage>=payload.cycles.pages;
      recent.addEventListener("click",()=>{state.cyclePage-=1;loadAll(false);}); old.addEventListener("click",()=>{state.cyclePage+=1;loadAll(false);});
      navigation.append(recent); navigation.append(node("span",`Cycle page ${payload.cycles.page} of ${payload.cycles.pages}`,"small text-muted")); navigation.append(old); chartCard.append(navigation);
    }
    body.append(chartCard);
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
    if (shouldPoll) state.timer=setTimeout(() => loadAll(true), 4000);
  }

  newer.addEventListener("click", () => { if (state.page > 1) { state.page -= 1; select(null); loadAll(false); } });
  older.addEventListener("click", () => { if (state.page < state.pages) { state.page += 1; select(null); loadAll(false); } });
  window.addEventListener("pagehide", () => clearTimeout(state.timer));
  loadAll(true);
})();
