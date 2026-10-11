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
    chart: null,
    timer: null,
  };

  function node(tag, text, className) {
    const element = document.createElement(tag);
    if (text !== undefined && text !== null) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  }

  function badge(status) {
    const styles = {active:"success", warmup:"warning", collecting:"info", queued:"secondary", stopped:"secondary", failed:"danger", error:"danger", interrupted:"warning"};
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
      button.addEventListener("click", () => { state.cyclePage=1; select(session.session_id); loadAll(false); });
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

  function renderChart(cycles) {
    if (state.chart) { state.chart.destroy(); state.chart = null; }
    const canvas = document.getElementById("monitoring-cycle-chart");
    if (!canvas || typeof Chart === "undefined") return;
    const ordered = cycles.slice().reverse();
    state.chart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels: ordered.map((cycle) => time((cycle.prediction_window || {}).end || (cycle.query_window || {}).end)),
        datasets: [{
          label: "Anomalous newly evaluated observations (%)",
          data: ordered.map((cycle) => cycle.anomaly_percentage === null || cycle.anomaly_percentage === undefined ? null : cycle.anomaly_percentage),
          borderColor: "#0d6efd", backgroundColor: "rgba(13,110,253,.12)", pointRadius: 4, tension: .2, spanGaps: false,
        }],
      },
      options: {responsive:true, maintainAspectRatio:false, scales:{y:{min:0,max:100,title:{display:true,text:"Anomaly percentage (%)"}},x:{title:{display:true,text:"Latest newly evaluated prediction time"}}}},
    });
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
    const details = node("details", null, "mb-3"); details.append(node("summary", "Provenance details"));
    const list = node("dl", null, "row mt-3");
    Object.entries(payload.provenance || {}).forEach(([key,value]) => { list.append(node("dt", key.replaceAll("_", " "), "col-sm-4")); list.append(node("dd", value || "Unavailable", "col-sm-8 provenance-value")); });
    details.append(list); body.append(details);
    body.append(node("p", "Automatic CausalRCA is not available in this restoration tranche.", "small text-muted"));
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
    card.append(body); detailNode.append(card); renderChart(cycles);
  }

  async function loadAll(schedule) {
    try {
      const list = await json(`${root.dataset.listUrl}?page=${state.page}&page_size=20`);
      renderSessions(list);
      if (state.selected) {
        const detailUrl = root.dataset.detailUrlTemplate.replace("SESSION_ID", encodeURIComponent(state.selected));
        renderDetail(await json(`${detailUrl}?cycle_page=${state.cyclePage}&cycle_page_size=100`));
      }
      clearError();
    } catch (error) { showError(error.message); }
    if (schedule !== false) { clearTimeout(state.timer); state.timer=setTimeout(() => loadAll(true), 4000); }
  }

  newer.addEventListener("click", () => { if (state.page > 1) { state.page -= 1; select(null); loadAll(false); } });
  older.addEventListener("click", () => { if (state.page < state.pages) { state.page += 1; select(null); loadAll(false); } });
  window.addEventListener("pagehide", () => clearTimeout(state.timer));
  loadAll(true);
})();
