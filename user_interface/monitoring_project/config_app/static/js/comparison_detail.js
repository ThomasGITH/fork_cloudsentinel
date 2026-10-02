(() => {
  const poller = document.querySelector("[data-comparison-poller]");
  let stopped = false;
  let controller = null;
  const terminal = new Set(["completed", "partial", "failed"]);
  async function poll() {
    if (stopped || !poller || poller.dataset.active !== "true") return;
    controller = new AbortController();
    try {
      const response = await fetch(poller.dataset.statusUrl, {signal: controller.signal, headers: {"Accept": "application/json"}});
      const payload = await response.json();
      if (!response.ok) {
        if ([400, 404, 409].includes(response.status)) {
          stopped = true;
          const alert = document.querySelector("[data-poll-error]");
          if (alert) { alert.hidden = false; alert.textContent = payload.error || "Comparison status is unavailable."; }
          return;
        }
        throw new Error(payload.error || "Comparison status is unavailable.");
      }
      const badge = document.querySelector("[data-comparison-status]");
      if (badge) { badge.textContent = payload.status; badge.className = `comparison-status comparison-status-${payload.status}`; }
      if (terminal.has(payload.status)) { window.location.reload(); return; }
      window.setTimeout(poll, 4000);
    } catch (error) {
      if (error.name === "AbortError") return;
      const alert = document.querySelector("[data-poll-error]");
      if (alert) { alert.hidden = false; alert.textContent = "Status could not be refreshed. Retrying…"; }
      window.setTimeout(poll, 6000);
    }
  }
  window.addEventListener("pagehide", () => { stopped = true; if (controller) controller.abort(); });
  if (poller) poll();

  const data = document.getElementById("comparison-results");
  const canvas = document.getElementById("comparison-timeline");
  const chartError = document.querySelector("[data-timeline-error]");
  if (!data || !canvas) return;

  function showChartError() {
    canvas.hidden = true;
    if (chartError) chartError.hidden = false;
  }

  if (typeof Chart === "undefined") {
    showChartError();
    return;
  }

  try {
    const results = JSON.parse(data.textContent);
    const colors = ["#004080", "#198754", "#d97706", "#6f42c1", "#0dcaf0"];
    const labels = Array.from(new Set(
      results.flatMap(item => (item.timeline || []).map(point => point.timestamp))
    )).sort();
    const datasets = results
      .filter(item => item.timeline && item.timeline.length)
      .map((item, index) => {
        const byTimestamp = new Map(
          item.timeline.map(point => [point.timestamp, point.score])
        );
        const anomalies = new Map(
          item.timeline.map(point => [point.timestamp, point.prediction === 1])
        );
        const color = colors[index % colors.length];
        return {
          label: item.display_name || item.model_id,
          data: labels.map(timestamp => (
            byTimestamp.has(timestamp) ? byTimestamp.get(timestamp) : null
          )),
          borderColor: color,
          pointBackgroundColor: color,
          pointRadius: labels.map(timestamp => anomalies.get(timestamp) ? 4 : 1),
          spanGaps: true,
          tension: .15
        };
      });

    if (!labels.length || !datasets.length) {
      showChartError();
      return;
    }

    new Chart(canvas, {
      type: "line",
      data: {labels, datasets},
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {legend: {position: "bottom"}},
        scales: {
          x: {ticks: {maxTicksLimit: 10}},
          y: {title: {display: true, text: "Anomaly score"}}
        }
      }
    });
  } catch (_error) {
    showChartError();
  }
})();
