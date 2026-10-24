(function () {
  "use strict";
  document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("telemetry-form");
    if (!form) return;
    const start = document.getElementById("telemetry-start");
    const end = document.getElementById("telemetry-end");
    [start, end].forEach(function (input) {
      if (input.dataset.initial) input.value = input.dataset.initial.replace("Z", "").slice(0, 16);
    });
    let chart = null;
    function utc(input) { return new Date(input.value + "Z").toISOString(); }
    function activateLink(node, href) {
      if (!href) return;
      node.href = href; node.classList.remove("disabled"); node.removeAttribute("aria-disabled");
    }
    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      const error = document.getElementById("telemetry-error"); error.hidden = true;
      const payload = {query: document.getElementById("telemetry-query").value, start_time: utc(start), end_time: utc(end), step: Number(document.getElementById("telemetry-step").value)};
      try {
        const response = await fetch(window.location.href, {method: "POST", headers: {"Content-Type": "application/json", "X-CSRFToken": form.querySelector("[name=csrfmiddlewaretoken]").value}, body: JSON.stringify(payload)});
        const data = await response.json(); if (!response.ok) throw new Error(data.error || "The query failed.");
        const colors = ["#0d6efd", "#dc3545", "#198754", "#fd7e14", "#6f42c1", "#20c997", "#d63384", "#6c757d"];
        const labels = data.series && data.series[0] ? (data.series[0].values || []).map(function (point) { return new Date(Number(point[0]) * 1000).toLocaleString(); }) : [];
        const datasets = (data.series || []).map(function (series, index) {
          const labels = series.metric || {}; const name = labels.__name__ || Object.entries(labels).map(function (item) { return item[0] + "=" + item[1]; }).join(", ") || "series " + (index + 1);
          return {label: name, data: (series.values || []).map(function (point) { return Number(point[1]); }), borderColor: colors[index % colors.length], pointRadius: 0, borderWidth: 2, tension: 0.15};
        });
        if (chart) chart.destroy(); chart = new Chart(document.getElementById("telemetry-chart"), {type: "line", data: {labels: labels, datasets: datasets}, options: {responsive: true, maintainAspectRatio: false, interaction: {mode: "nearest", intersect: false}}});
        document.getElementById("telemetry-summary").textContent = datasets.length + " series · " + data.start_time + " to " + data.end_time;
        const datasetParams = new URLSearchParams({namespace: document.getElementById("telemetry-namespace").value, query: data.query, start_time: data.start_time, end_time: data.end_time, step: String(data.step)});
        activateLink(document.getElementById("telemetry-dataset-link"), "/data-catalogue/new/?" + datasetParams.toString());
        activateLink(document.getElementById("telemetry-grafana"), data.external_links && data.external_links.grafana);
        activateLink(document.getElementById("telemetry-prometheus"), data.external_links && data.external_links.prometheus);
      } catch (caught) { error.textContent = caught.message; error.hidden = false; }
    });
  });
})();
