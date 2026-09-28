(function () {
  "use strict";
  document.addEventListener("DOMContentLoaded", function () {
    const poller = document.querySelector("[data-fetch-poller]");
    let timer = null;
    let controller = null;

    function setText(selector, value) {
      const node = document.querySelector(selector);
      if (node) node.textContent = value == null ? "" : String(value);
    }
    function renderWarnings(values) {
      const list = document.querySelector("[data-fetch-warnings]");
      if (!list) return;
      list.replaceChildren();
      (values || []).forEach(function (value) {
        const item = document.createElement("li");
        item.textContent = String(value);
        list.appendChild(item);
      });
    }
    async function poll() {
      if (!poller || controller) return;
      controller = new AbortController();
      try {
        const response = await fetch(poller.dataset.statusUrl, {headers: {Accept: "application/json"}, signal: controller.signal});
        const data = await response.json();
        if (!response.ok) {
          const statusError = new Error(data.error || "Fetch status is temporarily unavailable.");
          statusError.retryable = data.retryable !== false;
          throw statusError;
        }
        setText("[data-fetch-phase]", data.phase || data.status);
        const progress = data.progress || {};
        setText("[data-fetch-progress]", (progress.completed || 0) + " / " + (progress.total || 0));
        renderWarnings(data.warnings);
        const error = document.querySelector("[data-fetch-error]");
        if (error) { error.textContent = data.error || ""; error.hidden = !data.error; }
        if (data.status === "fetching" || data.status === "validating") timer = window.setTimeout(poll, 4000);
        else window.setTimeout(function () { window.location.reload(); }, 500);
      } catch (error) {
        if (error.name !== "AbortError") {
          const node = document.querySelector("[data-poll-network-error]");
          if (node) { node.textContent = error.message + " Retrying…"; node.hidden = false; }
          if (error.retryable !== false) timer = window.setTimeout(poll, 5000);
        }
      } finally { controller = null; }
    }
    if (poller && (poller.dataset.status === "fetching" || poller.dataset.status === "validating")) poll();
    window.addEventListener("pagehide", function () { if (timer) clearTimeout(timer); if (controller) controller.abort(); });
    document.querySelectorAll("form[data-fetch-form]").forEach(function (form) {
      form.addEventListener("submit", function () { form.querySelectorAll("button").forEach(function (button) { button.disabled = true; }); });
    });

    const preview = document.querySelector("[data-preview]");
    if (preview) {
      fetch(preview.dataset.url, {headers: {Accept: "application/json"}}).then(async function (response) {
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Preview is unavailable.");
        const table = document.createElement("table");
        table.className = "table table-sm table-striped mb-0";
        const head = document.createElement("thead");
        const row = document.createElement("tr");
        const featureNames = data.feature_names || [];
        ["timestamp"].concat(featureNames).forEach(function (name) { const cell = document.createElement("th"); cell.scope = "col"; cell.textContent = name; row.appendChild(cell); });
        head.appendChild(row); table.appendChild(head);
        const body = document.createElement("tbody");
        (data.rows || []).forEach(function (item) {
          const tr = document.createElement("tr");
          const timestampCell = document.createElement("td");
          timestampCell.textContent = item.timestamp == null ? "missing" : String(item.timestamp);
          tr.appendChild(timestampCell);
          const values = item.values || {};
          const missing = item.missing || {};
          featureNames.forEach(function (featureName) {
            const value = values[featureName];
            const td = document.createElement("td");
            td.textContent = value == null ? "missing" : String(value);
            if (missing[featureName] || value == null) td.className = "table-warning";
            tr.appendChild(td);
          });
          body.appendChild(tr);
        });
        table.appendChild(body); preview.replaceChildren(table);
      }).catch(function (error) { preview.replaceChildren(); const alert = document.createElement("div"); alert.className = "alert alert-secondary mb-0"; alert.setAttribute("role", "status"); alert.textContent = "Preview unavailable: " + error.message; preview.appendChild(alert); });
    }

    if (window.bootstrap && window.location.hash) {
      const trigger = document.querySelector("[data-bs-toggle='tab'][href='" + window.location.hash.replace(/[^#a-z-]/gi, "") + "']");
      if (trigger) new window.bootstrap.Tab(trigger).show();
    }
    document.querySelectorAll("[data-bs-toggle='tab']").forEach(function (tab) { tab.addEventListener("shown.bs.tab", function () { history.replaceState(null, "", tab.getAttribute("href")); }); });
  });
})();
