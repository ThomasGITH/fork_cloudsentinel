(function () {
  "use strict";
  document.addEventListener("DOMContentLoaded", function () {
    const root = document.querySelector("[data-run-poller]");
    if (!root || root.dataset.active !== "true") return;
    let timer = null;
    let controller = null;
    const terminal = new Set(["completed", "failed", "dispatch_failed", "validation_failed"]);
    function text(node, value) { if (node) node.textContent = value == null || value === "" ? "—" : String(value); }
    function dateTime(node, value) { text(node, window.CloudSentinel.formatDateTime(value)); }
    function badge(node, status) { if (!node) return; node.className = "model-status model-status-" + String(status || "unknown").replace(/[^a-z_]/g, ""); text(node, status); }
    function showMessage(node, value) { if (!node) return; node.textContent = value || ""; node.hidden = !value; }
    function progressText(child) {
      if (typeof child.display_detail === "string") return child.display_detail;
      const phase = typeof child.progress_phase === "string" ? child.progress_phase.replace(/_/g, " ").toLowerCase().replace(/\b\w/g, function (letter) { return letter.toUpperCase(); }) : "";
      const detail = child.status_detail || child.detail;
      if (typeof detail === "string") return phase && detail.toLowerCase() !== phase.toLowerCase() ? phase + " — " + detail : detail;
      if (detail && typeof detail === "object") {
        if (typeof detail.message === "string") return phase ? phase + " — " + detail.message : detail.message;
        if (Array.isArray(detail.values) && detail.values.length >= 4 && detail.values.slice(0, 4).every(Number.isInteger)) {
          const outer = (phase === "" || phase === "Training") ? "Epoch" : "Step";
          return (phase || "Training") + " — " + outer + " " + (detail.values[0] + 1) + " of " + detail.values[1] + " · batch " + (detail.values[2] + 1) + " of " + detail.values[3];
        }
      }
      return phase;
    }
    function render(data) {
      badge(root.querySelector("[data-parent-status]"), data.status);
      dateTime(root.querySelector("[data-run-started]"), data.started_at); dateTime(root.querySelector("[data-run-updated]"), data.updated_at); dateTime(root.querySelector("[data-run-completed]"), data.completed_at);
      let active = false;
      (data.children || []).forEach(function (child) {
        const escaped = window.CSS && CSS.escape ? CSS.escape(String(child.detector_id)) : String(child.detector_id).replace(/[^a-zA-Z0-9_-]/g, "");
        const card = root.querySelector('[data-child="' + escaped + '"]'); if (!card) return;
        badge(card.querySelector("[data-child-status]"), child.status); text(card.querySelector("[data-child-detail]"), progressText(child)); showMessage(card.querySelector("[data-child-validation]"), child.validation_error); showMessage(card.querySelector("[data-child-failure]"), child.failure_summary); text(card.querySelector("[data-child-promotion]"), child.promotion_status || child.model_status);
        if (!terminal.has(child.status)) active = true;
        if (child.model_id && child.model_status === "available") { const wrap = card.querySelector("[data-child-model]"); if (wrap && !wrap.querySelector("a")) { const link = document.createElement("a"); link.className = "btn btn-sm btn-outline-primary"; link.href = root.dataset.modelUrl.replace("__MODEL__", encodeURIComponent(child.model_id)); link.textContent = child.model_display_name || "View saved model"; wrap.appendChild(link); } }
      });
      const next = root.querySelector("[data-comparison-next]");
      if (next && data.comparison_flow) {
        const count = Number(data.comparison_flow.successful_count || 0);
        next.hidden = false;
        const summary = next.querySelector("[data-comparison-summary]");
        if (summary) summary.textContent = count + " model" + (count === 1 ? "" : "s") + " successfully trained. The next screen preselects these models and lets you add another compatible saved model if needed.";
        const link = next.querySelector("[data-comparison-link]");
        const unavailable = next.querySelector("[data-comparison-unavailable]");
        if (link) {
          link.hidden = !data.comparison_flow.can_compare;
          if (data.guided_comparison_url) link.href = data.guided_comparison_url;
        }
        if (unavailable) unavailable.hidden = Boolean(data.comparison_flow.can_compare);
      }
      return active;
    }
    async function poll() {
      if (controller) return; controller = new AbortController();
      try { const response = await fetch(root.dataset.statusUrl, {headers:{Accept:"application/json"}, signal:controller.signal}); const data = await response.json(); if (!response.ok) { const error = new Error(data.error || "Training status is unavailable."); error.retryable = data.retryable !== false; throw error; } showMessage(root.querySelector("[data-poll-error]"), ""); if (render(data)) timer = window.setTimeout(poll, 4000); }
      catch (error) { if (error.name !== "AbortError") { showMessage(root.querySelector("[data-poll-error]"), error.message + (error.retryable === false ? "" : " Retrying…")); if (error.retryable !== false) timer = window.setTimeout(poll, 5000); } }
      finally { controller = null; }
    }
    poll(); window.addEventListener("pagehide", function () { if (timer) window.clearTimeout(timer); if (controller) controller.abort(); });
  });
})();
