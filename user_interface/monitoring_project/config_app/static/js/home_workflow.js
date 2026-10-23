(function () {
    "use strict";

    const page = document.querySelector("[data-workflow-status-url]");
    if (!page) return;

    const statusUrl = page.dataset.workflowStatusUrl;
    const retry = page.querySelector("[data-status-retry]");
    const nextTitle = page.querySelector("[data-next-title]");
    const nextDetail = page.querySelector("[data-next-detail]");
    const nextAction = page.querySelector("[data-next-action]");

    function updateStep(name, value) {
        const step = page.querySelector(`[data-workflow-step="${name}"]`);
        if (!step || !value) return;
        step.dataset.state = value.state || "unavailable";
        step.querySelector("[data-step-status]").textContent = value.label || "Unknown";
        step.querySelector("[data-step-metric]").textContent = value.metric || "Status unknown";
        step.querySelector("[data-step-detail]").textContent = value.detail || "No status detail is available.";
    }

    function setLoading() {
        if (retry) {
            retry.disabled = true;
            retry.textContent = "Checking…";
        }
        nextTitle.textContent = "Checking workflow status…";
        nextDetail.textContent = "CloudSentinel is checking the configured source and existing experiment records.";
        nextAction.textContent = "Checking…";
        nextAction.classList.add("disabled");
        nextAction.setAttribute("aria-disabled", "true");
    }

    function setNextStep(value) {
        nextTitle.textContent = value.label;
        nextDetail.textContent = value.detail;
        nextAction.textContent = value.label;
        nextAction.href = value.url;
        nextAction.classList.remove("disabled");
        nextAction.removeAttribute("aria-disabled");
    }

    function loadStatus() {
        setLoading();
        fetch(statusUrl, { headers: { "Accept": "application/json" }, credentials: "same-origin" })
            .then(function (response) {
                if (!response.ok) throw new Error("status request failed");
                return response.json();
            })
            .then(function (payload) {
                updateStep("source", payload.source);
                updateStep("catalogue", payload.catalogue);
                updateStep("models", payload.models);
                updateStep("comparisons", payload.comparisons);
                updateStep("monitoring", payload.models && payload.models.live_ready_count === 0 ? {
                    state: "action",
                    label: "Model needed",
                    metric: "0 live-ready models",
                    detail: "Train or select a saved model with an inference contract and live-input recipe."
                } : payload.monitoring);
                updateStep("review", payload.monitoring);
                setNextStep(payload.next_step);
            })
            .catch(function () {
                nextTitle.textContent = "Workflow status is temporarily unavailable";
                nextDetail.textContent = "The workflow links remain available. Retry the status check when the services are reachable.";
                nextAction.textContent = "Open Data Catalogue";
                nextAction.href = page.querySelector('[data-workflow-step="catalogue"] a').href;
                nextAction.classList.remove("disabled");
                nextAction.removeAttribute("aria-disabled");
            })
            .finally(function () {
                if (retry) {
                    retry.disabled = false;
                    retry.textContent = "Refresh status";
                }
            });
    }

    if (retry) retry.addEventListener("click", loadStatus);
    loadStatus();
}());
