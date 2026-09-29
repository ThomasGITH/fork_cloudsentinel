(function () {
  "use strict";
  document.addEventListener("DOMContentLoaded", function () {
    const versionInput = document.querySelector("#version");
    document.querySelectorAll("[data-latest-version]").forEach(function (radio) {
      radio.addEventListener("change", function () {
        if (radio.checked && versionInput) versionInput.value = radio.dataset.latestVersion;
      });
      if (radio.checked && versionInput && !versionInput.value) versionInput.value = radio.dataset.latestVersion;
    });
    document.querySelectorAll("[data-nullable-mode]").forEach(function (select) {
      const input = select.nextElementSibling;
      function update() {
        if (!input) return;
        if (select.value === "none") { input.value = "__none__"; input.disabled = false; input.readOnly = true; }
        else { if (input.value === "__none__") input.value = ""; input.readOnly = false; }
      }
      select.addEventListener("change", update); update();
    });
    const form = document.querySelector("[data-training-wizard]");
    if (form) form.addEventListener("submit", function (event) {
      const submitter = event.submitter;
      if (!submitter || !submitter.matches("[data-submit-training]")) return;
      submitter.disabled = true;
      submitter.textContent = "Starting…";
    });
  });
})();
