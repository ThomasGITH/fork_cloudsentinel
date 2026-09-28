(function () {
  "use strict";
  document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("[data-overview-filter]");
    const loading = document.querySelector("[data-overview-loading]");
    if (!form || !loading) return;
    form.addEventListener("submit", function () {
      form.setAttribute("aria-busy", "true");
      form.querySelectorAll("button").forEach(function (button) { button.disabled = true; });
      loading.hidden = false;
    });
  });
})();
