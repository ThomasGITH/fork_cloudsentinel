(() => {
  "use strict";

  const copyText = async (text) => {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const field = document.createElement("textarea");
    field.value = text;
    field.setAttribute("readonly", "");
    field.style.position = "fixed";
    field.style.opacity = "0";
    document.body.appendChild(field);
    field.select();
    const copied = document.execCommand("copy");
    field.remove();
    if (!copied) throw new Error("Copy unavailable");
  };

  document.querySelectorAll("[data-copy-target]").forEach((button) => {
    button.addEventListener("click", async () => {
      const target = document.getElementById(button.dataset.copyTarget || "");
      const wrapper = button.closest(".code-copy");
      const status = wrapper ? wrapper.querySelector("[data-copy-status]") : null;
      if (!target) return;
      try {
        await copyText(target.textContent || "");
        if (status) status.textContent = "Copied";
      } catch (_error) {
        if (status) status.textContent = "Copy unavailable; select the text manually.";
      }
    });
  });
})();
