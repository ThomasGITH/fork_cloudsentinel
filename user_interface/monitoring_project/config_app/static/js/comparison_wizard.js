(() => {
  const form = document.querySelector("[data-comparison-form]");
  if (!form) return;
  form.addEventListener("submit", () => {
    const button = form.querySelector("[data-submit-button]");
    if (button) {
      button.disabled = true;
      button.textContent = "Starting comparison…";
    }
  });
})();
