(() => {
  const form = document.querySelector("[data-comparison-form]");
  if (!form) return;

  const filterRoot = form.querySelector("[data-model-filter-root]");
  if (filterRoot) {
    const cards = Array.from(filterRoot.querySelectorAll("[data-model-card]"));
    const search = filterRoot.querySelector("[data-model-search]");
    const algorithm = filterRoot.querySelector("[data-model-algorithm]");
    const createdFrom = filterRoot.querySelector("[data-model-created-from]");
    const createdUntil = filterRoot.querySelector("[data-model-created-until]");
    const selectedOnly = filterRoot.querySelector("[data-model-selected-only]");
    const count = filterRoot.querySelector("[data-model-filter-count]");
    const empty = filterRoot.querySelector("[data-model-filter-empty]");

    Array.from(new Set(cards.map((card) => card.dataset.modelAlgorithm).filter(Boolean)))
      .sort()
      .forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        algorithm.appendChild(option);
      });

    const timestamp = (value) => {
      if (!value) return null;
      const parsed = Date.parse(value);
      return Number.isNaN(parsed) ? null : parsed;
    };
    const applyFilters = () => {
      const term = search.value.trim().toLowerCase();
      const start = timestamp(createdFrom.value);
      const end = timestamp(createdUntil.value);
      let shown = 0;
      cards.forEach((card) => {
        const checkbox = card.querySelector('input[name="model_ids"]');
        const created = timestamp(card.dataset.modelCreated);
        const matches = (!term || card.dataset.modelName.includes(term) || card.dataset.modelId.includes(term))
          && (!algorithm.value || card.dataset.modelAlgorithm === algorithm.value)
          && (start === null || (created !== null && created >= start))
          && (end === null || (created !== null && created <= end))
          && (!selectedOnly.checked || Boolean(checkbox && checkbox.checked));
        card.hidden = !matches;
        if (matches) shown += 1;
      });
      count.textContent = shown + " of " + cards.length + " models shown";
      empty.hidden = shown !== 0;
    };
    [search, algorithm, createdFrom, createdUntil, selectedOnly].forEach((control) => {
      control.addEventListener(control === search ? "input" : "change", applyFilters);
    });
    filterRoot.addEventListener("change", (event) => {
      if (event.target.matches('input[name="model_ids"]') && selectedOnly.checked) applyFilters();
    });
    filterRoot.querySelector("[data-model-filter-clear]").addEventListener("click", () => {
      search.value = "";
      algorithm.value = "";
      createdFrom.value = "";
      createdUntil.value = "";
      selectedOnly.checked = false;
      applyFilters();
    });
    applyFilters();
  }

  form.addEventListener("submit", () => {
    const button = form.querySelector("[data-submit-button]");
    if (button) {
      button.disabled = true;
      button.textContent = "Starting comparison…";
    }
  });
})();
