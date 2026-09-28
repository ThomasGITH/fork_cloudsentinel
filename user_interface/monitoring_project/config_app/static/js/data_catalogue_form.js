(function () {
  "use strict";

  function localDateTime(isoValue) {
    if (!isoValue) return "";
    const date = new Date(isoValue);
    if (Number.isNaN(date.getTime())) return isoValue.slice(0, 16);
    return date.toISOString().slice(0, 16);
  }

  function initialiseUtcFields(root) {
    root.querySelectorAll("input[type='datetime-local'][data-initial]").forEach(function (input) {
      if (!input.value) input.value = localDateTime(input.dataset.initial);
    });
  }

  function syncUtcFields(form) {
    form.querySelectorAll("input[type='datetime-local'][data-utc-target]").forEach(function (input) {
      const hidden = form.querySelector("#" + input.dataset.utcTarget);
      if (hidden) hidden.value = input.value ? new Date(input.value + "Z").toISOString() : "";
    });
  }

  function setupPartition(form) {
    const selector = form.querySelector("[name='partition_mode']");
    const fields = form.querySelector("[data-partition-fields]");
    if (!selector || !fields) return;
    function update() {
      fields.hidden = selector.value !== "time_range";
      fields.querySelectorAll("input").forEach(function (input) { input.disabled = fields.hidden; });
    }
    selector.addEventListener("change", update);
    update();
  }

  function queryValue(card, name) {
    const field = card.querySelector("[data-query-field='" + name + "']");
    if (!field) return "";
    return field.type === "checkbox" ? field.checked : field.value;
  }

  function setupQueries(form) {
    const list = form.querySelector("[data-query-list]");
    const template = document.getElementById("query-card-template");
    const hidden = form.querySelector("[name='queries_json']");
    const initialNode = document.getElementById("initial-queries");
    if (!list || !template || !hidden) return;

    let initial = [];
    try { initial = JSON.parse(initialNode ? initialNode.textContent : "[]"); } catch (_) { initial = []; }
    if (!Array.isArray(initial) || !initial.length) initial = [{ required: true, execution_mode: "single" }];

    function updateNumbers() {
      list.querySelectorAll("[data-query-card]").forEach(function (card, index) {
        const number = card.querySelector("[data-query-number]");
        if (number) number.textContent = String(index + 1);
        const remove = card.querySelector("[data-remove-query]");
        if (remove) remove.disabled = list.children.length === 1;
      });
    }

    function addQuery(values) {
      const card = template.content.firstElementChild.cloneNode(true);
      Object.keys(values || {}).forEach(function (key) {
        const field = card.querySelector("[data-query-field='" + key + "']");
        if (!field) return;
        if (field.type === "checkbox") field.checked = Boolean(values[key]);
        else if (key === "identity_labels" && Array.isArray(values[key])) field.value = values[key].join(", ");
        else if (key === "parameters" && typeof values[key] === "object") field.value = JSON.stringify(values[key]);
        else field.value = values[key] == null ? "" : String(values[key]);
      });
      const context = values && values.target_context ? values.target_context : {};
      ["pod", "service"].forEach(function (key) {
        const field = card.querySelector("[data-query-field='target_context_" + key + "']");
        if (field) field.value = context[key] || "";
      });
      card.querySelector("[data-remove-query]").addEventListener("click", function () {
        card.remove();
        updateNumbers();
      });
      list.appendChild(card);
      updateNumbers();
    }

    initial.forEach(addQuery);
    const addButton = form.querySelector("[data-add-query]");
    if (addButton) addButton.addEventListener("click", function () { addQuery({ required: true, execution_mode: "single" }); });

    form.addEventListener("submit", function () {
      syncUtcFields(form);
      const values = [];
      list.querySelectorAll("[data-query-card]").forEach(function (card) {
        const query = {};
        ["query_id", "display_name", "feature_name", "promql_template", "required", "execution_mode", "target_type", "identity_labels", "target_context_pod", "target_context_service", "parameters", "expected_modality", "expected_result_type"].forEach(function (key) {
          query[key] = queryValue(card, key);
        });
        values.push(query);
      });
      hidden.value = JSON.stringify(values);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("form.catalogue-form").forEach(function (form) {
      initialiseUtcFields(form);
      setupPartition(form);
      setupQueries(form);
      if (!form.querySelector("[data-query-list]")) form.addEventListener("submit", function () { syncUtcFields(form); });
    });
  });
})();
