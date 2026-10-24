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

  const pluginForm = document.getElementById("plugin-upload-form");
  if (!pluginForm) return;
  const builder = pluginForm.querySelector("[data-manifest-builder]");
  const upload = pluginForm.querySelector("[data-manifest-upload]");
  const manifestFile = document.getElementById("manifest-file");
  const parameterList = pluginForm.querySelector("[data-parameter-list]");
  const parameterJson = pluginForm.querySelector("[data-parameters-json]");

  function updateManifestMode() {
    const mode = pluginForm.querySelector("[name=manifest_mode]:checked").value;
    builder.hidden = mode !== "builder";
    upload.hidden = mode !== "upload";
    builder.querySelectorAll("input, select, textarea").forEach((field) => {
      field.disabled = mode !== "builder";
    });
    upload.querySelectorAll("input, select, textarea").forEach((field) => {
      field.disabled = mode !== "upload";
    });
    manifestFile.required = mode === "upload";
    ["detector-id", "detector-name", "detector-version", "detector-description", "entry-point"].forEach((id) => {
      document.getElementById(id).required = mode === "builder";
    });
  }

  function parameterRow() {
    const row = document.createElement("div");
    row.className = "border rounded p-3";
    row.setAttribute("data-parameter-row", "");
    row.innerHTML = '<div class="row g-2"><div class="col-md-4"><label class="form-label small">Name</label><input class="form-control form-control-sm" data-param="name" pattern="[A-Za-z_][A-Za-z0-9_]*" required></div><div class="col-md-3"><label class="form-label small">Type</label><select class="form-select form-select-sm" data-param="type"><option value="integer">Integer</option><option value="number">Number</option><option value="string">String</option><option value="boolean">Boolean</option></select></div><div class="col-md-5"><label class="form-label small">Default</label><input class="form-control form-control-sm" data-param="default" required></div><div class="col-12"><label class="form-label small">Description</label><input class="form-control form-control-sm" data-param="description" required></div><div class="col-md-3"><label class="form-label small">Minimum</label><input class="form-control form-control-sm" data-param="minimum" type="number"></div><div class="col-md-3"><label class="form-label small">Maximum</label><input class="form-control form-control-sm" data-param="maximum" type="number"></div><div class="col-md-4"><label class="form-label small">Allowed values</label><input class="form-control form-control-sm" data-param="allowed" placeholder="comma separated"></div><div class="col-md-2 d-flex flex-column justify-content-end"><div class="form-check"><input class="form-check-input" data-param="advanced" type="checkbox"><label class="form-check-label small">Advanced</label></div><button class="btn btn-sm btn-link text-danger p-0 text-start" type="button" data-remove-parameter>Remove</button></div></div>';
    row.querySelector("[data-remove-parameter]").addEventListener("click", () => row.remove());
    parameterList.appendChild(row);
  }

  function typedValue(raw, type) {
    if (type === "boolean") {
      if (!/^(true|false)$/i.test(raw)) throw new Error("Boolean defaults must be true or false.");
      return raw.toLowerCase() === "true";
    }
    if (type === "integer") {
      const value = Number(raw); if (!Number.isInteger(value)) throw new Error("Integer defaults must be whole numbers."); return value;
    }
    if (type === "number") {
      const value = Number(raw); if (!Number.isFinite(value)) throw new Error("Number defaults must be numeric."); return value;
    }
    return raw;
  }

  pluginForm.querySelectorAll("[name=manifest_mode]").forEach((field) => field.addEventListener("change", updateManifestMode));
  pluginForm.querySelector("[data-add-parameter]").addEventListener("click", parameterRow);
  pluginForm.addEventListener("submit", (event) => {
    if (pluginForm.querySelector("[name=manifest_mode]:checked").value !== "builder") return;
    try {
      const result = {};
      parameterList.querySelectorAll("[data-parameter-row]").forEach((row) => {
        const get = (name) => row.querySelector(`[data-param="${name}"]`);
        const name = get("name").value.trim(); const type = get("type").value;
        if (!name) throw new Error("Every training parameter needs a name.");
        const metadata = {type: type, default: typedValue(get("default").value.trim(), type), description: get("description").value.trim(), advanced: get("advanced").checked};
        if (!metadata.description) throw new Error(`Training parameter ${name} needs a description.`);
        ["minimum", "maximum"].forEach((field) => { if (get(field).value !== "") metadata[field] = Number(get(field).value); });
        const allowed = get("allowed").value.split(",").map((value) => value.trim()).filter(Boolean);
        if (allowed.length) metadata.allowed_values = allowed.map((value) => typedValue(value, type));
        result[name] = metadata;
      });
      parameterJson.value = JSON.stringify(result);
    } catch (error) {
      event.preventDefault(); window.alert(error.message);
    }
  });
  updateManifestMode();
})();
