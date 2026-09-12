const STEPS = [
  ["experiment", "01", "Experiment"],
  ["dataset", "02", "Dataset"],
  ["train", "03", "Train"],
  ["evaluate", "04", "Evaluate"],
  ["promote", "05", "Promote"],
  ["production", "06", "Production"],
];

const state = {
  experimentName: localStorage.getItem("wb.experimentName") || "sentiment-workbench",
  experimentId: localStorage.getItem("wb.experimentId") || "",
  datasetId: localStorage.getItem("wb.datasetId") || "",
  lastRunId: localStorage.getItem("wb.lastRunId") || "",
  modelName: localStorage.getItem("wb.modelName") || "sentiment-classifier",
};

function persist() {
  localStorage.setItem("wb.experimentName", state.experimentName);
  localStorage.setItem("wb.experimentId", state.experimentId);
  localStorage.setItem("wb.datasetId", state.datasetId);
  localStorage.setItem("wb.lastRunId", state.lastRunId);
  localStorage.setItem("wb.modelName", state.modelName);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || response.statusText);
  }
  return payload;
}

function showPanel(id) {
  document.querySelectorAll(".panel").forEach((el) => el.classList.toggle("active", el.id === `panel-${id}`));
  document.querySelectorAll(".step").forEach((el) => el.classList.toggle("active", el.dataset.step === id));
}

function renderSteps() {
  const nav = document.getElementById("steps");
  nav.innerHTML = STEPS.map(([id, num, label]) => {
    const ready =
      (id === "experiment" && state.experimentId) ||
      (id === "dataset" && state.datasetId) ||
      (id === "train" && state.lastRunId) ||
      (id === "evaluate" && state.experimentId) ||
      (id === "promote" && state.lastRunId) ||
      (id === "production" && state.modelName);
    return `<button class="step ${ready ? "ready" : ""}" data-step="${id}" type="button">
      <span>${num}</span><b>${label}</b>
    </button>`;
  }).join("");
  nav.querySelectorAll(".step").forEach((btn) => btn.addEventListener("click", () => showPanel(btn.dataset.step)));
}

function setHint(id, message, isError = false) {
  const el = document.getElementById(id);
  el.textContent = message;
  el.classList.toggle("error", Boolean(isError));
}

function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(3);
}

async function refreshHealth() {
  const health = await api("/api/health");
  const mlflow = document.getElementById("mlflow-pill");
  mlflow.textContent = health.mlflow_reachable ? "MLflow · connected" : "MLflow · unreachable";
  mlflow.className = `pill ${health.mlflow_reachable ? "ok" : "bad"}`;
  const openai = document.getElementById("openai-pill");
  openai.textContent = health.openai_configured ? `OpenAI · ${health.default_openai_model}` : "OpenAI · not configured";
  openai.className = `pill ${health.openai_configured ? "ok" : "warn"}`;
  const link = document.getElementById("mlflow-link");
  link.href = health.mlflow_ui_url;
  document.getElementById("promote-name").value = health.registered_model_name;
  state.modelName = health.registered_model_name;
  persist();
  return health;
}

async function refreshExperiments() {
  const { experiments } = await api("/api/experiments");
  const list = document.getElementById("experiment-list");
  if (!experiments.length) {
    list.innerHTML = "<li>No experiments yet.</li>";
    return;
  }
  list.innerHTML = experiments
    .map(
      (exp) => `<li>
        <button type="button" class="ghost pick-exp" data-id="${exp.experiment_id}" data-name="${exp.name}">${exp.name}</button>
        · <a href="${exp.url}" target="_blank" rel="noreferrer">MLflow</a>
      </li>`
    )
    .join("");
  list.querySelectorAll(".pick-exp").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.experimentId = btn.dataset.id;
      state.experimentName = btn.dataset.name;
      document.getElementById("experiment-name").value = state.experimentName;
      persist();
      renderSteps();
      setHint("experiment-msg", `Selected ${state.experimentName}.`);
    });
  });
}

document.getElementById("experiment-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = document.getElementById("experiment-name").value.trim();
  try {
    const exp = await api("/api/experiments", { method: "POST", body: JSON.stringify({ name }) });
    state.experimentName = exp.name;
    state.experimentId = exp.experiment_id;
    persist();
    renderSteps();
    setHint("experiment-msg", `Using experiment ${exp.name}.`);
    await refreshExperiments();
  } catch (err) {
    setHint("experiment-msg", err.message, true);
  }
});

function renderDataset(summary) {
  state.datasetId = summary.dataset_id;
  persist();
  renderSteps();
  document.getElementById("dataset-summary").innerHTML = `
    <p><strong>${summary.name}</strong> · ${summary.n_train} train / ${summary.n_test} test · labels: ${summary.labels.join(", ")}</p>
    <p class="hint">${summary.description} Primary metric: ${summary.primary_metric}.</p>
  `;
  document.getElementById("dataset-preview").innerHTML = summary.preview
    .map((row) => `<tr><td>${row.split}</td><td>${row.text}</td><td>${row.label}</td></tr>`)
    .join("");
}

async function loadDataset(mode, datasetId) {
  const summary = await api("/api/datasets", {
    method: "POST",
    body: JSON.stringify({ mode, dataset_id: datasetId }),
  });
  renderDataset(summary);
}

document.getElementById("load-bundled").addEventListener("click", () => loadDataset("bundled", "sentiment-v1"));
document.getElementById("load-generated").addEventListener("click", () =>
  loadDataset("generated", "sentiment-generated")
);

document.getElementById("train-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.experimentName) return setHint("train-msg", "Create an experiment first.", true);
  if (!state.datasetId) return setHint("train-msg", "Load a dataset first.", true);
  const btn = document.getElementById("train-btn");
  btn.disabled = true;
  setHint("train-msg", "Running evaluation and logging to MLflow…");
  try {
    const result = await api("/api/runs/train", {
      method: "POST",
      body: JSON.stringify({
        experiment_name: state.experimentName,
        dataset_id: state.datasetId,
        model: document.getElementById("train-model").value,
        prompt_variant: document.getElementById("train-prompt").value,
        temperature: Number(document.getElementById("train-temp").value),
      }),
    });
    state.lastRunId = result.run_id;
    state.experimentId = result.experiment_id;
    persist();
    renderSteps();
    setHint("train-msg", `Logged run ${result.run_id} · ${result.primary_metric}=${fmt(result.metrics.f1_macro)}`);
    document.getElementById("promote-run").value = result.run_id;
    document.getElementById("train-result").innerHTML = `
      <div class="card muted-card">
        <p>Resolved model: <span class="metric">${result.resolved_model}</span></p>
        <p>F1 macro <strong class="metric">${fmt(result.metrics.f1_macro)}</strong> ·
           precision ${fmt(result.metrics.precision_macro)} ·
           recall ${fmt(result.metrics.recall_macro)} ·
           accuracy ${fmt(result.metrics.accuracy)}</p>
        <p><a href="${result.url}" target="_blank" rel="noreferrer">Open this run in MLflow ↗</a></p>
      </div>`;
    await refreshRuns();
  } catch (err) {
    setHint("train-msg", err.message, true);
  } finally {
    btn.disabled = false;
  }
});

async function refreshRuns() {
  if (!state.experimentId) return;
  const data = await api(`/api/runs?experiment_id=${encodeURIComponent(state.experimentId)}`);
  const best = data.runs[0]?.run_id;
  document.getElementById("runs-body").innerHTML = data.runs
    .map((run) => {
      const isBest = run.run_id === best;
      return `<tr>
        <td class="metric ${isBest ? "best" : ""}">${run.run_id.slice(0, 8)}</td>
        <td>${run.model || "—"}</td>
        <td>${run.prompt_variant || "—"}</td>
        <td class="metric ${isBest ? "best" : ""}">${fmt(run.f1_macro)}</td>
        <td class="metric">${fmt(run.metrics.precision_macro)}</td>
        <td class="metric">${fmt(run.metrics.recall_macro)}</td>
        <td class="metric">${fmt(run.accuracy)}</td>
        <td><a href="${run.url}" target="_blank" rel="noreferrer">MLflow</a>
            <button type="button" class="ghost use-run" data-id="${run.run_id}">promote</button></td>
      </tr>`;
    })
    .join("");
  document.querySelectorAll(".use-run").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.lastRunId = btn.dataset.id;
      document.getElementById("promote-run").value = btn.dataset.id;
      persist();
      showPanel("promote");
    });
  });
}

document.getElementById("refresh-runs").addEventListener("click", () => refreshRuns().catch((err) => alert(err.message)));

document.getElementById("promote-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const result = await api("/api/models/promote", {
      method: "POST",
      body: JSON.stringify({
        run_id: document.getElementById("promote-run").value.trim(),
        model_name: document.getElementById("promote-name").value.trim(),
        alias: document.getElementById("promote-alias").value,
      }),
    });
    state.modelName = result.name;
    persist();
    renderSteps();
    setHint("promote-msg", `Version ${result.version} is now @${result.alias}.`);
    await refreshRegistry();
  } catch (err) {
    setHint("promote-msg", err.message, true);
  }
});

async function refreshRegistry() {
  const data = await api("/api/models");
  const champion = data.champion;
  document.getElementById("registry-card").innerHTML = champion
    ? `<p>Registered model <strong>${champion.name}</strong></p>
       <p class="metric">${champion.champion_uri || "no champion alias yet"}</p>
       <p>Aliases: ${JSON.stringify(champion.aliases)}</p>
       <p><a href="${champion.url}" target="_blank" rel="noreferrer">Open Model Registry in MLflow ↗</a></p>`
    : "<p>No registered model yet.</p>";
}

document.getElementById("predict-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const texts = document
    .getElementById("predict-texts")
    .value.split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  try {
    const result = await api("/api/production/predict", {
      method: "POST",
      body: JSON.stringify({ texts, model_name: state.modelName, alias: "champion" }),
    });
    setHint("predict-msg", `Loaded ${result.model_uri}`);
    document.getElementById("predict-result").innerHTML = `
      <div class="card muted-card">
        <p>${result.note}</p>
        <p>Trace experiment: <span class="metric">${result.trace_experiment}</span>
           ${result.trace_id ? `· trace ${result.trace_id}` : ""}</p>
        <p><a href="${result.registry_url}" target="_blank" rel="noreferrer">Registry ↗</a></p>
        <div class="table-wrap"><table><thead><tr><th>Text</th><th>Prediction</th></tr></thead>
        <tbody>${result.predictions.map((row) => `<tr><td>${row.text}</td><td>${row.prediction}</td></tr>`).join("")}</tbody>
        </table></div>
      </div>`;
  } catch (err) {
    setHint("predict-msg", err.message, true);
  }
});

async function boot() {
  renderSteps();
  document.getElementById("experiment-name").value = state.experimentName;
  if (state.lastRunId) document.getElementById("promote-run").value = state.lastRunId;
  showPanel("experiment");
  try {
    await refreshHealth();
    await refreshExperiments();
    if (state.datasetId) {
      const datasets = await api("/api/datasets");
      const current = datasets.datasets.find((item) => item.dataset_id === state.datasetId);
      if (current) renderDataset(current);
    } else {
      const datasets = await api("/api/datasets");
      if (datasets.datasets[0]) renderDataset(datasets.datasets[0]);
    }
    if (state.experimentId) await refreshRuns();
    await refreshRegistry();
  } catch (err) {
    setHint("experiment-msg", err.message, true);
  }
}

boot();
