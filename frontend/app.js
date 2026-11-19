/**
 * Play Store Review Analyzer Frontend Application Logic
 */

// API Base URL Detection: Use relative paths when served over HTTP/HTTPS (works on any port or reverse proxy), fallback to localhost:8000 for file:// or dev
const API_BASE = window.location.protocol.startsWith("http")
  ? ""
  : "http://localhost:8000";

// Global App State
const state = {
  rawData: null,
  filteredReviews: [],
  currentPage: 1,
  pageSize: 15,
  charts: {
    aspects: null,
    sentiment: null,
    stacked: null,
  },
};

// DOM Elements
const form = document.getElementById("analyze-form");
const appIdInput = document.getElementById("app-id");
const sortOrderSelect = document.getElementById("sort-order");
const reviewCountInput = document.getElementById("review-count");
const countDisplay = document.getElementById("count-display");
const ollamaModelInput = document.getElementById("ollama-model");
const modelsList = document.getElementById("models-list");
const submitBtn = document.getElementById("submit-btn");
const spinner = document.getElementById("loading-spinner");
const alertBox = document.getElementById("alert-box");
const alertMsg = document.getElementById("alert-message");
const alertClose = document.getElementById("alert-close");
const resultsContainer = document.getElementById("results-container");

// KPI Elements
const kpiTotal = document.getElementById("kpi-total");
const kpiPositive = document.getElementById("kpi-positive");
const kpiPositiveCount = document.getElementById("kpi-positive-count");
const kpiNegative = document.getElementById("kpi-negative");
const kpiNegativeCount = document.getElementById("kpi-negative-count");
const kpiTopAspect = document.getElementById("kpi-top-aspect");

// Table & Filter Elements
const tableBody = document.getElementById("reviews-table-body");
const tableSearch = document.getElementById("table-search");
const filterSentiment = document.getElementById("filter-sentiment");
const filterAspect = document.getElementById("filter-aspect");
const tableCountLabel = document.getElementById("table-count-label");
const prevPageBtn = document.getElementById("prev-page-btn");
const nextPageBtn = document.getElementById("next-page-btn");
const paginationInfo = document.getElementById("pagination-info");

// Suggestions & Tab Elements
const suggestionsContent = document.getElementById("suggestions-content");
const copySuggestionsBtn = document.getElementById("copy-suggestions-btn");
const tabButtons = document.querySelectorAll(".tab-btn");
const tabPanes = document.querySelectorAll(".tab-pane");
const backendStatus = document.getElementById("backend-status");

function normalizeModelTag(tag) {
  if (!tag) return "";
  let t = String(tag).trim().toLowerCase();
  if (t.includes("/")) t = t.split("/").pop();
  return t.includes(":") ? t : `${t}:latest`;
}

function updateModelBadge(modelName) {
  const badge = document.getElementById("model-status") || document.querySelector(".header-badges .status-model");
  if (badge && modelName) {
    badge.innerHTML = `<span class="status-dot"></span> ${escapeHtml(modelName)} (Ollama)`;
  }
}

if (ollamaModelInput) {
  ollamaModelInput.addEventListener("input", () => {
    ollamaModelInput.dataset.userEdited = "true";
    const val = ollamaModelInput.value.trim();
    if (val) {
      updateModelBadge(val);
    }
  });
}

async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (res.ok) {
      const data = await res.json();
      backendStatus.innerHTML = `<span class="status-dot"></span> API Online`;
      backendStatus.classList.remove("status-model");
      backendStatus.classList.add("status-ready");
    }
  } catch (err) {
    backendStatus.innerHTML = `<span class="status-dot" style="background:#f59e0b;"></span> Backend Offline (Start API on :8000)`;
  }

  // Load available models from Ollama
  try {
    const res = await fetch(`${API_BASE}/api/models`);
    if (res.ok) {
      const data = await res.json();
      if (data.default_model) {
        if (ollamaModelInput && !ollamaModelInput.dataset.userEdited) {
          ollamaModelInput.value = data.default_model;
        }
        updateModelBadge(data.default_model);
      }
      if (modelsList && Array.isArray(data.installed_models)) {
        modelsList.innerHTML = "";
        const installedSet = new Set(data.installed_models.map(normalizeModelTag));
        const combined = Array.from(new Set([
          data.default_model,
          ...data.installed_models,
          "qwen3.5:0.8b",
          "qwen2.5:0.5b",
          "qwen2.5-coder:1.5b",
          "llama3.2:1b",
          "mistral:7b"
        ]));
        combined.forEach(m => {
          if (!m) return;
          const opt = document.createElement("option");
          opt.value = m;
          const isInstalled = installedSet.has(normalizeModelTag(m));
          opt.label = isInstalled ? `${m} (Installed)` : `${m} (Auto-pull on demand)`;
          opt.textContent = opt.label;
          modelsList.appendChild(opt);
        });
      }
    }
  } catch (err) {
    // Silently use defaults if offline
  }
}


// -------------------------------------------------------------
// UI Event Listeners
// -------------------------------------------------------------
reviewCountInput.addEventListener("input", (e) => {
  countDisplay.textContent = e.target.value;
});

// Quick chips click
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    appIdInput.value = chip.dataset.appid;
    appIdInput.focus();
  });
});

// Alert close
alertClose.addEventListener("click", () => {
  alertBox.classList.add("hidden");
});

function showAlert(message) {
  alertMsg.textContent = message;
  alertBox.classList.remove("hidden");
  alertBox.scrollIntoView({ behavior: "smooth", block: "center" });
}

function hideAlert() {
  alertBox.classList.add("hidden");
}

// Tabs switching
tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const targetTab = btn.dataset.tab;
    tabButtons.forEach((b) => b.classList.remove("active"));
    tabPanes.forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(targetTab).classList.add("active");

    // Trigger Chart.js resize if switching to visualizations tab
    if (targetTab === "tab-visualizations") {
      Object.values(state.charts).forEach((chart) => {
        if (chart) chart.resize();
      });
    }
  });
});

// Copy suggestions
copySuggestionsBtn.addEventListener("click", () => {
  if (!state.rawData || !state.rawData.suggestions_markdown) return;
  navigator.clipboard.writeText(state.rawData.suggestions_markdown).then(() => {
    const originalText = copySuggestionsBtn.textContent;
    copySuggestionsBtn.textContent = "✅ Copied!";
    setTimeout(() => {
      copySuggestionsBtn.textContent = originalText;
    }, 2000);
  });
});

// -------------------------------------------------------------
// Form Submit & Analysis
// -------------------------------------------------------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideAlert();

  const appId = appIdInput.value.trim();
  const sortOrder = sortOrderSelect.value;
  const count = parseInt(reviewCountInput.value, 10);
  const ollamaModel = ollamaModelInput ? ollamaModelInput.value.trim() : null;

  if (!appId) {
    showAlert("Please enter an App ID.");
    return;
  }

  // Set loading state
  submitBtn.disabled = true;
  spinner.classList.remove("hidden");
  submitBtn.querySelector(".btn-text").textContent = "Analyzing reviews & running AI...";

  try {
    const payload = {
      app_id: appId,
      sort_order: sortOrder,
      count: count,
    };
    if (ollamaModel) {
      payload.ollama_model = ollamaModel;
    }

    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server returned status ${res.status}`);
    }

    const data = await res.json();
    state.rawData = data;
    renderResults(data);
  } catch (err) {
    showAlert(`Analysis failed: ${err.message}`);
  } finally {
    submitBtn.disabled = false;
    spinner.classList.add("hidden");
    submitBtn.querySelector(".btn-text").textContent = "🚀 Fetch & Analyze Reviews";
  }
});

// -------------------------------------------------------------
// Render Results
// -------------------------------------------------------------
function renderResults(data) {
  const summary = data.summary || {};
  const total = summary.total || data.reviews.length || 0;
  const posCount = summary.positive_count || 0;
  const negCount = summary.negative_count || 0;
  const posPct = summary.positive_percentage || 0;
  const negPct = summary.negative_percentage || 0;

  // Render KPI Metrics
  kpiTotal.textContent = total.toLocaleString();
  kpiPositive.textContent = `${posPct}%`;
  kpiPositiveCount.textContent = `${posCount} positive reviews`;
  kpiNegative.textContent = `${negPct}%`;
  kpiNegativeCount.textContent = `${negCount} negative reviews`;

  const aspectEntries = Object.entries(summary.labels_distribution || {});
  kpiTopAspect.textContent = aspectEntries.length > 0 ? aspectEntries[0][0] : "None";

  // Populate Aspect Filter Dropdown
  filterAspect.innerHTML = '<option value="ALL">All Categories</option>';
  aspectEntries.forEach(([aspect]) => {
    const opt = document.createElement("option");
    opt.value = aspect;
    opt.textContent = aspect;
    filterAspect.appendChild(opt);
  });

  // Render AI Suggestions Markdown
  const modelUsed = data.model_used || (ollamaModelInput ? ollamaModelInput.value.trim() : "") || "Ollama";
  updateModelBadge(modelUsed);
  const suggestionsSubtitle = document.querySelector(".suggestions-header .section-subtitle");
  if (suggestionsSubtitle) {
    suggestionsSubtitle.textContent = `Actionable recommendations synthesized by ${modelUsed} based on negative feedback clusters.`;
  }

  if (data.suggestions_markdown && typeof marked !== "undefined") {
    suggestionsContent.innerHTML = marked.parse(data.suggestions_markdown);
  } else {
    suggestionsContent.innerHTML = "<p>No suggestions generated.</p>";
  }

  // Populate Reviews Table
  state.filteredReviews = [...data.reviews];
  state.currentPage = 1;
  renderTable();

  // Render Charts
  renderCharts(summary);

  // Unhide results container
  resultsContainer.classList.remove("hidden");
  resultsContainer.scrollIntoView({ behavior: "smooth" });
}

// -------------------------------------------------------------
// Table Filtering & Pagination
// -------------------------------------------------------------
function applyTableFilters() {
  if (!state.rawData) return;

  const searchQuery = tableSearch.value.trim().toLowerCase();
  const sentimentFilter = filterSentiment.value;
  const aspectFilter = filterAspect.value;

  state.filteredReviews = state.rawData.reviews.filter((r) => {
    // 1. Sentiment filter
    if (sentimentFilter !== "ALL" && r.sentiment !== sentimentFilter) {
      return false;
    }

    // 2. Aspect filter
    if (aspectFilter !== "ALL") {
      const labels = (r.labels || []).map((l) => String(l).toLowerCase());
      if (!labels.includes(aspectFilter.toLowerCase())) {
        return false;
      }
    }

    // 3. Search query
    if (searchQuery) {
      const content = (r.content || "").toLowerCase();
      const user = (r.userName || "").toLowerCase();
      if (!content.includes(searchQuery) && !user.includes(searchQuery)) {
        return false;
      }
    }

    return true;
  });

  state.currentPage = 1;
  renderTable();
}

tableSearch.addEventListener("input", applyTableFilters);
filterSentiment.addEventListener("change", applyTableFilters);
filterAspect.addEventListener("change", applyTableFilters);

function renderTable() {
  const list = state.filteredReviews;
  const total = list.length;
  const totalPages = Math.max(1, Math.ceil(total / state.pageSize));

  if (state.currentPage > totalPages) state.currentPage = totalPages;
  if (state.currentPage < 1) state.currentPage = 1;

  const startIdx = (state.currentPage - 1) * state.pageSize;
  const endIdx = Math.min(startIdx + state.pageSize, total);
  const pageItems = list.slice(startIdx, endIdx);

  tableCountLabel.textContent = `Showing ${pageItems.length > 0 ? startIdx + 1 : 0}-${endIdx} of ${total}`;
  paginationInfo.textContent = `Page ${state.currentPage} of ${totalPages}`;

  prevPageBtn.disabled = state.currentPage <= 1;
  nextPageBtn.disabled = state.currentPage >= totalPages;

  if (pageItems.length === 0) {
    tableBody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding: 2rem; color: #64748b;">No matching reviews found.</td></tr>`;
    return;
  }

  tableBody.innerHTML = pageItems
    .map((r) => {
      const isPositive = r.sentiment === "Positive";
      const sentimentBadge = `<span class="badge ${isPositive ? "badge-positive" : "badge-negative"}">${r.sentiment || "Unknown"}</span>`;
      
      const aspectTags = (r.labels || [])
        .filter((l) => l && String(l).toLowerCase() !== "unknown")
        .map((l) => `<span class="tag-pill">${escapeHtml(String(l).trim())}</span>`)
        .join("");

      const formattedDate = r.at ? String(r.at).split(/[ T]/)[0] : "N/A";
      const ver = r.appVersion ? String(r.appVersion).trim() : "";
      const appVer = ver ? (ver.toLowerCase().startsWith("v") ? ver : `v${ver}`) : "vUnknown";

      return `
        <tr>
          <td>
            <div class="user-name">${escapeHtml(r.userName || "Anonymous")}</div>
            <div style="font-size:0.75rem; color:#94a3b8;">${renderStars(r.score)}</div>
          </td>
          <td>
            <div class="review-content">${escapeHtml(r.content || "")}</div>
          </td>
          <td>${sentimentBadge}</td>
          <td><div class="aspect-tags">${aspectTags || "<span style='color:#94a3b8;'>-</span>"}</div></td>
          <td>
            <div style="font-size:0.8rem; font-weight:600; color:#475569;">${formattedDate}</div>
            <div style="font-size:0.75rem; color:#94a3b8;">${escapeHtml(appVer)}</div>
          </td>
        </tr>
      `;
    })
    .join("");
}

prevPageBtn.addEventListener("click", () => {
  if (state.currentPage > 1) {
    state.currentPage--;
    renderTable();
  }
});

nextPageBtn.addEventListener("click", () => {
  const totalPages = Math.ceil(state.filteredReviews.length / state.pageSize);
  if (state.currentPage < totalPages) {
    state.currentPage++;
    renderTable();
  }
});

function renderStars(score) {
  const num = parseInt(score, 10) || 0;
  return "★".repeat(Math.min(5, Math.max(0, num))) + "☆".repeat(Math.max(0, 5 - num));
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// -------------------------------------------------------------
// Interactive Chart.js Visualizations
// -------------------------------------------------------------
function renderCharts(summary) {
  if (typeof Chart === "undefined") return;

  // 1. Aspect Category Distribution Chart (Horizontal Bar)
  const aspectDist = summary.labels_distribution || {};
  const aspectLabels = Object.keys(aspectDist);
  const aspectCounts = Object.values(aspectDist);

  if (state.charts.aspects) state.charts.aspects.destroy();
  const ctxAspects = document.getElementById("chart-aspects").getContext("2d");
  state.charts.aspects = new Chart(ctxAspects, {
    type: "bar",
    data: {
      labels: aspectLabels,
      datasets: [
        {
          label: "Reviews Mentioning Category",
          data: aspectCounts,
          backgroundColor: "#4f46e5",
          borderRadius: 6,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1e1b4b",
          titleFont: { family: "Plus Jakarta Sans", weight: "bold" },
        },
      },
      scales: {
        x: { beginAtZero: true, grid: { color: "#f1f5f9" } },
        y: { grid: { display: false } },
      },
    },
  });

  // 2. Overall Sentiment Breakdown (Donut Chart)
  if (state.charts.sentiment) state.charts.sentiment.destroy();
  const ctxSentiment = document.getElementById("chart-sentiment").getContext("2d");
  state.charts.sentiment = new Chart(ctxSentiment, {
    type: "doughnut",
    data: {
      labels: ["Positive", "Negative"],
      datasets: [
        {
          data: [summary.positive_count || 0, summary.negative_count || 0],
          backgroundColor: ["#10b981", "#ef4444"],
          borderWidth: 2,
          borderColor: "#ffffff",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { font: { family: "Plus Jakarta Sans", weight: "600" } } },
      },
      cutout: "68%",
    },
  });

  // 3. Sentiment per Category (Stacked Bar Chart)
  const sentimentByLabel = summary.sentiment_by_label || {};
  const stackedCategories = Object.keys(sentimentByLabel);
  const posPerCategory = stackedCategories.map((c) => sentimentByLabel[c]?.Positive || 0);
  const negPerCategory = stackedCategories.map((c) => sentimentByLabel[c]?.Negative || 0);

  if (state.charts.stacked) state.charts.stacked.destroy();
  const ctxStacked = document.getElementById("chart-stacked").getContext("2d");
  state.charts.stacked = new Chart(ctxStacked, {
    type: "bar",
    data: {
      labels: stackedCategories,
      datasets: [
        {
          label: "Positive Reviews",
          data: posPerCategory,
          backgroundColor: "#10b981",
          borderRadius: 4,
        },
        {
          label: "Negative Reviews",
          data: negPerCategory,
          backgroundColor: "#ef4444",
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { stacked: true, grid: { display: false } },
        y: { stacked: true, beginAtZero: true, grid: { color: "#f1f5f9" } },
      },
      plugins: {
        legend: { position: "top", labels: { font: { family: "Plus Jakarta Sans", weight: "600" } } },
      },
    },
  });
}

// Initial health check on load
checkBackendHealth();
