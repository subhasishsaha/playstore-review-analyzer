# Play Store Review Analyzer 📱

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-53%20Passed-brightgreen)](tests/)

A modern full-stack web application and REST API for scraping, analyzing, and synthesizing user feedback from the Google Play Store using Deep Learning (RoBERTa), Ensemble Machine Learning, and local LLM reasoning (Ollama).

---

## 🌟 Key Features

- **Binary Sentiment Classification**: Deep learning inference powered by fine-tuned **RoBERTa** classifying user feedback into Positive and Negative sentiments.
- **Multi-Label Aspect Categorization**: TF-IDF feature extraction combined with an ensemble classifier (**Random Forest + XGBoost**) automatically tagging reviews with actionable aspect categories (*Bugs*, *UI/UX*, *Login Problems*, *Performance*, *Features*).
- **Local AI Developer Suggestions & Dynamic Model Auto-Pull**: Synthesizes structured, technical engineering recommendations for negative review clusters using local LLMs (**Qwen 3.5**, **LLaMA 3.2**, **Mistral**, etc.). If a requested model is missing, the backend automatically pulls it on-demand via the Ollama client.
- **Modern Responsive Dashboard**: Pure HTML5/CSS3/JavaScript (ES6+) UI featuring glassmorphism styling, live KPI metric cards, searchable/filterable review tables, and interactive **Chart.js** data visualizations.
- **FastAPI REST API**: High-performance asynchronous API endpoints (`/api/analyze`, `/api/models`, `/api/health`, `/docs`) with automated static frontend serving, CORS support, and OpenAPI documentation.
- **Zero-Dependency Node Server**: Optional lightweight Node.js server on port 3000 for standalone frontend development with transparent reverse-proxying of `/api/*` endpoints to FastAPI on port 8000.

---

## 🏗️ System Architecture

```text
Play Store Review Analyzer
├── frontend/             # Python-independent Web UI (HTML5, CSS3, JS, Chart.js)
│   ├── index.html        # Responsive dashboard interface
│   ├── styles.css        # Modern glassmorphism CSS design system
│   ├── app.js            # Client-side analytics, state & Chart.js charts
│   ├── server.js         # Lightweight Node.js server with API proxying
│   ├── favicon.svg       # SVG application icon
│   └── package.json      # Node scripts (`npm start`)
├── src/                  # Modular backend & ML pipeline
│   ├── api.py            # FastAPI REST API & static asset mounting
│   ├── app.py            # Application entrypoint (Uvicorn launcher)
│   ├── config.py         # Configuration constants & environment resolution
│   ├── models.py         # PyTorch RoBERTa & Ensemble model bundle loading
│   ├── pipeline.py       # Orchestration (Scrape -> Clean -> Predict -> Suggest)
│   ├── preprocessing.py  # Regex cleaning, emoji removal, English filtering
│   ├── scraper.py        # Play Store review fetching via google_play_scraper
│   └── suggestions.py    # Negative review grouping & Ollama LLM prompting
├── Models/               # ML & Deep Learning artifacts (excluded from git via .gitkeep)
│   ├── ensemble_models/  # Trained ensemble classifier, MultiLabelBinarizer & TF-IDF
│   └── sentiment_model/  # Fine-tuned RoBERTa PyTorch weights (best_model_state.bin)
├── scripts/              # Operational, caching & cloud utility scripts
│   ├── README.md         # Guide to utility scripts
│   ├── score.py          # Azure ML scoring endpoint script
│   ├── update_local_model.py # RoBERTa model/tokenizer caching utility
│   └── upload_model_to_hf.py # Hugging Face Model Hub upload automation
├── Dataset/              # Historical annotated benchmark dataset (gpreviews_2.csv)
├── Notebooks/            # Research, training & exploration Jupyter notebooks
├── Review/               # Academic conference paper, presentation & SOP deliverables
├── docs/                 # Architectural documentation & project specifications
│   └── PROJECT_STRUCTURE.md # Comprehensive directory & component breakdown
├── tests/                # Automated test suite (53 unit & integration tests)
├── .env.example          # Sanitized template for environment variables
├── .gitattributes        # Git line ending configuration
├── .gitignore            # Git ignore rules for virtual environments, caches, models
├── CODE_OF_CONDUCT.md    # Contributor Covenant community standards
├── CONTRIBUTING.md       # Contribution guidelines & testing standards
├── LICENSE               # Apache 2.0 license
├── README.md             # Project documentation & quickstart guide
├── requirements.txt      # Python dependencies (FastAPI, PyTorch, Ollama, etc.)
└── SECURITY.md           # Security policy and vulnerability disclosure protocol
```

For an exhaustive breakdown of individual modules, see [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

---

## 🚀 Quick Start

### 1. Docker Compose Setup (Recommended)

Run the full stack—FastAPI backend, pre-warmed ML inference, pure JS dashboard, and Ollama with automated model pre-pulling—in isolated containers:

```bash
# 1. Clone the repository
git clone https://github.com/subhasishsaha/playstore-review-analyzer.git
cd playstore-review-analyzer

# 2. Launch FastAPI web app and Ollama with persistent model caching
docker compose up -d

# 3. View startup and initialization logs
docker compose logs -f
```

- Open **`http://localhost:8000`** in your browser to access the dashboard.
- Models are persisted in the `playstore_ollama_models` Docker volume across restarts.

---

### 2. Local Python Setup

```bash
# 1. Clone the repository
git clone https://github.com/subhasishsaha/playstore-review-analyzer.git
cd playstore-review-analyzer

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure environment variables
copy .env.example .env  # On Linux/macOS: cp .env.example .env

# 5. Launch the application
python src/app.py
```

Open **`http://localhost:8000`** in your browser to access the dashboard.  
Visit **`http://localhost:8000/docs`** for interactive OpenAPI / Swagger API documentation.

---

### 3. Optional: Run Frontend via Node.js

You can also run the frontend independently using the zero-dependency Node.js server:

```bash
cd frontend
npm start
```

The Node.js server runs at `http://localhost:3000` and automatically proxies all `/api/*` requests to FastAPI on port 8000.

---

## 🔌 REST API Reference

### 1. Health Check
`GET /api/health`

```bash
curl -X GET "http://localhost:8000/api/health"
```

**Response:**
```json
{
  "status": "online",
  "service": "Play Store Review Analyzer API",
  "ollama_model": "qwen3.5:0.8b"
}
```

---

### 2. List Installed Ollama Models
`GET /api/models`

```bash
curl -X GET "http://localhost:8000/api/models"
```

**Response:**
```json
{
  "default_model": "qwen3.5:0.8b",
  "installed_models": [
    "qwen3.5:0.8b",
    "llama3.2:1b"
  ]
}
```

---

### 3. Analyze App Reviews
`POST /api/analyze`

Scrapes live reviews, executes sentiment classification and aspect tagging, and generates local LLM developer suggestions.

**Request Payload:**
```json
{
  "app_id": "com.spotify.music",
  "count": 100,
  "sort_order": "Most Relevant",
  "ollama_model": "qwen3.5:0.8b"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "app_id": "com.spotify.music",
    "count": 100,
    "sort_order": "Most Relevant",
    "ollama_model": "qwen3.5:0.8b"
  }'
```

**Response Format:**
```json
{
  "status": "success",
  "app_id": "com.spotify.music",
  "total": 100,
  "summary": {
    "total": 100,
    "positive_count": 68,
    "negative_count": 32,
    "positive_percentage": 68.0,
    "negative_percentage": 32.0,
    "labels_distribution": {
      "bugs": 18,
      "ui/ux": 12,
      "performance": 8,
      "features": 5
    }
  },
  "suggestions_markdown": "### Suggestions for: **Bugs**\n1. Fix intermittent playback buffering...\n2. Resolve audio pause glitches...",
  "reviews": [
    {
      "reviewId": "gp:AOqpTOE...",
      "userName": "Alex",
      "content": "Great music selection but crashes after update.",
      "score": 2,
      "sentiment": "Negative",
      "labels": ["bugs", "performance"],
      "at": "2026-01-15"
    }
  ]
}
```

---

## ⚙️ Configuration & Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `HOST` | `127.0.0.1` | Network bind address for the API server |
| `PORT` | `8000` | Port for the FastAPI server |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endpoint URL of the Ollama inference server |
| `OLLAMA_MODEL` | `qwen3.5:0.8b` | Default LLM model for suggestion synthesis |
| `DEFAULT_REVIEW_COUNT` | `100` | Default review count fetched per query (10 - 500) |

---

## 🛠️ Utility & Maintenance Scripts

All operational and maintenance scripts reside in [`scripts/`](scripts/):

- [`scripts/update_local_model.py`](scripts/update_local_model.py): Downloads and caches the base RoBERTa model (`FacebookAI/roberta-base`) and tokenizer locally.
- [`scripts/upload_model_to_hf.py`](scripts/upload_model_to_hf.py): Automates uploading model weights and preprocessors to the Hugging Face Hub (`ssaha007/playstore-models`).
- [`scripts/score.py`](scripts/score.py): Production scoring script for deploying the sentiment model onto Azure Machine Learning endpoints.

See [scripts/README.md](scripts/README.md) for execution instructions.

---

## 🧪 Testing

The repository includes a comprehensive 53-test automated unit and integration suite covering API endpoints, model listings, preprocessing edge cases, Ollama model auto-pulling verification, and inference output alignment.

Run tests using the project virtual environment:
```bash
python -m unittest discover -s tests
```

---

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
