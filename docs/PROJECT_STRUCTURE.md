# Project Structure: Play Store Review Analyzer

================================================================================
Directory Hierarchy
================================================================================
.
├── Dataset/
│   └── gpreviews_2.csv                   # Annotated review dataset (reviewId, text, score, sentiment, labels)
│
├── docs/                                 # Project documentation & architectural references
│   └── PROJECT_STRUCTURE.md              # Detailed component breakdown & design specification
│
├── frontend/                             # Python-independent modern JS/HTML/CSS Web UI
│   ├── index.html                        # Responsive dashboard HTML (Data table, Suggestions, Charts)
│   ├── styles.css                        # Modern CSS design system (glassmorphism, responsive grid)
│   ├── app.js                            # Client-side state, Chart.js visualizations, API integration
│   ├── package.json                      # Node.js project manifest & scripts (`npm start`, `npm run dev`)
│   ├── favicon.svg                       # SVG application icon
│   └── server.js                         # Zero-dependency Node.js server (serves UI & proxies `/api/*`)
│
├── Models/
│   ├── ensemble_models/                  # Multi-label classification models & transformers
│   │   ├── ensemble_model.pkl            # Trained ensemble classifier (Random Forest + XGBoost)
│   │   ├── multilabel_binarizer.pkl      # MultiLabelBinarizer instance for encoding/decoding aspect tags
│   │   └── tfidf.pkl                     # Fitted TF-IDF vectorizer for text feature extraction
│   └── sentiment_model/
│       └── best_model_state.bin          # PyTorch weights for fine-tuned RoBERTa sentiment classifier
│
├── Notebooks/                            # Research, prototyping, and training workflows
│   ├── binary-sentiment-classification-using-roberta.ipynb
│   │                                     # Fine-tuning RoBERTa for binary sentiment classification
│   ├── initial-multi-labelling-using-bart-large.ipynb
│   │                                     # Zero-shot classification & multi-labeling via BART-Large-MNLI
│   ├── multi-label-classification-using-ensemble-approach.ipynb
│   │                                     # Training & evaluation of multi-label ensemble ML models
│   └── suggestion-using-gemini.ipynb     # Prompt engineering & suggestion generation using Gemini API
│
├── Review/                               # Academic & presentation deliverables (Study Oriented Project)
│   ├── references-sop.bib                # BibTeX references for literature review & citations
│   ├── SOP-Final Review.pptx             # Final project presentation slide deck
│   ├── sop-pipeline.png                  # System architecture and pipeline diagram
│   └── study-oriented-project-conference.pdf # Research conference paper documenting methodology & results
│
├── scripts/                              # Utility and deployment scripts
│   ├── README.md                         # Guide to utility and maintenance scripts
│   ├── score.py                          # Azure Machine Learning scoring and inference endpoint script
│   ├── update_local_model.py             # Utility to cache Hugging Face base models/tokenizers locally
│   └── upload_model_to_hf.py             # Automation script to push model artifacts to Hugging Face Hub
│
├── src/                                  # Modular core application & API package
│   ├── __init__.py                       # Package initialization & public API exports
│   ├── api.py                            # FastAPI backend REST API (`/api/analyze`, `/api/health`, `/api/models`)
│   ├── app.py                            # Application entry point (launches Uvicorn server on port 8000)
│   ├── config.py                         # App configuration, constants, environment variables
│   ├── models.py                         # SentimentClassifier architecture, asset loading, inference
│   ├── pipeline.py                       # Pipeline orchestration (Scrape -> Clean -> Predict -> Suggest)
│   ├── preprocessing.py                  # Text cleaning (regex, emoji) & English sentence filtering
│   ├── scraper.py                        # Google Play Store review scraping module
│   └── suggestions.py                    # Category aggregation & Qwen (Ollama) recommendation generator
│
├── tests/                                # Automated unit & integration test suite
│   ├── __init__.py                       # Test package initializer
│   ├── test_api.py                       # FastAPI endpoint & static file mounting integration tests
│   ├── test_config.py                    # Configuration constants & default values tests
│   ├── test_models.py                    # Model bundle path resolution & architecture unit tests
│   ├── test_pipeline.py                  # Pipeline data sanitization & input validation tests
│   ├── test_preprocessing.py             # Regex cleaning, emoji removal, and language detection tests
│   └── test_suggestions.py               # Negative review grouping & suggestion fallback tests
│
├── .env.example                          # Sanitized template for environment variables
├── .gitattributes                        # Git attributes configuration
├── .gitignore                            # Git ignore rules for virtual environments, caches, models
├── CODE_OF_CONDUCT.md                    # Contributor Covenant code of conduct
├── CONTRIBUTING.md                       # Contribution guidelines & branching standards
├── LICENSE                               # Project licensing information (Apache 2.0)
├── README.md                             # Project documentation & quickstart guide
├── requirements.txt                      # Project Python dependencies (FastAPI, PyTorch, Ollama, etc.)
└── SECURITY.md                           # Security policy and vulnerability disclosure protocol

================================================================================
Component Breakdown & Roles
================================================================================

1. Python-Independent Frontend (frontend/)
   - Technology Stack: HTML5, CSS3, modern JavaScript (ES6+), Chart.js, Marked.js.
   - Core Features:
     * Query Controls: Package App ID input with quick-select chips, sorting order, and review count slider.
     * KPI Metrics: Live metric cards displaying Total Reviews, Positive/Negative % rates, and dominant aspect.
     * Interactive Data View: Responsive table with sentiment pill badges, category tags, search, and pagination.
     * AI Developer Action Items: Categorized suggestion cards synthesized by local Qwen 0.8B with one-click copy.
     * Dynamic Visualizations: Chart.js powered horizontal aspect distribution, sentiment donut, and stacked category breakdown.
   - Node.js Support:
     * `frontend/server.js`: Zero-dependency Node.js HTTP server running on port 3000 with automatic proxying of `/api/*` requests to port 8000.
     * `frontend/package.json`: Run via `npm start` or `npm run dev`.

2. Backend REST API (src/api.py & src/app.py)
   - Framework: FastAPI + Uvicorn.
   - Endpoints:
     * `POST /api/analyze`: Accepts `{ app_id, sort_order, count, ollama_model }` and returns complete structured analysis JSON.
     * `GET /api/health`: Health status and default Ollama model metadata.
     * `GET /api/models`: Returns list of locally installed Ollama models and default model.
     * `GET /`: Serves the static JS/HTML/CSS frontend.
     * `GET /docs`: Interactive OpenAPI / Swagger UI.

3. Machine Learning & Inference Pipeline (src/)
   - Scraper (`src/scraper.py`): Fetches live Play Store reviews via `google_play_scraper`.
   - Preprocessing (`src/preprocessing.py`): Emoji, URL, regex cleaning, and sentence-level language detection.
   - Models (`src/models.py`): RoBERTa PyTorch sentiment classification and TF-IDF + Ensemble multi-label categorization.
   - Local LLM Suggestions (`src/suggestions.py`): Dynamic Ollama model selection, on-demand automatic model pulling (`client.pull()`), and recommendation generation.

4. Test Suite (tests/)
   - Framework: Standard library `unittest` and FastAPI `TestClient`.
   - Coverage: 53 tests covering API endpoints, model listings, dynamic Ollama model parameter propagation, automatic model pulling verification, reasoning model fallback & think tag stripping, dict and string response format parsing, tag normalization, preprocessing edge cases, JSON data sanitization, model array length preservation and alignment, and suggestion generation fallback/grouping.
