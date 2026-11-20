# Project Structure: Play Store Review Analyzer (Hugging Face / Gradio Edition)

================================================================================
Directory Hierarchy
================================================================================
.
├── app.py                            # Main Gradio Blocks application & ZeroGPU entry point
│
├── docs/                             # Project documentation & architectural references
│   └── PROJECT_STRUCTURE.md          # Component breakdown & design specification
│
├── Models/                           # Model artifacts cache directory (gitignored via .gitkeep)
│   ├── ensemble_models/              # Multi-label classification models & transformers
│   │   ├── ensemble_model.pkl        # Trained ensemble classifier (Random Forest + XGBoost)
│   │   ├── multilabel_binarizer.pkl  # MultiLabelBinarizer instance for encoding/decoding aspect tags
│   │   └── tfidf.pkl                 # Fitted TF-IDF vectorizer for text feature extraction
│   └── sentiment_model/
│       └── best_model_state.bin      # PyTorch weights for fine-tuned RoBERTa sentiment classifier
│
├── scripts/                          # Utility and maintenance scripts
│   ├── README.md                     # Guide to utility and maintenance scripts
│   ├── score.py                      # Azure Machine Learning scoring and inference endpoint script
│   ├── update_local_model.py         # Utility to cache Hugging Face base models/tokenizers locally
│   └── upload_model_to_hf.py         # Automation script to push model artifacts to Hugging Face Hub
│
├── src/                              # Core application & ML inference package
│   ├── __init__.py                   # Package initialization & public API exports
│   ├── app.py                        # Secondary launcher delegating to root app.py
│   ├── config.py                     # App configuration, constants, HF model identifiers
│   ├── models.py                     # SentimentClassifier architecture, asset loading, inference
│   ├── pipeline.py                   # Pipeline orchestration (Scrape -> Clean -> Predict -> Suggest)
│   ├── preprocessing.py              # Text cleaning (regex, emoji) & English sentence filtering
│   ├── scraper.py                    # Google Play Store review scraping module
│   └── suggestions.py                # Category aggregation & Qwen ZeroGPU suggestion generator
│
├── tests/                            # Automated test suite
│   ├── __init__.py                   # Test package initialization
│   ├── test_config.py                # Configuration and constant validation
│   ├── test_models.py                # Resource resolution and model loading tests
│   ├── test_pipeline.py              # Pipeline sanitization and validation tests
│   ├── test_preprocessing.py         # Text cleaning and preprocessing tests
│   └── test_suggestions.py           # Negative review grouping & Qwen prompting tests
│
├── .repo-push/                       # Repository publishing & multi-platform configuration
│   ├── config.yaml                   # Target configuration for Hugging Face Spaces
│   └── actions.yaml                  # Declarative actions for Space deployment
│
├── requirements.txt                  # Python runtime dependencies for Hugging Face Space
├── README.md                         # Hugging Face Spaces app card and user documentation
├── LICENSE                           # Apache 2.0 open-source license
├── CONTRIBUTING.md                    # Contribution guidelines
├── CODE_OF_CONDUCT.md                 # Contributor covenant code of conduct
└── SECURITY.md                       # Security vulnerability reporting policy
