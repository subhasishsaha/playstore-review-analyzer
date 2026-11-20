---
title: Playstore App Analyzer
emoji: 📱
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.32.0
python_version: '3.12'
app_file: app.py
pinned: false
license: apache-2.0
short_description: Play Store Review Analyzer with ZeroGPU & Qwen.
---

# Play Store Review Analyzer 📱

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/ssaha007/playstore-app-analyzer)
[![ZeroGPU](https://img.shields.io/badge/ZeroGPU-Accelerated-blueviolet)](https://huggingface.co/docs/hub/spaces-zerogpu)
[![Model](https://img.shields.io/badge/LLM-Qwen2.5--1.5B--Instruct-blue)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![Sentiment](https://img.shields.io/badge/RoBERTa-Sentiment%20Classification-orange)](https://huggingface.co/ssaha007/playstore-models)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

An interactive, GPU-accelerated **Gradio** web application deployed on **Hugging Face Spaces** that scrapes, classifies, and synthesizes Google Play Store user feedback into actionable engineering recommendations.

---

## 🌟 Key Capabilities

- **Automated Google Play Scraping**: Fetches up to 300+ reviews for any valid Android App ID (e.g. `com.google.android.gm`, `com.spotify.music`, `com.whatsapp`) with support for *Most Relevant* and *Newest* sort orders.
- **RoBERTa Sentiment Classification**: Binary sentiment analysis (Positive / Negative) powered by a fine-tuned `FacebookAI/roberta-base` PyTorch model.
- **Multi-Label Aspect Categorization**: TF-IDF feature extraction combined with an ensemble classifier (**Random Forest + XGBoost**) that tags reviews with relevant operational categories (*Bugs*, *UI/UX*, *Login Problems*, *Performance*, *Features*).
- **ZeroGPU AI Developer Recommendations**: Leverages [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) under dynamic Hugging Face **ZeroGPU** (`@spaces.GPU(duration=60)`) to synthesize concrete, high-priority bug fixes and architectural recommendations for developers.
- **Interactive Visual Analytics**:
  - **Aspect Distribution**: Plotly horizontal bar chart of the most frequent review categories.
  - **Sentiment Distribution**: Plotly donut chart highlighting positive vs. negative review percentages.
  - **Sentiment by Aspect Breakdown**: Grouped bar chart comparing positive and negative feedback across each topic.
  - **Negative Word Cloud**: Visual representation of the most prominent complaint keywords.
- **Searchable Reviews Table**: Interactive data table displaying user names, scores, timestamps, app versions, predicted sentiments, and aspect tags.

---

## 🏗️ Repository Architecture

This branch (`huggingface`) is optimized specifically for **Hugging Face Spaces (Gradio SDK)**:

```text
playstore-review-analyzer/
├── app.py                  # Primary Gradio Blocks application & ZeroGPU launcher
├── requirements.txt        # Hugging Face Spaces Python dependencies
├── README.md               # Space documentation with Hugging Face YAML frontmatter
├── .repo-push/             # Remote configuration & target actions
│   ├── config.yaml         # Hugging Face Space target configuration
│   └── actions.yaml        # Space synchronization declarative action
├── src/                    # Machine learning & inference pipeline
│   ├── app.py              # Secondary launcher delegating to root app.py
│   ├── config.py           # Configuration constants & model identifiers
│   ├── models.py           # RoBERTa & Ensemble model bundle loading
│   ├── pipeline.py         # Scraping, classification, and analytics orchestration
│   ├── preprocessing.py    # Text cleaning, regex, emoji removal, and language detection
│   ├── scraper.py          # Google Play Store review extraction
│   └── suggestions.py      # ZeroGPU Qwen2.5-1.5B-Instruct prompting engine
├── Models/                 # Model artifacts cache directory (.gitkeep)
├── scripts/                # Utility and synchronization scripts
└── tests/                  # Automated unit test suite
```

---

## ⚡ ZeroGPU Acceleration

In Hugging Face Spaces free tier, Docker images are not supported. This space utilizes the native **Gradio SDK** paired with **ZeroGPU**, which dynamically grants Nvidia A100 / L4 GPU compute on demand:

```python
import spaces
from src.suggestions import get_qwen_model

@spaces.GPU(duration=60)
def process_reviews(app_id: str, sort_order: str, count: int):
    # Runs RoBERTa & Qwen2.5-1.5B-Instruct on allocated GPU
    ...
```

When running locally or on non-CUDA environments, a graceful fallback mock decorator is used automatically so developers can test without requiring GPU hardware.

---

## 🚀 Quick Start (Local Development)

### 1. Clone the Repository
```bash
git clone https://github.com/subhasishsaha/playstore-review-analyzer.git
cd playstore-review-analyzer
git checkout huggingface
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch Gradio Web App
```bash
python app.py
```

- Open **`http://localhost:7860`** in your browser to interact with the dashboard.
- Enter an App ID like `com.google.android.gm` and click **🚀 Fetch & Analyze Reviews**.

---

## 🧪 Testing

Run the automated test suite locally:

```bash
python -m unittest discover tests
```

---

## 📄 License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
