# Utility & Deployment Scripts

This directory contains operational, maintenance, and deployment scripts for the Play Store Review Analyzer.

---

## 📜 Available Scripts

### 1. `update_local_model.py`
Downloads and caches the base RoBERTa transformer model (`FacebookAI/roberta-base`) and tokenizer into the local `Models/` directory (`Models/roberta_base` and `Models/roberta_tokenizer`).

**Usage:**
```bash
python scripts/update_local_model.py
```

---

### 2. `upload_model_to_hf.py`
Automates uploading the local `Models/` folder artifacts to the Hugging Face Model Hub repository (`ssaha007/playstore-models`).

**Usage:**
```bash
python scripts/upload_model_to_hf.py
```

---

### 3. `score.py`
Inference scoring script designed for deployment on Azure Machine Learning endpoints. Loads fine-tuned RoBERTa PyTorch weights and serves prediction payloads.
