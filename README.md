---
title: Playstore Review Analyzer
emoji: 📱
colorFrom: blue
colorTo: green
sdk: gradio
app_file: src/app.py
pinned: false
---

# Play Store Review Analyzer 📱

This is a **Gradio** application that analyzes Google Play Store reviews using a custom RoBERTa model and Generative AI.

**Features:**
* **Sentiment Analysis:** Classifies reviews as Positive or Negative using a fine-tuned RoBERTa model.
* **Topic Modeling:** Auto-tags reviews with categories (e.g., "Bugs", "UI/UX", "Login Issues").
* **AI Suggestions:** Uses Google Gemini to generate actionable improvements for negative feedback.
* **Visualizations:** Interactive charts for sentiment and label distribution.