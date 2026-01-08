# Play Store Review Analyzer

An advanced NLP-powered application to scrape, analyze, and derive actionable insights from Google Play Store reviews. This tool utilizes an ensemble of Machine Learning models for classification, Deep Learning for sentiment analysis, and Generative AI (Google Gemini) for strategic recommendations.

## Features


* **Real-time Scraping:** Fetch reviews dynamically from the Google Play Store using App IDs.
* **Text Preprocessing:** Automated cleaning, emoji removal, and non-English sentence filtering.
* **Sentiment Analysis:** Fine-tuned **RoBERTa** model to classify reviews as Positive or Negative.
* **Multi-Label Classification:** Classifies reviews into specific categories (e.g., `Bugs`, `UI/UX`, `Ads`, `Cost`) using an **Ensemble Model** (Logistic Regression + XGBoost + Random Forest).
* **GenAI Insights:** Uses **Google Gemini API** to generate actionable suggestions for developers based on negative feedback clusters.
* **Interactive Visualizations:**
    * Label & Sentiment distribution charts (Plotly).
    * Category-specific sentiment breakdown.
    * App Version sentiment tracking.
    * Word Clouds.



## Repository Structure

```text
├── Dataset/
│   └── gpreviews_2.csv                  # Dataset used for training models
├── Test Notebooks/
│   ├── binary-sentiment-classification-using-roberta.ipynb  # RoBERTa training
│   ├── initial-multi-labelling-using-bart-large.ipynb       # Zero-shot labeling
│   ├── multi-label-classification-using-ensemble-approach.ipynb # Ensemble training
│   └── suggestion-using-gemini.ipynb    # GenAI experimentation
├── Models/                              # Directory for saved models (Created after training)
├── app.py                               # Main Streamlit Application
├── LICENSE
├── requirements.txt                     # Python dependencies
└── update_local_model.py
```
## Installation

- **Clone the Repository** 
    ```
    git clone https://github.com/subhasishsaha/sentiment-analysis.git
    cd play-store-review-analyzer
    ```
- **Install Dependencies**
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
- **Setup Environment Variables**

    Create a .env file in the root directory and add your Google Gemini API key.

    ```
    GEMINI_API_KEY=your_api_key_here
    ```
- **Generate/Download Models**

    ***Note**: The repository structure provided does not contain the trained model binaries (due to size). You must run the notebooks to generate them before running the app.*

    - **Labeling Model**: Run multi-label-classification-using-ensemble-approach.ipynb to generate:

        - *multilabel_binarizer.pkl*

        - *ensemble_model.pkl*

        - *tfidf.pkl*

        Place these in **Models/ensemble_models/**.
    
    - **Sentiment Model**: Run binary-sentiment-classification-using-roberta.ipynb to generate:

        - *roberta_tokenizer folder*

        - *best_model_state.bin*

        Place these in Models/roberta_tokenizer/ and Models/sentiment_model/ respectively.


## Run Locally

**Run the Application**
```
streamlit run app.py
```
## Methodology
The project employs a multi-stage NLP pipeline:

- **Data Labeling (Zero-Shot)**: The raw dataset was initially labeled using facebook/bart-large-mnli (Zero-Shot Classification) to create ground truth tags for specific app issues (See initial-multi-labelling-using-bart-large.ipynb).

- **Multi-Label Classification**: A lightweight ensemble model (TF-IDF vectorization fed into Voting Classifier of Logistic Regression, XGBoost, and Random Forest) is used for real-time tag inference on the web app.

- **Sentiment Analysis**: A customized PyTorch architecture utilizing FacebookAI/roberta-base was fine-tuned to detect sentiment nuances in app reviews.

- **Recommendation Engine**: Negative reviews are grouped by their predicted tags. These clusters are fed into Google Gemini-1.5-flash to generate summarized, technical suggestions for app developers.

**NOTE:** At the time of creating this, gemini 1.5 flash was available. You may need to change the Gemini Model to use the Recommendation Engine.

## Usage

- **Input App ID**: Find the package name in the Play Store URL (e.g., for WhatsApp, the ID is com.whatsapp).

- **Select Parameters**: Choose between "Newest" or "Most Relevant" and select the number of reviews to fetch.

- **Analyze**: Click "Fetch & Analyze".

- **Explore**:

    - View the raw dataframe with predicted labels.

    - Read AI-generated suggestions for fixing bugs/issues.

    - Navigate through the "Visualizations" dropdown to see charts.
    


## Contributing


Contributions are welcome! Please feel free to submit a Pull Request.

