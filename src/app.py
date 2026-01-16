import gradio as gr
import pandas as pd
import numpy as np
import re
import emoji
import ast
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from langdetect import detect, LangDetectException
from google_play_scraper import reviews, Sort
import joblib
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
from collections import defaultdict
from dotenv import load_dotenv
import os
from huggingface_hub import hf_hub_download

# --- Configuration & Setup ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please add it to your .env file or Secrets.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Resource Loading (Cached Globally) ---
def load_resources():
    try:
        repo_id = "ssaha007/playstore-models"
        
        # Download paths
        mlb_path = hf_hub_download(repo_id=repo_id, filename="ensemble_models/multilabel_binarizer.pkl")
        ensemble_path = hf_hub_download(repo_id=repo_id, filename="ensemble_models/ensemble_model.pkl")
        tfidf_path = hf_hub_download(repo_id=repo_id, filename="ensemble_models/tfidf.pkl")
        weights_path = hf_hub_download(repo_id=repo_id, filename="sentiment_model/best_model_state.bin")

        # Load Models
        mlb = joblib.load(mlb_path)
        ensemble = joblib.load(ensemble_path)
        tfidf = joblib.load(tfidf_path)
        
        # Load Architecture
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        class Sentiment_Classifier(nn.Module):
            def __init__(self, n_classes):
                super(Sentiment_Classifier, self).__init__()
                self.roberta = AutoModel.from_pretrained("roberta-base")
                # Freeze/Unfreeze logic
                for param in self.roberta.parameters():
                    param.requires_grad = False
                for layer in self.roberta.encoder.layer[-2:]:
                    for param in layer.parameters():
                        param.requires_grad = True       
                self.drop = nn.Dropout(p=0.1)
                self.out = nn.Linear(self.roberta.config.hidden_size, n_classes)

            def forward(self, input_ids, attention_mask):
                output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = output.last_hidden_state[:, 0, :]
                output = self.drop(pooled_output)
                output = self.out(output)
                return output
    
        model = Sentiment_Classifier(n_classes = 2)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        
        return mlb, ensemble, tfidf, tokenizer, model

    except Exception as e:
        raise RuntimeError(f"Failed to load resources: {e}")

# Load resources once at startup
print("⏳ Loading models...")
mlb, ensemble, tfidf, tokenizer, model = load_resources()
print("✅ Models loaded.")
    
softmax = nn.Softmax(dim=1)

# --- Helper Functions ---
def preprocess(text):
    text = emoji.replace_emoji(text or "", "")
    text = re.sub(r"http\S+|www\S+|<.*?>|\n|\w*\d\w*", '', text).strip()
    return text if text else np.nan

def filter_english_sentences(text):
    if not isinstance(text, str):
        return np.nan
    sentences = [s.strip() for s in text.split('.') if len(s.split()) > 1]
    english = []
    for s in sentences:
        try:
            if detect(s) == 'en':
                english.append(s)
        except LangDetectException:
            continue
    return '. '.join(english) if english else np.nan

def get_labels_batch(texts):
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []
    X = tfidf.transform(texts)
    y_pred = ensemble.predict(X)
    y_pred = np.array(y_pred)
    all_labels = mlb.inverse_transform(y_pred)
    return [labels if labels else ("unknown",) for labels in all_labels]

def predict_sentiments(texts, batch_size=32):
    sentiments = []
    texts_to_process = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts_to_process:
        return []
        
    for i in range(0, len(texts_to_process), batch_size):
        batch = texts_to_process[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256)
        with torch.no_grad():
            outputs = model(enc["input_ids"], enc["attention_mask"])
            probs = softmax(outputs)
            sentiments.extend(["Positive" if p == 1 else "Negative" for p in torch.argmax(probs, dim=1).tolist()])
    return sentiments

def prepare_label_to_reviews(df, max_reviews_per_label=10):
    label_to_reviews = defaultdict(list)
    for _, row in df[df["sentiment"] == "Negative"].iterrows(): 
        labels = row.get("labels", [])
        if isinstance(labels, str):
            try:
                labels = ast.literal_eval(labels)
            except (ValueError, SyntaxError):
                labels = []
        for label in labels:
            clean = label.strip().lower()
            if clean != "unknown":
                label_to_reviews[clean].append(row["content"])
    return {k: list(dict.fromkeys(v))[:max_reviews_per_label] for k, v in label_to_reviews.items()}

def generate_gemini_suggestions(label_to_reviews, model_name="gemini-2.5-flash-lite", max_reviews=3):
    model_gemini = genai.GenerativeModel(model_name)
    formatted_suggestions = ""
    
    for label, reviews in label_to_reviews.items():
        if not reviews:
            continue
        prompt = f"""You are an AI assistant. Based on the following negative reviews about the category '{label}', suggest actionable improvements:\n\n{chr(10).join('- ' + r for r in reviews[:max_reviews])}"""
        try:
            response = model_gemini.generate_content(prompt)
            suggestion_text = response.text.strip()
            # Format as Markdown for Gradio
            formatted_suggestions += f"### Suggestions for: **{label.title()}**\n{suggestion_text}\n\n---\n\n"
        except Exception as e:
            formatted_suggestions += f"### {label.title()}\n⚠️ Error: {e}\n\n"
            
    if not formatted_suggestions:
        return "No actionable negative reviews found to generate suggestions."
    return formatted_suggestions

# --- Core Logic Wrapper ---
def analyze_reviews(app_id, sort_order, count):
    if not app_id:
        raise gr.Error("Please enter a valid App ID.")

    try:
        # 1. Fetch
        sort_map = {'Newest': Sort.NEWEST, 'Most Relevant': Sort.MOST_RELEVANT}
        result, _ = reviews(app_id, sort=sort_map[sort_order], count=int(count), lang="en", country="us")
        
        if not result:
            raise gr.Error("No reviews fetched. Check App ID.")
        
        df = pd.DataFrame(result)

        # 2. Process
        df_processed = df.copy()
        df_processed["content_processed"] = df_processed["content"].apply(preprocess)
        df_processed.dropna(subset=['content_processed'], inplace=True)
        
        if not df_processed.empty:
            df_processed["content_processed"] = df_processed["content_processed"].apply(filter_english_sentences)
            df_processed.dropna(subset=['content_processed'], inplace=True)

        if df_processed.empty:
            raise gr.Error("No valid English content found.")

        texts = df_processed["content_processed"].tolist()
        df_processed["labels"] = get_labels_batch(texts)
        df_processed["sentiment"] = predict_sentiments(texts)
        
        df = df.merge(df_processed[['reviewId', 'sentiment', 'labels']], on='reviewId', how='right')
        df.reset_index(drop=True, inplace=True)

        # 3. Generate Suggestions
        label_to_reviews = prepare_label_to_reviews(df)
        suggestions_md = generate_gemini_suggestions(label_to_reviews)

        # Return: Dataframe (for display), Dataframe (for State), Suggestions
        display_cols = ['userName', 'content', 'at', 'appVersion', 'sentiment', 'labels']
        return df[display_cols], df, suggestions_md

    except Exception as e:
        raise gr.Error(f"Analysis Failed: {str(e)}")

# --- Visualization Logic ---
def update_visualization(df, viz_type, selected_cat=None, selected_ver=None):
    if df is None or df.empty:
        return None

    if viz_type == "Label Distribution":
        all_labels = df["labels"].explode().dropna()
        if all_labels.empty: return None
        label_counts = all_labels.value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']
        fig = px.bar(label_counts, x='Count', y='Label', orientation='h',
                     title="Distribution of Review Labels", color='Count',
                     color_continuous_scale=px.colors.sequential.Magma)
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        return fig

    elif viz_type == "Sentiment Distribution":
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                     title="Overall Sentiment Distribution", hole=0.3,
                     color_discrete_map={'Positive': 'mediumseagreen', 'Negative': 'indianred'})
        return fig

    elif viz_type == "Word Cloud":
        # Note: Gradio needs a Matplotlib Figure object, not pyplot state
        text_data = df[df["sentiment"] == "Negative"]["content"].dropna() # Default to negative for impact
        if text_data.empty: return None
        text = " ".join(text_data.tolist())
        wordcloud = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        return fig

    # For dropdown logic, we can just return the label distribution as default 
    # or expand this function to accept specific inputs if we add more UI controls.
    return None

# --- Gradio UI ---
with gr.Blocks(title="Play Store Review Analyzer") as demo:
    gr.Markdown("# 📱 Play Store Review Analyzer")
    
    # Store the full dataframe in a hidden state component
    df_state = gr.State()

    with gr.Row():
        app_id_input = gr.Textbox(label="App ID", placeholder="e.g., com.google.android.gm")
        sort_input = gr.Dropdown(["Most Relevant", "Newest"], label="Sort Order", value="Most Relevant")
        count_input = gr.Number(label="Review Count", value=100, step=10)
    
    analyze_btn = gr.Button("🚀 Fetch & Analyze", variant="primary")

    with gr.Tabs():
        with gr.TabItem("📊 Data"):
            data_output = gr.Dataframe(label="Analyzed Reviews", headers=['userName', 'content', 'at', 'appVersion', 'sentiment', 'labels'])
        
        with gr.TabItem("💡 Suggestions"):
            suggestions_output = gr.Markdown()
            
        with gr.TabItem("🧭 Visualizations"):
            with gr.Row():
                viz_dropdown = gr.Dropdown(
                    ["Label Distribution", "Sentiment Distribution", "Word Cloud"], 
                    label="Select Visualization", 
                    value="Label Distribution"
                )
            viz_output = gr.Plot(label="Visualization")

    # Event: Click Analyze
    analyze_btn.click(
        fn=analyze_reviews,
        inputs=[app_id_input, sort_input, count_input],
        outputs=[data_output, df_state, suggestions_output]
    )

    # Event: Change Visualization Dropdown (or when data loads)
    # We chain this so it triggers after analysis or when user changes dropdown
    viz_dropdown.change(
        fn=update_visualization,
        inputs=[df_state, viz_dropdown],
        outputs=[viz_output]
    )
    
    # Also update viz when analysis finishes (default view)
    analyze_btn.click(
        fn=update_visualization,
        inputs=[df_state, viz_dropdown],
        outputs=[viz_output]
    )

if __name__ == "__main__":
    demo.launch()