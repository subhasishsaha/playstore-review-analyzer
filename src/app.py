import gradio as gr
import pandas as pd
import numpy as np
import re
import emoji
import ast
import plotly.express as px
import matplotlib
matplotlib.use('Agg') # FIX: Force non-interactive backend for server stability
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
    print("⚠️ GEMINI_API_KEY not found. AI suggestions will not work.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Resource Loading ---
def load_resources():
    try:
        repo_id = "ssaha007/playstore-models"
        
        mlb_path = hf_hub_download(repo_id=repo_id, filename="ensemble_models/multilabel_binarizer.pkl")
        ensemble_path = hf_hub_download(repo_id=repo_id, filename="ensemble_models/ensemble_model.pkl")
        tfidf_path = hf_hub_download(repo_id=repo_id, filename="ensemble_models/tfidf.pkl")
        weights_path = hf_hub_download(repo_id=repo_id, filename="sentiment_model/best_model_state.bin")

        mlb = joblib.load(mlb_path)
        ensemble = joblib.load(ensemble_path)
        tfidf = joblib.load(tfidf_path)
        
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        class Sentiment_Classifier(nn.Module):
            def __init__(self, n_classes):
                super(Sentiment_Classifier, self).__init__()
                self.roberta = AutoModel.from_pretrained("roberta-base")
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

def generate_gemini_suggestions(label_to_reviews, model_name="gemini-1.5-flash", max_reviews=3):
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API Key missing. Skipping suggestions."
    
    model_gemini = genai.GenerativeModel(model_name)
    formatted_suggestions = ""
    
    for label, reviews in label_to_reviews.items():
        if not reviews:
            continue
        prompt = f"""You are an AI assistant. Based on the following negative reviews about the category '{label}', suggest actionable improvements:\n\n{chr(10).join('- ' + r for r in reviews[:max_reviews])}"""
        try:
            response = model_gemini.generate_content(prompt)
            suggestion_text = response.text.strip()
            formatted_suggestions += f"### Suggestions for: **{label.title()}**\n{suggestion_text}\n\n---\n\n"
        except Exception as e:
            formatted_suggestions += f"### {label.title()}\n⚠️ Error: {e}\n\n"
            
    if not formatted_suggestions:
        return "No actionable negative reviews found to generate suggestions."
    return formatted_suggestions

def get_unique_labels(df):
    if df is None or df.empty:
        return []
    all_labels = df["labels"].explode().dropna()
    unique = sorted(list(set([l for l in all_labels if l != 'unknown'])))
    return unique

# --- Core Analysis Logic ---
def analyze_reviews(app_id, sort_order, count):
    if not app_id:
        raise gr.Error("Please enter a valid App ID.")

    try:
        sort_map = {'Newest': Sort.NEWEST, 'Most Relevant': Sort.MOST_RELEVANT}
        result, _ = reviews(app_id, sort=sort_map[sort_order], count=int(count), lang="en", country="us")
        
        if not result:
            raise gr.Error("No reviews fetched. Check App ID.")
        
        df = pd.DataFrame(result)
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

        label_to_reviews = prepare_label_to_reviews(df)
        suggestions_md = generate_gemini_suggestions(label_to_reviews)

        display_cols = ['userName', 'content', 'at', 'appVersion', 'sentiment', 'labels']
        return df[display_cols], df, suggestions_md

    except Exception as e:
        raise gr.Error(f"Analysis Failed: {str(e)}")

# --- Visualization Logic (FIXED) ---

def update_viz_controls(viz_type, df):
    """
    Updates the sub-options dropdown.
    """
    if df is None or df.empty:
        return gr.update(visible=False)

    if viz_type == "Sentiment by Label":
        labels = get_unique_labels(df)
        if not labels:
             return gr.update(visible=False)
        # We explicitly set the 'value' to the first label to ensure the next function gets a valid input
        return gr.update(label="Select Label", choices=labels, value=labels[0], visible=True)

    elif viz_type == "Word Cloud":
        return gr.update(label="Select Sentiment", choices=["Positive", "Negative"], value="Positive", visible=True)

    else:
        return gr.update(visible=False)

def update_visualization(df, viz_type, param_selection):
    """
    Generates the plot. Includes fix for Tuple/List mismatch and blank graphs.
    """
    # --- Helper to create a "No Data" message plot ---
    def create_error_plot(message):
        fig = px.bar(x=[0], y=[0])
        fig.update_layout(
            xaxis={"visible": False}, 
            yaxis={"visible": False}, 
            annotations=[{
                "text": message, 
                "xref": "paper", "yref": "paper", 
                "showarrow": False, "font": {"size": 20}
            }]
        )
        return fig

    if df is None or df.empty:
        return create_error_plot("Please fetch reviews first.")

    # 1. Label Distribution
    if viz_type == "Label Distribution":
        all_labels = df["labels"].explode().dropna()
        if all_labels.empty: 
            return create_error_plot("No labels found in these reviews.")
            
        label_counts = all_labels.value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']
        fig = px.bar(label_counts, x='Count', y='Label', orientation='h',
                     title="Distribution of Review Labels", color='Count',
                     color_continuous_scale=px.colors.sequential.Magma)
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        return fig

    # 2. Overall Sentiment
    elif viz_type == "Sentiment Distribution":
        if 'sentiment' not in df.columns:
            return create_error_plot("Sentiment data missing.")
            
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                     title="Overall Sentiment Distribution", hole=0.3,
                     color_discrete_map={'Positive': 'mediumseagreen', 'Negative': 'indianred'})
        return fig

    # 3. Sentiment by Label (CRITICAL FIX HERE)
    elif viz_type == "Sentiment by Label":
        if not param_selection:
            unique_labels = get_unique_labels(df)
            if unique_labels:
                param_selection = unique_labels[0]
            else:
                return create_error_plot("No categories found to analyze.")
        
        # FIX: Check for both list AND tuple, or string fallback
        def has_label(row_labels, target):
            if isinstance(row_labels, (list, tuple)):
                return target in row_labels
            elif isinstance(row_labels, str):
                return target in row_labels # Handle case if data became string
            return False

        mask = df['labels'].apply(lambda x: has_label(x, param_selection))
        filtered_df = df[mask]
        
        if filtered_df.empty:
            return create_error_plot(f"No reviews found for '{param_selection}'")

        sentiment_counts = filtered_df['sentiment'].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                     title=f"Sentiment Distribution for '{param_selection}'", hole=0.3,
                     color_discrete_map={'Positive': 'mediumseagreen', 'Negative': 'indianred'})
        return fig

    # 4. Word Cloud
    elif viz_type == "Word Cloud":
        target_sentiment = param_selection if param_selection else "Positive"
        
        text_data = df[df["sentiment"] == target_sentiment]["content"].dropna()
        if text_data.empty: 
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.text(0.5, 0.5, f"No {target_sentiment} reviews found.", 
                    horizontalalignment='center', verticalalignment='center', fontsize=12)
            ax.axis("off")
            return fig
        
        text = " ".join(text_data.tolist())
        cmap = "Greens" if target_sentiment == "Positive" else "Reds"
        
        wordcloud = WordCloud(width=800, height=400, background_color="white", colormap=cmap, collocations=False).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{target_sentiment} Word Cloud")
        plt.close(fig) 
        return fig

    return create_error_plot("Select a visualization type.")
# --- Gradio UI ---
with gr.Blocks(title="Play Store Review Analyzer") as demo:
    gr.Markdown("# 📱 Play Store Review Analyzer")
    
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
                    ["Label Distribution", "Sentiment Distribution", "Sentiment by Label", "Word Cloud"], 
                    label="Select Visualization", 
                    value="Label Distribution"
                )
                viz_param = gr.Dropdown(label="Options", visible=False)
            
            viz_output = gr.Plot(label="Visualization")

    # --- Chains ---
    
    # 1. Analyze: Fetch -> Update Controls -> Update Viz
    analyze_btn.click(
        fn=analyze_reviews,
        inputs=[app_id_input, sort_input, count_input],
        outputs=[data_output, df_state, suggestions_output]
    ).then(
        fn=update_viz_controls,
        inputs=[viz_dropdown, df_state],
        outputs=[viz_param]
    ).then(
        fn=update_visualization,
        inputs=[df_state, viz_dropdown, viz_param],
        outputs=[viz_output]
    )

    # 2. Change Viz Type: Update Controls -> Update Viz
    viz_dropdown.change(
        fn=update_viz_controls,
        inputs=[viz_dropdown, df_state],
        outputs=[viz_param]
    ).then(
        fn=update_visualization,
        inputs=[df_state, viz_dropdown, viz_param],
        outputs=[viz_output]
    )

    # 3. Change Sub-Option: Update Viz only
    viz_param.change(
        fn=update_visualization,
        inputs=[df_state, viz_dropdown, viz_param],
        outputs=[viz_output]
    )

if __name__ == "__main__":
    demo.launch()