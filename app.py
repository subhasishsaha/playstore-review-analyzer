"""
Play Store Review Analyzer — Gradio Web Application with ZeroGPU Support.
Powered by RoBERTa (Sentiment), Multi-Label Ensemble (Aspects), and Qwen2.5-1.5B-Instruct (AI Suggestions).
"""
import os
import sys
from pathlib import Path
from typing import Tuple, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Hugging Face ZeroGPU integration
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    class _MockSpaces:
        def GPU(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            def decorator(func):
                return func
            return decorator
    spaces = _MockSpaces()

from src.pipeline import analyze_reviews_data
from src.config import HF_SUGGESTIONS_MODEL, HF_SUGGESTIONS_MODEL_URL


def create_aspect_bar_chart(labels_distribution: dict) -> go.Figure:
    """Creates a horizontal bar chart of aspect categories."""
    if not labels_distribution:
        fig = go.Figure()
        fig.update_layout(title="No Aspect Categories Identified")
        return fig

    labels = list(labels_distribution.keys())
    counts = list(labels_distribution.values())

    fig = px.bar(
        x=counts,
        y=labels,
        orientation="h",
        labels={"x": "Number of Reviews", "y": "Aspect Category"},
        title="Distribution of Review Aspects / Categories",
        color=counts,
        color_continuous_scale="Purples",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
    )
    return fig


def create_sentiment_pie_chart(positive: int, negative: int) -> go.Figure:
    """Creates a donut pie chart of sentiment distribution."""
    if positive + negative == 0:
        fig = go.Figure()
        fig.update_layout(title="No Sentiment Data Available")
        return fig

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Positive", "Negative"],
                values=[positive, negative],
                hole=0.45,
                marker=dict(colors=["#10B981", "#EF4444"]),
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        title="Overall Sentiment Ratio",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
    )
    return fig


def create_sentiment_by_aspect_chart(sentiment_by_label: dict) -> go.Figure:
    """Creates a grouped bar chart of sentiment breakdown per aspect."""
    if not sentiment_by_label:
        fig = go.Figure()
        fig.update_layout(title="No Aspect Sentiment Breakdown")
        return fig

    aspects = list(sentiment_by_label.keys())
    pos_counts = [sentiment_by_label[a].get("Positive", 0) for a in aspects]
    neg_counts = [sentiment_by_label[a].get("Negative", 0) for a in aspects]

    fig = go.Figure(
        data=[
            go.Bar(name="Positive", x=aspects, y=pos_counts, marker_color="#10B981"),
            go.Bar(name="Negative", x=aspects, y=neg_counts, marker_color="#EF4444"),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Sentiment by Aspect Category",
        xaxis_title="Category",
        yaxis_title="Review Count",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
    )
    return fig


def create_wordcloud(df: pd.DataFrame):
    """Generates a Matplotlib WordCloud figure for negative complaints."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
    if df is None or df.empty or "content" not in df.columns:
        ax.text(0.5, 0.5, "No reviews available", ha="center", va="center")
        ax.axis("off")
        return fig

    neg_reviews = df[df.get("sentiment", "") == "Negative"]["content"].dropna().tolist()
    if not neg_reviews:
        neg_reviews = df["content"].dropna().tolist()

    text = " ".join(neg_reviews).strip()
    if not text:
        ax.text(0.5, 0.5, "No complaint text available", ha="center", va="center")
        ax.axis("off")
        return fig

    try:
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="magma",
            max_words=80,
            collocations=False,
        ).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
    except Exception as e:
        ax.text(0.5, 0.5, f"Wordcloud generation: {e}", ha="center", va="center")
        ax.axis("off")

    plt.tight_layout(pad=0)
    return fig


@spaces.GPU(duration=60)
def process_reviews(
    app_id: str,
    sort_order: str,
    count: int
) -> Tuple[str, str, str, str, str, go.Figure, go.Figure, go.Figure, Any, pd.DataFrame]:
    """
    Main execution pipeline decorated with @spaces.GPU for Hugging Face ZeroGPU.
    Fetches Google Play reviews, runs RoBERTa & Ensemble models, and generates
    AI recommendations via Qwen2.5-1.5B-Instruct.
    """
    if not app_id or not app_id.strip():
        raise gr.Error("Please provide a valid Google Play App ID (e.g., com.google.android.gm).")

    try:
        result = analyze_reviews_data(
            app_id=app_id.strip(),
            sort_order=sort_order,
            count=int(count),
            ollama_model=HF_SUGGESTIONS_MODEL
        )
    except Exception as e:
        raise gr.Error(f"Analysis failed: {str(e)}")

    summary = result.get("summary", {})
    total = summary.get("total", 0)
    pos = summary.get("positive_count", 0)
    neg = summary.get("negative_count", 0)
    pos_pct = summary.get("positive_percentage", 0.0)
    neg_pct = summary.get("negative_percentage", 0.0)
    labels_dist = summary.get("labels_distribution", {})
    sentiment_by_label = summary.get("sentiment_by_label", {})

    top_aspect = list(labels_dist.keys())[0] if labels_dist else "None"

    # KPI Strings
    total_stat = f"**{total}** Reviews"
    pos_stat = f"**{pos}** ({pos_pct}%)"
    neg_stat = f"**{neg}** ({neg_pct}%)"
    aspect_stat = f"**{top_aspect}**"

    # Charts
    fig_aspect = create_aspect_bar_chart(labels_dist)
    fig_sentiment = create_sentiment_pie_chart(pos, neg)
    fig_breakdown = create_sentiment_by_aspect_chart(sentiment_by_label)

    # DataFrame
    reviews_data = result.get("reviews", [])
    df = pd.DataFrame(reviews_data)
    if not df.empty:
        # Format labels column as clean comma-separated string for display
        if "labels" in df.columns:
            df["labels"] = df["labels"].apply(
                lambda x: ", ".join(x) if isinstance(x, (list, tuple)) else str(x)
            )
        display_cols = [c for c in ["userName", "content", "sentiment", "labels", "score", "at", "appVersion"] if c in df.columns]
        display_df = df[display_cols]
    else:
        display_df = pd.DataFrame()

    fig_wc = create_wordcloud(df)
    suggestions_md = result.get("suggestions_markdown", "No suggestions generated.")

    return (
        total_stat,
        pos_stat,
        neg_stat,
        aspect_stat,
        suggestions_md,
        fig_aspect,
        fig_sentiment,
        fig_breakdown,
        fig_wc,
        display_df
    )


# Custom CSS for modern layout
CUSTOM_CSS = """
.app-container { max-width: 1200px; margin: auto; }
.metric-box { border-radius: 8px; padding: 12px; background: rgba(255, 255, 255, 0.05); }
.badge-pill { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; margin-right: 6px; font-weight: 600; }
"""

# Build Gradio Blocks Application
with gr.Blocks(title="Play Store Review Analyzer") as demo:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")
    gr.Markdown(
        """
        # 📱 Google Play Store Review Analyzer
        ### Automated Review Scraping, RoBERTa Sentiment Classification, Aspect Categorization & Qwen AI Recommendations.
        
        <span class="badge-pill" style="background:#4F46E5; color:white;">⚡ ZeroGPU Powered</span>
        <span class="badge-pill" style="background:#059669; color:white;">🤖 Qwen2.5-1.5B-Instruct</span>
        <span class="badge-pill" style="background:#D97706; color:white;">🎯 RoBERTa + Ensemble ML</span>
        """
    )

    with gr.Row():
        app_id_input = gr.Textbox(
            label="Google Play App ID",
            placeholder="e.g., com.google.android.gm, com.spotify.music, com.whatsapp",
            value="com.google.android.gm",
            scale=3,
        )
        sort_input = gr.Dropdown(
            choices=["Most Relevant", "Newest"],
            value="Most Relevant",
            label="Sort Order",
            scale=1,
        )
        count_input = gr.Slider(
            minimum=10,
            maximum=300,
            value=100,
            step=10,
            label="Reviews Count",
            scale=1,
        )

    with gr.Row():
        analyze_btn = gr.Button("🚀 Fetch & Analyze Reviews", variant="primary", size="lg")

    gr.Examples(
        examples=[
            ["com.google.android.gm", "Most Relevant", 100],
            ["com.spotify.music", "Most Relevant", 100],
            ["com.whatsapp", "Newest", 50],
        ],
        inputs=[app_id_input, sort_input, count_input],
    )

    gr.Markdown("---")

    # KPI Statistics Row
    with gr.Row():
        with gr.Column(scale=1):
            stat_total = gr.Markdown("### Total Reviews\n--")
        with gr.Column(scale=1):
            stat_pos = gr.Markdown("### Positive Reviews\n--")
        with gr.Column(scale=1):
            stat_neg = gr.Markdown("### Negative Reviews\n--")
        with gr.Column(scale=1):
            stat_aspect = gr.Markdown("### Top Aspect\n--")

    # Main Output Tabs
    with gr.Tabs():
        with gr.TabItem("💡 AI Developer Recommendations", id="tab_suggestions"):
            suggestions_output = gr.Markdown(
                "Click **Fetch & Analyze Reviews** to generate actionable improvements powered by `Qwen/Qwen2.5-1.5B-Instruct`."
            )

        with gr.TabItem("📊 Interactive Visualizations", id="tab_viz"):
            with gr.Row():
                viz_aspect = gr.Plot(label="Aspect Distribution")
                viz_sentiment = gr.Plot(label="Sentiment Distribution")
            with gr.Row():
                viz_breakdown = gr.Plot(label="Sentiment by Aspect")
                viz_wordcloud = gr.Plot(label="Negative Review WordCloud")

        with gr.TabItem("📋 Analyzed Reviews Table", id="tab_data"):
            data_output = gr.DataFrame(
                label="Reviews with Predicted Sentiments & Aspects",
                headers=["userName", "content", "sentiment", "labels", "score", "at", "appVersion"],
                wrap=True,
            )

    # Wire event listener
    analyze_btn.click(
        fn=process_reviews,
        inputs=[app_id_input, sort_input, count_input],
        outputs=[
            stat_total,
            stat_pos,
            stat_neg,
            stat_aspect,
            suggestions_output,
            viz_aspect,
            viz_sentiment,
            viz_breakdown,
            viz_wordcloud,
            data_output,
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
