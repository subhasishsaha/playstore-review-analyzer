"""
FastAPI REST API server for Play Store Review Analyzer.
Serves analysis endpoints and the static HTML/JS/CSS frontend.
"""
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Ensure path resolution
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import analyze_reviews_data
from src.config import DEFAULT_OLLAMA_MODEL, DEFAULT_REVIEW_COUNT


class AnalyzeRequest(BaseModel):
    app_id: str = Field(..., description="Google Play Store App Package Name, e.g., com.spotify.music")
    sort_order: Optional[str] = Field("Most Relevant", description="'Most Relevant' or 'Newest'")
    count: Optional[int] = Field(DEFAULT_REVIEW_COUNT, ge=10, le=500, description="Number of reviews to fetch (10-500)")
    ollama_model: Optional[str] = Field(None, description="Ollama model to use for suggestions (e.g. qwen3.5:0.8b, qwen2.5:0.5b)")


app = FastAPI(
    title="Play Store Review Analyzer API",
    description="Backend API for review analysis, sentiment categorization, and LLM suggestions.",
    version="2.0.0"
)

# Enable CORS for modern JS frontends (Node.js, Vite, Live Server, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Play Store Review Analyzer API",
        "ollama_model": DEFAULT_OLLAMA_MODEL,
    }


@app.get("/api/models")
def get_models():
    """Returns locally installed Ollama models and the default model."""
    from src.suggestions import get_installed_models
    installed = get_installed_models()
    return {
        "default_model": DEFAULT_OLLAMA_MODEL,
        "installed_models": installed,
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """
    Scrapes, analyzes sentiment, tags aspect categories, and generates LLM developer recommendations.
    """
    try:
        data = analyze_reviews_data(
            app_id=req.app_id,
            sort_order=req.sort_order or "Most Relevant",
            count=req.count or DEFAULT_REVIEW_COUNT,
            ollama_model=req.ollama_model
        )
        return data
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    fav_file = FRONTEND_DIR / "favicon.svg"
    if fav_file.is_file():
        return FileResponse(fav_file, media_type="image/svg+xml")
    raise HTTPException(status_code=404)


# Serve static frontend if directory exists (mounted at root after all /api routes)
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Launches the Uvicorn ASGI server."""
    import uvicorn
    if reload:
        uvicorn.run("src.api:app", host=host, port=port, reload=reload, app_dir=str(PROJECT_ROOT))
    else:
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server(port=8000)
