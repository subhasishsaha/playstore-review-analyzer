"""
Main application launcher for Play Store Review Analyzer.
Runs the FastAPI server that serves both the REST API and the independent JS/HTML/CSS frontend.
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
from src.api import run_server

HOST = os.getenv("HOST", "127.0.0.1")
raw_port = os.getenv("PORT", "8000")
try:
    PORT = int(raw_port) if raw_port and raw_port.strip() else 8000
except ValueError:
    PORT = 8000

if __name__ == "__main__":
    run_server(host=HOST, port=PORT)

