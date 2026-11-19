# Production Dockerfile for Play Store Review Analyzer
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install curl for container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies with caching layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source, frontend dashboard, and documentation
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY docs/ ./docs/

# Create and switch to non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
