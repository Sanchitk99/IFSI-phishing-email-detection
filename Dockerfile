# syntax=docker/dockerfile:1

# -----------------------------------------------------------------------------
# Stage 1: Build React Frontend SPA
# -----------------------------------------------------------------------------
FROM node:20-alpine AS frontend-builder

WORKDIR /build

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 2: Python Backend & Unified Static Server
# -----------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Install minimal system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests and source code
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install python production dependencies directly from pyproject.toml
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Pre-cache ONNX embedding model to eliminate cold-start runtime downloads
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'onnx/model.onnx'); hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'tokenizer.json')"

# Copy trained M1 model artifact, clean RAG exemplars, and benchmark summary data
COPY models/ ./models/
COPY results/rag_index_clean/ ./results/rag_index_clean/
COPY results/full_benchmark_summary.json results/report.md ./results/

# Copy compiled frontend from Stage 1 into frontend/dist
COPY --from=frontend-builder /build/dist/ ./frontend/dist/

# Set environment variables
ENV PYTHONPATH="/app/src:/app"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Start FastAPI server with 1 worker for 512 MB RAM safety (supports cloud PORT injection)
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
