# IFSI Phishing Email Detection — Deployment Guide

This guide details how to deploy the IFSI Phishing Detection system into production or staging environments, including quick zero-cost public submission links without requiring an external GPU/LLM server immediately.

---

## ⚡ Quick Start: Deploying for Submission (No LLM Required)

If you need a live public URL immediately to submit your project, the Frontend and Backend can be deployed right now without self-hosting an LLM or GPU:
- **M1 (TF-IDF + Logistic Regression)** runs 100% locally in Python (<15 ms).
- **Domain Telemetry Analyzer** runs 100% locally (entropy, SPF/DKIM heuristic checks, structural parsing).
- **FAISS Vector Store** (1,321 exemplars) runs 100% locally on CPU via `sentence-transformers`.
- **M4 (Two-Stage RAG)** gracefully returns domain telemetry and exemplar retrieval, falling back to a structured manual review status instead of throwing errors when Ollama is offline.
- **All Dashboards, McNemar Tests, Mistake Analysis, and Live Test pages** render completely.

---

### Method A: Single-URL Unified Deployment (Render / Railway / Cloud Run)

The root `Dockerfile` packages both the built React SPA and FastAPI backend into a single container. You only need to deploy **one** web service.

#### Step 1: Push your code to GitHub
```bash
git add .
git commit -m "feat: unified deployment configuration"
git push origin main
```

#### Step 2: Deploy on Render.com (Free / Hobby tier)
1. Log in to [Render.com](https://render.com) and click **New +** $\to$ **Web Service**.
2. Connect your GitHub repository.
3. Configure settings:
   - **Name**: `ifsi-phishing-detection`
   - **Environment**: `Docker`
   - **Dockerfile Path**: `Dockerfile` (root)
   - **Instance Type**: `Standard` or `Free`
4. Click **Deploy Web Service**.
5. Render builds the React app, prepares the Python API, and gives you a live HTTPS link:
   `https://ifsi-phishing-detection.onrender.com`

> **Note**: Both the React web application and all API endpoints (`/health`, `/presets`, `/classify`) are served from this single link!

---

### Method B: Decoupled Deployment (Vercel Frontend + Render/Railway Backend)

If you prefer deploying the Frontend on Vercel and Backend on Render/Railway:

#### Step 1: Deploy Backend to Render or Railway
1. Create a **New Web Service** from your GitHub repo.
2. Select **Docker** environment and point Dockerfile to `Dockerfile.backend`.
3. Once deployed, copy your backend URL: e.g. `https://ifsi-backend.onrender.com`.

#### Step 2: Deploy Frontend to Vercel
1. Log in to [Vercel.com](https://vercel.com) and click **Add New...** $\to$ **Project**.
2. Import your GitHub repository.
3. Set the following project settings:
   - **Root Directory**: `frontend`
   - **Framework Preset**: `Vite`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Under **Environment Variables**, add:
   - `VITE_API_BASE_URL` = `https://ifsi-backend.onrender.com`
5. Click **Deploy**. Vercel will provide your live URL: `https://ifsi-frontend.vercel.app`.

---

### Method C: Instant Public URL in 30 Seconds (Cloudflare Tunnel)

If your local server is currently running and you need an instant public link to share or submit immediately without cloud builds:

1. Download or run the official Cloudflare Tunnel CLI:
   ```bash
   npx cloudflared tunnel --url http://127.0.0.1:8000
   ```
2. Cloudflare immediately prints a live public HTTPS address:
   `https://<random-subdomain>.trycloudflare.com`
3. Anyone visiting this URL will see the complete application live from your machine.

---

## 2. System Architecture Overview

The system consists of three coordinated tiers:

```
[ Client Browser ]
        │ (HTTP/HTTPS :80 / :443)
        ▼
[ React 18 SPA + FastAPI Backend :8000 ]
   ├── M1: TF-IDF + Logistic Regression (Local CPU, <15ms)
   ├── FAISS Vector Store: 1,321 Exemplars (Local memory)
   ├── Domain Telemetry Analyzer
   └── Stage 1 & Stage 2 Orchestration
        │
        ▼ (HTTP :11434 - Optional / Staged)
[ Ollama LLM Service (`qwen3.5:4b`) ]
```

### Hardware Sizing Recommendations

| Resource | Minimum (CPU Only) | Recommended (With GPU) |
|---|---|---|
| **CPU** | 4 Cores (x86_64 or ARM64) | 4+ Cores |
| **RAM** | 8 GB System Memory | 16 GB System Memory |
| **GPU / VRAM** | None (CPU inference: ~6-8s/call) | NVIDIA GPU with **$\ge 4$ GB VRAM** (e.g. RTX 3060, T4, A10G) |
| **Disk** | 20 GB SSD | 30 GB NVMe SSD |

---

## 3. Production Docker Compose Deployment (With Ollama)

When ready to host the local LLM alongside the application, use `docker-compose.yml`:

### Step 1: Launch Services
```bash
docker compose up -d --build
```

### Step 2: (Optional) GPU Acceleration
If your host has an NVIDIA GPU and NVIDIA Container Toolkit installed, uncomment the `deploy` block in `docker-compose.yml`:
```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Step 3: Pull LLM Model
```bash
docker exec -it ifsi-ollama ollama pull qwen3.5:4b
```

---

## 4. Connecting a Remote LLM Later

When you are ready to connect an LLM, you do **not** need to redeploy the frontend:
1. In your backend cloud environment (Render, Cloud Run, AWS), set the environment variable:
   - `OLLAMA_API_BASE=https://your-gpu-server:11434`
2. Or use any OpenAI-compatible or vLLM server by updating the backend inference connector.
3. The backend automatically switches from offline fallback mode to live Stage 1 & Stage 2 LLM inference.

---

## 5. Environment Variables Reference

| Variable | Default Value | Description |
|---|---|---|
| `OLLAMA_API_BASE` | `http://127.0.0.1:11434` | URL of the Ollama or remote LLM inference endpoint |
| `PRIMARY_MODEL` | `qwen3.5:4b` | Default LLM model identifier for Stage 1 and Stage 2 |
| `VITE_API_BASE_URL` | `""` (same origin) | Backend URL for the React frontend client |
| `CORS_ORIGINS` | `*` (regex matching) | Comma-separated allowed CORS origins |
| `PYTHONPATH` | `src:.` | Python path configuration for package imports |
| `FAISS_INDEX_PATH` | `results/rag_index_clean` | File path to the indexed RAG exemplar corpus |

---

## 6. Verification Checklist

1. **Health Endpoint**:
   ```bash
   curl -s http://localhost:8000/health
   ```
2. **Model Catalog**:
   ```bash
   curl -s http://localhost:8000/models
   ```
3. **M1 Phishing Classification Test**:
   ```bash
   curl -X POST http://localhost:8000/classify/m1 \
     -H "Content-Type: application/json" \
     -d '{"text": "URGENT: Your account has been compromised. Verify now."}'
   ```
4. **Web UI Access**:
   Open your deployed URL in any web browser and confirm the Live Test and Comparison pages load cleanly.
