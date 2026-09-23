# RUNBOOK — How to Run Locally & Deploy

This document covers everything needed to run the Multi-Step Research Assistant
on your local machine and deploy it publicly on Render (free tier).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [API Keys You Need](#2-api-keys-you-need)
3. [Option A — Run Locally with Docker (Recommended)](#3-option-a--run-locally-with-docker-recommended)
4. [Option B — Run Locally Without Docker](#4-option-b--run-locally-without-docker)
5. [Seeding the Knowledge Base](#5-seeding-the-knowledge-base)
6. [Running Tests](#6-running-tests)
7. [Deploying to Render](#7-deploying-to-render)
8. [Deploying to Railway (Alternative)](#8-deploying-to-railway-alternative)
9. [Deploying to Fly.io (Alternative)](#9-deploying-to-flyio-alternative)
10. [Environment Variable Reference](#10-environment-variable-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [Quick Reference Commands](#12-quick-reference-commands)

---

## 1. Prerequisites

### For local development (Docker path)

| Tool | Minimum version | Install |
|---|---|---|
| Docker Desktop | 24+ | https://www.docker.com/products/docker-desktop |
| Docker Compose | v2 (bundled with Docker Desktop) | (bundled) |
| Python | 3.11+ | https://www.python.org/downloads |
| Git | any | https://git-scm.com |

### For local development (no-Docker path)

Everything above **minus** Docker, **plus**:
- Qdrant running locally (Docker image or binary)
- Tesseract OCR installed on your OS (for OCR features)
- Poppler utilities (for PDF-to-image conversion)

### For deployment

- GitHub account (to push your repo)
- Render account: https://render.com (free tier available)
- Qdrant Cloud account: https://cloud.qdrant.io (free 1 GB cluster)

---

## 2. API Keys You Need

| Service | Purpose | Where to get it | Cost |
|---|---|---|---|
| **OpenRouter** | LLM (NVIDIA Nemotron) + Embeddings | https://openrouter.ai - Dashboard - API Keys | ~$0.001-$0.01/request |
| **Tavily** | Web search fallback | https://app.tavily.com - API Keys | Free: 1,000 searches/month |
| **Qdrant Cloud** | Vector DB (production) | https://cloud.qdrant.io - Clusters | Free: 1 GB cluster |

> **Local development:** You only need OPENROUTER_API_KEY and TAVILY_API_KEY.
> Qdrant runs in Docker locally — no cloud key needed.

---

## 3. Option A — Run Locally with Docker (Recommended)

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/research-assistant.git
cd research-assistant
```

### Step 2 — Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Minimum required values in .env:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
TAVILY_API_KEY=tvly-your-key-here
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

### Step 3 — Build and start

```bash
docker compose up --build
```

First build takes 3-5 minutes (downloads Qdrant image, installs Tesseract + Poppler + Python deps).
Subsequent starts take ~10 seconds.

### Step 4 — Verify it is running

```bash
curl http://localhost:8000/health
```

Expected:
```json
{"status":"ok","qdrant_reachable":true,"doc_count":0,"bm25_corpus_size":0}
```

- UI: http://localhost:8000/ui
- Swagger: http://localhost:8000/docs

### Step 5 — Seed the knowledge base

In a separate terminal (app must already be running):

```bash
pip install httpx
python scripts/seed_kb.py
```

Expected output:
```
[seed_kb] Found 5 files to ingest
[seed_kb] Submitted 5 ingestion jobs
  + eu_ai_act_summary.md  — 12 chunks
  + nist_ai_rmf_summary.md — 14 chunks
  + autom8ai_employee_handbook.md — 18 chunks
  + q3_engineering_retro.md — 8 chunks
  + meeting_notes_aug_2025.md — 10 chunks
[seed_kb] All files ingested successfully
```

### Step 6 — Try a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Compare how the EU AI Act and NIST RMF handle risk classification\", \"include_trace\": true}"
```

### Stopping

```bash
docker compose down        # stop containers, keep Qdrant data
docker compose down -v     # stop AND delete Qdrant volume (fresh start)
```

---

## 4. Option B — Run Locally Without Docker

Use this for faster iteration with hot-reload.

### Step 1 — Start Qdrant (Docker only for Qdrant)

```bash
docker run -d -p 6333:6333 \
  -v qdrant_local:/qdrant/storage \
  --name qdrant \
  qdrant/qdrant:v1.11.0
```

Or download the Qdrant binary from: https://github.com/qdrant/qdrant/releases

### Step 2 — Install system dependencies (OCR)

**macOS:**
```bash
brew install tesseract poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install -y tesseract-ocr poppler-utils
```

**Windows:**
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Poppler: https://github.com/oschwartz10612/poppler-windows/releases
- Add both directories to your PATH

OCR is optional. Without Tesseract, scanned PDFs and images will be skipped but
all other formats (PDF text, DOCX, HTML, TXT, MD) work fine.

### Step 3 — Create virtual environment

```bash
python -m venv .venv

# macOS/Linux:
source .venv/bin/activate

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Windows CMD:
.venv\Scripts\activate.bat
```

### Step 4 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Step 6 — Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

The --reload flag auto-restarts on code changes.

### Step 7 — Seed and open

```bash
python scripts/seed_kb.py
# open http://localhost:8000/ui
```

---

## 5. Seeding the Knowledge Base

The data/sample_docs/ directory contains 5 sample documents:

| File | Topics |
|---|---|
| eu_ai_act_summary.md | EU AI Act 4-tier risk pyramid, GPAI, penalties |
| nist_ai_rmf_summary.md | NIST RMF 4 functions, trustworthy AI, EU comparison |
| autom8ai_employee_handbook.md | Leave policy, security, performance review |
| q3_engineering_retro.md | Arjun Sharma (L5) led migration, team structure |
| meeting_notes_aug_2025.md | Changelog, Q&A, roadmap |

### Adding your own documents

```bash
cp my_document.pdf data/sample_docs/
python scripts/seed_kb.py    # already-ingested files skipped (SHA-256 dedup)
```

**Supported formats:** .pdf .docx .html .htm .txt .md .png .jpg .jpeg .tiff .csv .xlsx

### Upload via the UI

1. Open http://localhost:8000/ui
2. Click "Choose files" and select documents
3. Click "Upload"
4. Watch status icons: spinning arrow (processing) -> checkmark (done)

### Upload via API

```bash
# Single file
curl -X POST http://localhost:8000/documents -F "files=@doc.pdf"

# Multiple files
curl -X POST http://localhost:8000/documents -F "files=@a.pdf" -F "files=@b.docx"

# Web page
curl -X POST http://localhost:8000/documents/url \
  -H "Content-Type: application/json" \
  -d "{\"url\": \"https://example.com/page\"}"
```

---

## 6. Running Tests

### Unit tests (no live server needed)

```bash
pytest tests/test_ingestion.py tests/test_graph_routing.py -v
```

Runs in ~5-10 seconds. No API keys or Qdrant needed.

### End-to-end tests (requires running server + seeded KB)

```bash
pytest tests/test_e2e_questions.py -v -s -m e2e
```

### Run E2E against the deployed URL

```bash
E2E_API_URL=https://your-app.onrender.com \
  pytest tests/test_e2e_questions.py -v -s -m e2e
```

---

## 7. Deploying to Render

Render is the recommended platform. Free tier supports Docker builds.

### Step 1 — Create a Qdrant Cloud cluster

1. Sign up at https://cloud.qdrant.io (free)
2. Create a cluster -> select a free tier region
3. Copy the cluster URL: https://abc123.us-east4-0.gcp.cloud.qdrant.io:6333
4. Go to API Keys -> create a key -> copy it

### Step 2 — Create .gitignore

Make sure your .gitignore contains:

```
.env
__pycache__/
*.pyc
.venv/
conversations.db
qdrant_storage/
```

### Step 3 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/research-assistant.git
git push -u origin main
```

### Step 4 — Create Render Web Service

1. Go to https://render.com -> Dashboard -> New -> Web Service
2. Connect GitHub and select your repository
3. Configure:

| Setting | Value |
|---|---|
| Name | research-assistant |
| Region | US East |
| Branch | main |
| Runtime | Docker |
| Dockerfile path | ./Dockerfile |
| Instance Type | Free |

### Step 5 — Set environment variables

In Render dashboard -> your service -> Environment tab, add:

| Key | Value |
|---|---|
| OPENROUTER_API_KEY | sk-or-v1-your-key |
| TAVILY_API_KEY | tvly-your-key |
| QDRANT_URL | https://abc123...qdrant.io:6333 |
| QDRANT_API_KEY | your-qdrant-cloud-key |
| LLM_MODEL | nvidia/llama-3.1-nemotron-70b-instruct |
| EMBEDDING_MODEL | nvidia/nv-embedqa-e5-v5 |

### Step 6 — Deploy

Click Create Web Service.
Render builds the Docker image (~5-10 min), then starts the container.
You get a URL: https://research-assistant-xxxx.onrender.com

### Step 7 — Seed the live KB

```bash
SEED_API_URL=https://research-assistant-xxxx.onrender.com \
  python scripts/seed_kb.py
```

### Step 8 — Verify

```bash
curl https://research-assistant-xxxx.onrender.com/health
```

- Swagger (must be public): https://your-app.onrender.com/docs
- UI: https://your-app.onrender.com/ui

### Render Free Tier Gotchas

| Issue | Cause | Fix |
|---|---|---|
| Cold start ~50 seconds | Free tier spins down after 15 min idle | Warn users in README; not a bug |
| 512 MB RAM limit | Low memory on free tier | Slim image; no cross-encoder |
| SQLite resets on redeploy | No persistent disk on free tier | Use Postgres for production |
| Build takes ~10 minutes | Tesseract + deps compilation | Cached on subsequent deploys |

---

## 8. Deploying to Railway (Alternative)

Railway gives $5 free credit on signup, no sleep on inactivity.

```bash
npm install -g @railway/cli
railway login
railway init
railway up

# Set environment variables
railway variables set OPENROUTER_API_KEY=sk-or-v1-...
railway variables set TAVILY_API_KEY=tvly-...
railway variables set QDRANT_URL=https://...
railway variables set QDRANT_API_KEY=...

railway open    # get your public URL
```

Railway auto-detects and uses the Dockerfile.

---

## 9. Deploying to Fly.io (Alternative)

Fly.io supports persistent volumes, better for SQLite persistence.

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

fly auth login
fly launch --dockerfile Dockerfile --no-deploy

fly secrets set OPENROUTER_API_KEY=sk-or-v1-...
fly secrets set TAVILY_API_KEY=tvly-...
fly secrets set QDRANT_URL=https://...
fly secrets set QDRANT_API_KEY=...

fly deploy
fly status
fly logs
```

For persistent SQLite on Fly.io:
```bash
fly volumes create conversation_data --size 1
```

Add to fly.toml:
```toml
[mounts]
  source = "conversation_data"
  destination = "/app"
```

---

## 10. Environment Variable Reference

| Variable | Default | Description |
|---|---|---|
| OPENROUTER_API_KEY | (required) | OpenRouter key for LLM + embeddings |
| LLM_MODEL | nvidia/llama-3.1-nemotron-70b-instruct | LLM model on OpenRouter |
| EMBEDDING_MODEL | nvidia/nv-embedqa-e5-v5 | Embedding model (1024-dim) |
| OPENROUTER_BASE_URL | https://openrouter.ai/api/v1 | OpenRouter API base |
| TAVILY_API_KEY | (required for web search) | Tavily web search key |
| QDRANT_URL | http://localhost:6333 | Qdrant cluster URL |
| QDRANT_API_KEY | (empty = no auth) | Qdrant Cloud API key |
| QDRANT_COLLECTION | research_kb | Collection name in Qdrant |
| CHUNK_SIZE | 1000 | Characters per chunk |
| CHUNK_OVERLAP | 180 | Overlap between chunks |
| RETRIEVAL_TOP_K | 6 | Chunks returned after RRF fusion |
| MAX_RETRIES | 2 | Max query rewrites before web search |
| MAX_SUB_QUESTIONS | 4 | Max decomposed sub-questions |
| GRAPH_RECURSION_LIMIT | 25 | LangGraph safety cap on node visits |
| EMBEDDING_DIM | 1024 | Embedding vector size (must match model) |

---

## 11. Troubleshooting

### "Connection refused" on Qdrant

qdrant_reachable: false in /health response.

- Docker path: run `docker ps` to check if Qdrant container is running.
- No-Docker path: make sure Qdrant binary/container is on port 6333.

### "401 Unauthorized" from OpenRouter

LLM or embedding calls fail with HTTP 401.

Check your key is valid:
```bash
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer YOUR_KEY"
```

Top up credits at https://openrouter.ai/settings/credits if empty.

### Embedding dimension mismatch

Qdrant rejects vectors with a dimension error.

Delete and recreate the Qdrant collection if you changed the embedding model.
Default dimension is 1024 for nvidia/nv-embedqa-e5-v5.
If you configure a different embedding model, set `EMBEDDING_DIM` to that
model's returned vector size before creating the collection. For example,
`nvidia/nemotron-3-embed-1b:free` uses `EMBEDDING_DIM=2048`.

To reset: `docker compose down -v && docker compose up --build`

### OCR not working (scanned PDFs return empty)

- Local: run `tesseract --version` to verify installation.
- Docker: rebuild with `docker compose build --no-cache`.

### "Graph recursion limit reached"

Increase GRAPH_RECURSION_LIMIT in .env (default 25).
This allows more looping but could increase latency on bad queries.

### First request slow (30-60 seconds locally)

Normal. Python imports, Qdrant connection, and LangGraph compilation happen on first request.
The lifespan hook pre-warms the graph. Subsequent requests are much faster.

### pip install fails on Windows (pytesseract, pdf2image)

These are Python wrappers around C binaries. Install the binaries first:
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Poppler: https://github.com/oschwartz10612/poppler-windows/releases
Then add both to your PATH and reinstall Python packages.

### Render deploy fails with out-of-memory

- Do NOT install `unstructured[all-docs]` (pulls PyTorch, ~2 GB).
- Check requirements.txt for accidental torch pulls.
- Upgrade to Render Starter ($7/mo) for 512 MB -> 2 GB RAM.

---

## 12. Quick Reference Commands

```bash
# Health check
curl http://localhost:8000/health | python -m json.tool

# List all documents
curl http://localhost:8000/documents | python -m json.tool

# Ask a question with trace
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is the parental leave policy?\", \"include_trace\": true}" \
  | python -m json.tool

# Upload a file
curl -X POST http://localhost:8000/documents -F "files=@doc.pdf"

# Qdrant dashboard (local)
# Navigate to: http://localhost:6333/dashboard

# View live server logs (Docker)
docker compose logs -f app

# Rebuild after code changes
docker compose up --build

# Run unit tests only (fast, no server needed)
pytest tests/test_ingestion.py tests/test_graph_routing.py -v

# Run all tests (server must be running)
pytest -v -s

# Run E2E against deployed URL
E2E_API_URL=https://your-app.onrender.com pytest tests/test_e2e_questions.py -v -s -m e2e
```
