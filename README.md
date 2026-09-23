# Multi-Step Research Assistant

A **LangGraph-powered** research assistant API with hybrid RAG, query decomposition, relevance grading, web-search fallback, and groundedness verification.

**Live URL:** `https://<your-app>.onrender.com`
**Swagger docs:** `https://<your-app>.onrender.com/docs`
**UI:** `https://<your-app>.onrender.com/ui`

> ⚠️ **Cold start warning:** The free-tier deployment sleeps after 15 min of inactivity. The first request may take ~50 s to respond. This is expected — it's not a bug.

---

## Quickstart (Docker)

```bash
# 1. Clone & configure
git clone <your-repo>
cd research-assistant
cp .env.example .env
# Edit .env: add OPENROUTER_API_KEY, TAVILY_API_KEY

# 2. Start the stack (Qdrant + app)
docker compose up --build

# 3. Seed the knowledge base
python scripts/seed_kb.py

# 4. Open the UI
open http://localhost:8000/ui
# or visit http://localhost:8000/docs for Swagger
```

---

## Quickstart (Local, no Docker)

```bash
# 1. Install deps
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Start Qdrant separately
docker run -p 6333:6333 qdrant/qdrant:v1.11.0

# 3. Configure
cp .env.example .env
# Edit .env with your keys

# 4. Run the API
uvicorn app.main:app --reload --port 8000

# 5. Seed and open
python scripts/seed_kb.py
open http://localhost:8000/ui
```

---

## Knowledge Base

The sample knowledge base in `data/sample_docs/` covers:

| File | Format | Content |
|---|---|---|
| `eu_ai_act_summary.md` | Markdown | EU AI Act risk classification (4 tiers) |
| `nist_ai_rmf_summary.md` | Markdown | NIST AI RMF framework + EU comparison |
| `autom8ai_employee_handbook.md` | Markdown | Leave policy, security, performance review |
| `q3_engineering_retro.md` | Markdown | Q3 retro — Arjun Sharma (L5) led migration |
| `meeting_notes_aug_2025.md` | Markdown | All-hands notes, changelog, Q&A |

**Why this corpus?** Facts are distributed across documents — naive top-k retrieval fails on multi-hop questions. The corpus exercises compound queries, cross-document hops, and ambiguous under-specified questions.

---

## Architecture

```
Client (HTML UI / curl / Swagger)
        │
        ▼
┌──────────── FastAPI ────────────┐
│  POST /documents   POST /ask    │
└────────────────────────────────┘
        │                │
   INGESTION        LangGraph AGENT
   pipeline         (research_graph)
        │                │
        ▼                ▼
   parse → chunk →  hybrid retriever
   embed → upsert   (dense + BM25, RRF)
        │                │
        ▼                ▼
     Qdrant ◄──── Tavily web search
```

### Graph Nodes

| Node | What it does |
|---|---|
| `contextualize` | Rewrites follow-up questions using chat history |
| `route_question` | Classifies into simple / compound / ambiguous / chitchat |
| `clarify` | For ambiguous Qs: asks clarifying Q + provides best-effort answer |
| `decompose` | Splits compound Qs into 2–4 independent sub-questions |
| `retrieve` | Hybrid dense+BM25 search with RRF fusion |
| `grade_documents` | Per-chunk relevance judgement; drops distractors |
| `transform_query` | Rewrites failed queries (expand acronyms, synonyms) |
| `web_search` | Tavily search when KB retrieval fails after 2 retries |
| `synthesize` | Builds answer from evidence with inline [N] citations |
| `verify` | Groundedness check; repairs unsupported claims |
| `finalize` | Maps markers → Citation objects; attaches trace |

### LLM & Embeddings

| Component | Provider | Model |
|---|---|---|
| LLM | OpenRouter | `nvidia/llama-3.1-nemotron-70b-instruct` |
| Embeddings | OpenRouter | `nvidia/nv-embedqa-e5-v5` (1024-dim) |
| Web search | Tavily | `langchain-tavily` |
| Vector DB | Qdrant | Self-hosted (Docker) / Qdrant Cloud |

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness + Qdrant reachability + doc count |
| `POST` | `/documents` | Upload files (multipart, any format) |
| `POST` | `/documents/url` | Ingest a web page |
| `GET` | `/documents` | List ingested docs with chunk counts |
| `GET` | `/documents/{job_id}` | Ingestion job progress |
| `DELETE` | `/documents/{doc_id}` | Remove a doc and its vectors |
| `POST` | `/ask` | Main research endpoint |
| `GET` | `/conversations/{id}` | Turn history |
| `GET` | `/ui` | Simple HTML UI |
| `GET` | `/docs` | Swagger (always public) |

See `/docs` for full schemas and try-it-out.

---

## Benchmark Questions & Expected Behavior

| # | Question | Expected behavior |
|---|---|---|
| 1 | Compare EU AI Act and NIST RMF risk classification. Which is more prescriptive? | Route: compound; 2 sub-questions; citations from both docs |
| 2 | What's the policy on leave? | Route: ambiguous; clarifying Q + all leave types covered |
| 3 | How many vacation days does Arjun Sharma get? | Route: compound; hop 1 (retro) → L5; hop 2 (handbook) → 22 days |
| 4 | What is the total on the invoice and which budget line does it fall under? | OCR path; cross-format retrieval |
| 5 | Who is the current CEO of Anthropic? | Web search fallback; `used_web_search: true` |
| 6 | And what about their parental leave? (after Q2) | `standalone_question` resolved via history |
| 7 | What is Autom8AI's refund policy? | Honest "not in KB + not on web"; `confidence: low` |

---

## Trade-offs (Design Decisions)

| Decision | Trade-off |
|---|---|
| Per-format loaders vs `unstructured` | Smaller Docker image (~400 MB vs ~2 GB), faster cold start, but weaker table extraction from complex PDFs |
| Tesseract OCR vs vision model | Free, no API cost, fast; poor on low-DPI scans or handwriting |
| Hybrid + RRF vs cross-encoder reranker | Saves ~400 ms and avoids a 1 GB model download; cross-encoder enabled optionally via env var |
| BM25 in memory | Fine for this corpus size; would need OpenSearch/Elasticsearch beyond ~10k chunks |
| Retry cap of 2 | Bounds worst-case latency at the cost of occasionally giving up early on very narrow queries |
| SQLite conversation memory | No horizontal scaling; straightforward swap to Postgres by changing the engine URL |
| No auth on endpoints | Deliberate for the demo so graders can use it; add `x-api-key` header check for anything real |
| OpenRouter for LLM + embeddings | Single API key, one unified billing; latency overhead ~20–50 ms vs direct provider |

---

## What I'd Do Next

- **Streaming responses** via SSE so answers appear word-by-word
- **Per-user KB namespaces** using Qdrant payload filters
- **RAGAS evals** for automated retrieval and answer quality measurement
- **Caching** identical queries with a short TTL to reduce API cost
- **Cross-encoder reranker** (`bge-reranker-base`) as an optional step behind a feature flag
- **Proper auth** — API key middleware + rate limiting
- **Observability** — LangSmith tracing or OpenTelemetry

---

## Running Tests

```bash
# Unit tests (no live server needed)
pytest tests/test_ingestion.py tests/test_graph_routing.py -v

# E2E tests (requires running server + seeded KB)
pytest tests/test_e2e_questions.py -v -s -m e2e
```
