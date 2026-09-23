"""
app/main.py — FastAPI application entry point.

Responsibilities:
- Create the FastAPI app with metadata for Swagger
- Mount static files at /ui
- Include API routers
- Lifespan: bootstrap Qdrant collection + BM25 index on startup, init SQLite
- Health endpoint
- CORS (open for the demo; lock down for production)
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.ask import router as ask_router
from app.api.documents import router as documents_router
from app.api.schemas import HealthResponse
from app.config import get_settings
from app.core.memory import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)
_cfg = get_settings()


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== Research Assistant starting up ===")

    # 1. SQLite conversation store
    init_db()
    log.info("SQLite conversation store initialised")

    # 2. Qdrant collection
    try:
        from app.retrieval.vectorstore import bootstrap_collection
        bootstrap_collection()
    except Exception as e:
        log.warning("Qdrant bootstrap failed (continuing without): %s", e)

    # 3. BM25 index from existing Qdrant data
    try:
        from app.retrieval.bm25 import bootstrap_bm25_from_qdrant
        bootstrap_bm25_from_qdrant()
    except Exception as e:
        log.warning("BM25 bootstrap failed (continuing without): %s", e)

    # 4. Warm the LangGraph compiler (avoids first-request lag)
    try:
        from app.graph.builder import get_compiled_graph
        get_compiled_graph()
        log.info("LangGraph graph compiled and ready")
    except Exception as e:
        log.warning("Graph pre-compilation failed: %s", e)

    yield

    log.info("=== Research Assistant shutting down ===")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Step Research Assistant",
    description=(
        "A LangGraph-powered research assistant with hybrid RAG, "
        "query decomposition, relevance grading, web search fallback, "
        "and groundedness verification.\n\n"
        "LLM: NVIDIA Nemotron via OpenRouter | Embeddings: NVIDIA NV-EmbedQA via OpenRouter"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Open for the demo; restrict origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ask_router)
app.include_router(documents_router)

# ── Static files + UI ─────────────────────────────────────────────────────────
import os
_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/ui", include_in_schema=False)
async def ui():
    return FileResponse(os.path.join(_static_dir, "index.html"))


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    """Liveness + readiness probe."""
    qdrant_ok = False
    doc_count = 0
    try:
        from app.retrieval.vectorstore import get_qdrant_client
        client = get_qdrant_client()
        info = client.get_collection(_cfg.qdrant_collection)
        doc_count = info.points_count or 0
        qdrant_ok = True
    except Exception:
        pass

    from app.retrieval.bm25 import get_bm25_index
    bm25_size = get_bm25_index().corpus_size()

    return HealthResponse(
        status="ok" if qdrant_ok else "degraded",
        qdrant_reachable=qdrant_ok,
        doc_count=doc_count,
        bm25_corpus_size=bm25_size,
    )


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Multi-Step Research Assistant", "ui": "/ui", "docs": "/docs"}
