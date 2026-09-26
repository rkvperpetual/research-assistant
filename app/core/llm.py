"""
app/core/llm.py — Provider wrappers.

All LLM and embedding calls go through these helpers.
To swap a model, change ONE variable in .env; nothing else needs to change.

We use OpenRouter with NVIDIA Nemotron for the LLM and NVIDIA NV-EmbedQA for
embeddings. OpenRouter exposes an OpenAI-compatible REST API, so we can use
langchain-openai with a custom base_url.
"""
from __future__ import annotations

import httpx
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings

_cfg = get_settings()

# ─── HTTP client shared by OpenRouter calls ───────────────────────────────────
_http_client = httpx.Client(timeout=120)

# ─── LLM ─────────────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.0, streaming: bool = False) -> ChatGoogleGenerativeAI:
    """
    Return a ChatGoogleGenerativeAI instance pointed at Google's Gemini API.
    temperature=0.0 for structured / grading calls; higher for synthesis.
    """
    return ChatGoogleGenerativeAI(
        model=_cfg.llm_model,
        google_api_key=_cfg.google_api_key,
        temperature=temperature,
        streaming=streaming,
    )


def get_synthesis_llm() -> ChatGoogleGenerativeAI:
    """Slightly warmer LLM for answer synthesis."""
    return get_llm(temperature=0.2)


# ─── Embeddings ───────────────────────────────────────────────────────────────

def get_embeddings() -> OpenAIEmbeddings:
    """
    Return an OpenAIEmbeddings instance pointing at OpenRouter's embedding
    endpoint for nvidia/nv-embedqa-e5-v5 (1024-dim).
    """
    return OpenAIEmbeddings(
        model=_cfg.embedding_model,
        openai_api_key=_cfg.openrouter_api_key,
        openai_api_base=_cfg.openrouter_base_url,
        # OpenRouter embeddings endpoint: /api/v1/embeddings
        check_embedding_ctx_length=False,
        default_headers={
            "HTTP-Referer": "https://github.com/autom8ai/research-assistant",
            "X-Title": "Autom8AI Research Assistant",
        },
    )
