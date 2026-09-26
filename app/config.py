"""
app/config.py — Pydantic-settings config, reads from .env
Every env var is defined once here; nothing else reads os.environ directly.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Google Gemini (LLM) ─────────────────────────────────────────────────
    google_api_key: str = ""
    llm_model: str = "gemini-1.5-flash"

    # ── OpenRouter (Embeddings) ─────────────────────────────────────────────
    openrouter_api_key: str = ""
    # NVIDIA embedding via OpenRouter
    embedding_model: str = "nvidia/nemotron-3-embed-1b:free"
    # OpenRouter base URL
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ── Tavily ──────────────────────────────────────────────────────────────
    tavily_api_key: str = ""

    # ── Qdrant ──────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "research_kb"

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 180

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieval_top_k: int = 6

    # ── Graph / agent ────────────────────────────────────────────────────────
    max_retries: int = 2
    max_sub_questions: int = 4
    graph_recursion_limit: int = 25

    # ── Embedding dimension (for Qdrant collection creation) ─────────────────
    # nvidia/nemotron-3-embed-1b → 2048-dim via OpenRouter
    embedding_dim: int = 2048


@lru_cache
def get_settings() -> Settings:
    return Settings()
