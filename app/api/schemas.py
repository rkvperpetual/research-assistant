"""
app/api/schemas.py — Pydantic request/response models for all endpoints.
"""
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ─── /ask ────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096)
    conversation_id: Optional[str] = Field(default=None, description="UUID; omit for a new conversation")
    include_trace: bool = Field(default=True)


class TraceStepOut(BaseModel):
    step: int
    node: str
    detail: str
    ms: int


class CitationOut(BaseModel):
    id: int
    source: str
    page: Optional[int]
    snippet: str
    score: float
    origin: Literal["knowledge_base", "web_search"]


class AskResponse(BaseModel):
    answer: str
    conversation_id: str
    route: str
    sub_questions: List[str] = []
    citations: List[CitationOut] = []
    used_web_search: bool = False
    confidence: Literal["high", "medium", "low"] = "medium"
    grounded: bool = True
    trace: List[TraceStepOut] = []
    latency_ms: int


# ─── /documents ───────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    job_id: str
    status: str
    filename: str


class BulkIngestResponse(BaseModel):
    jobs: List[IngestResponse]


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    chunks: int = 0
    chunks_indexed: int = 0
    progress: int = 0
    stage: str = "queued"
    doc_id: Optional[str] = None
    file_type: Optional[str] = None
    error: Optional[str] = None


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    source_type: str
    chunk_count: int
    ingested_at: str


class HealthResponse(BaseModel):
    status: str
    qdrant_reachable: bool
    doc_count: int
    bm25_corpus_size: int


class UrlIngestRequest(BaseModel):
    url: str
    job_id: Optional[str] = None


class ConversationResponse(BaseModel):
    conversation_id: str
    turns: List[dict]
