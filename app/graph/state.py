"""
app/graph/state.py — LangGraph state definition for the research agent.

Everything the graph nodes read and write lives here.
TypedDict fields with Annotated[..., operator.add] are "append-only" lists;
all other fields are overwritten on each update.
"""
from __future__ import annotations

import operator
import time
from typing import Annotated, List, Literal, Optional
from typing_extensions import TypedDict


class Evidence(TypedDict):
    text: str
    source: str          # filename
    page: Optional[int]
    chunk_index: Optional[int]
    score: float
    origin: Literal["knowledge_base", "web_search"]
    sub_question_index: int


class Citation(TypedDict):
    id: int
    source: str
    page: Optional[int]
    snippet: str
    score: float
    origin: Literal["knowledge_base", "web_search"]


class TraceStep(TypedDict):
    step: int
    node: str
    detail: str
    ms: int


class ResearchState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    question: str
    conversation_id: str
    include_trace: bool

    # ── Contextualisation ────────────────────────────────────────────────────
    standalone_question: str
    chat_history: List[dict]

    # ── Routing ──────────────────────────────────────────────────────────────
    route: Literal["simple", "compound", "ambiguous", "chitchat"]

    # ── Sub-question loop ────────────────────────────────────────────────────
    sub_questions: List[str]
    current_sq_index: int

    # ── Retrieval / grading ──────────────────────────────────────────────────
    evidence: Annotated[List[Evidence], operator.add]   # accumulate across sub-qs
    graded_relevant: bool
    retry_count: int

    # ── Web search ───────────────────────────────────────────────────────────
    used_web_search: bool
    web_results: List[dict]

    # ── Synthesis / verification ─────────────────────────────────────────────
    draft_answer: str
    grounded: bool
    answer: str
    citations: List[Citation]

    # ── Meta ─────────────────────────────────────────────────────────────────
    trace: Annotated[List[TraceStep], operator.add]
    start_ms: float


def make_trace_step(step_num: int, node: str, detail: str, start: float) -> TraceStep:
    elapsed = int((time.perf_counter() - start) * 1000)
    return TraceStep(step=step_num, node=node, detail=detail, ms=elapsed)
