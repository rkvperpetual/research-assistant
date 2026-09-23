"""
app/graph/builder.py — StateGraph wiring.

Constructs the full research agent graph with conditional edges.
The compiled graph is cached as a module-level singleton.

Graph topology:
  START → contextualize → route_question
    ├─ chitchat  ─────────────────────────────────► synthesize
    ├─ ambiguous ─► clarify ─► finalize
    ├─ compound  ─► decompose ─► retrieve
    └─ simple    ─────────────► retrieve
                                  ▼
                            grade_documents
            ┌─────────────────────┼──────────────────────┐
      relevant│         not relevant & retries<2  not relevant & retries≥2
             │                   │                       │
             │          transform_query             web_search
             │                   │                       │
             │                   └──► retrieve            │
             │                                           ▼
             ▼                                      synthesize  ← web results go straight here
       more sub-questions? ──yes──► retrieve (next index)         (no re-grading web results)
             │no
             ▼
         synthesize ──► verify ──► finalize → END
                          (repair done inline in verify node; always exits to finalize)
"""
from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.graph.nodes import (
    clarify,
    contextualize,
    decompose,
    finalize,
    grade_documents,
    retrieve,
    route_question,
    synthesize,
    transform_query,
    verify,
    web_search,
)
from app.graph.state import ResearchState

log = logging.getLogger(__name__)
_cfg = get_settings()

# ─── Conditional edge functions ───────────────────────────────────────────────


def _route_after_router(
    state: ResearchState,
) -> Literal["clarify", "decompose", "retrieve", "synthesize"]:
    route = state.get("route", "simple")
    if route == "chitchat":
        return "synthesize"
    if route == "ambiguous":
        return "clarify"
    if route == "compound":
        return "decompose"
    return "retrieve"  # simple


def _route_after_grade(
    state: ResearchState,
) -> Literal["advance_sq", "transform_query", "web_search"]:
    if state.get("graded_relevant", False):
        return "advance_sq"  # relevant — check for more sub-questions
    retry = state.get("retry_count", 0)
    if retry < _cfg.max_retries:
        return "transform_query"
    return "web_search"


def _route_after_transform(state: ResearchState) -> Literal["retrieve"]:
    return "retrieve"


def _route_after_verify(
    state: ResearchState,
) -> Literal["finalize"]:
    # The verify node already does one inline repair when grounded=False.
    # Always exit to finalize — routing back to synthesize creates an infinite
    # loop because verify will keep returning grounded=False on the same evidence.
    return "finalize"


def _route_after_retrieve(
    state: ResearchState,
) -> Literal["grade_documents"]:
    # Always grade after retrieve
    return "grade_documents"


def _maybe_next_sub_question(
    state: ResearchState,
) -> Literal["retrieve", "synthesize"]:
    sqs = state.get("sub_questions") or []
    idx = state.get("current_sq_index", 0)
    next_idx = idx + 1

    if next_idx < len(sqs):
        # More sub-questions to process
        return "retrieve"
    return "synthesize"


# ─── Build ────────────────────────────────────────────────────────────────────

def _next_sq_retrieve(state: ResearchState) -> dict:
    """Advance the sub-question index before the next retrieve loop."""
    idx = state.get("current_sq_index", 0)
    return {"current_sq_index": idx + 1, "retry_count": 0, "graded_relevant": False}


def build_graph() -> StateGraph:
    g = StateGraph(ResearchState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    g.add_node("contextualize", contextualize)
    g.add_node("route_question", route_question)
    g.add_node("clarify", clarify)
    g.add_node("decompose", decompose)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("transform_query", transform_query)
    g.add_node("web_search", web_search)
    g.add_node("synthesize", synthesize)
    g.add_node("verify", verify)
    g.add_node("finalize", finalize)
    g.add_node("advance_sq", _next_sq_retrieve)  # index bumper

    # ── Edges ──────────────────────────────────────────────────────────────
    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "route_question")

    g.add_conditional_edges(
        "route_question",
        _route_after_router,
        {
            "clarify": "clarify",
            "decompose": "decompose",
            "retrieve": "retrieve",
            "synthesize": "synthesize",
        },
    )

    # Clarify → finalize (answer is already in draft_answer)
    g.add_edge("clarify", "finalize")

    # Decompose → retrieve (index starts at 0)
    g.add_edge("decompose", "retrieve")

    # Retrieve → grade
    g.add_edge("retrieve", "grade_documents")

    # Grade → branch
    # NOTE: _route_after_grade now returns "advance_sq" (not "synthesize") for
    # the relevant case, so the mapping key must match.
    g.add_conditional_edges(
        "grade_documents",
        _route_after_grade,
        {
            "advance_sq": "advance_sq",     # relevant: check for more sub-Qs
            "transform_query": "transform_query",
            "web_search": "web_search",
        },
    )

    # After advancing sub-question index: synthesize or retrieve next
    g.add_conditional_edges(
        "advance_sq",
        _maybe_next_sub_question,
        {
            "retrieve": "retrieve",
            "synthesize": "synthesize",
        },
    )

    g.add_edge("transform_query", "retrieve")

    # FIX: web_search goes DIRECTLY to synthesize — do NOT re-grade web results.
    # Routing back through grade_documents creates:  web_search → grade → web_search (∞ loop)
    # because retry_count is already ≥ max_retries when we reach web_search.
    g.add_edge("web_search", "synthesize")

    g.add_edge("synthesize", "verify")

    # FIX: verify always exits to finalize.
    # The inline repair already happened inside the verify node.
    # Routing back to synthesize creates: synthesize → verify → synthesize (∞ loop)
    # because the same insufficient evidence produces the same ungrounded answer.
    g.add_edge("verify", "finalize")

    g.add_edge("finalize", END)

    return g


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        g = build_graph()
        _compiled_graph = g.compile()
        log.info("LangGraph research graph compiled")
    return _compiled_graph


def reset_compiled_graph() -> None:
    """Force recompile — call after code changes during development."""
    global _compiled_graph
    _compiled_graph = None
