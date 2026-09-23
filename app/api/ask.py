"""
app/api/ask.py — POST /ask endpoint.

Runs the LangGraph research agent, persists the conversation turn,
and returns a structured AskResponse including the reasoning trace.
"""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException

from app.api.schemas import AskRequest, AskResponse, CitationOut, TraceStepOut
from app.config import get_settings
from app.core.memory import append_turn, load_history, new_conversation_id
from app.graph.builder import get_compiled_graph
from app.graph.state import ResearchState

log = logging.getLogger(__name__)
_cfg = get_settings()

router = APIRouter(tags=["ask"])


def _confidence(state: ResearchState) -> str:
    """Heuristic confidence label based on grounding + web search usage."""
    grounded = state.get("grounded", True)
    used_web = state.get("used_web_search", False)
    evidence = state.get("evidence", [])

    if grounded and not used_web and len(evidence) >= 3:
        return "high"
    if grounded:
        return "medium"
    return "low"


@router.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest):
    """
    Main research endpoint.
    Runs the multi-step LangGraph agent and returns the answer with trace.
    """
    t_start = time.perf_counter()

    # ── Conversation setup ────────────────────────────────────────────────
    conv_id = payload.conversation_id or new_conversation_id()
    history = load_history(conv_id)

    # ── Build initial state ───────────────────────────────────────────────
    initial_state: ResearchState = {
        "question": payload.question,
        "conversation_id": conv_id,
        "include_trace": payload.include_trace,
        "standalone_question": payload.question,
        "chat_history": history,
        "route": "simple",
        "sub_questions": [],
        "current_sq_index": 0,
        "evidence": [],
        "graded_relevant": False,
        "retry_count": 0,
        "used_web_search": False,
        "web_results": [],
        "draft_answer": "",
        "grounded": True,
        "answer": "",
        "citations": [],
        "trace": [],
        "start_ms": t_start,
    }

    # ── Run the graph ─────────────────────────────────────────────────────
    try:
        graph = get_compiled_graph()
        final_state: ResearchState = graph.invoke(
            initial_state,
            config={"recursion_limit": _cfg.graph_recursion_limit},
        )
    except Exception as e:
        log.exception("Graph execution failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    # ── Persist turn ──────────────────────────────────────────────────────
    append_turn(conv_id, "user", payload.question)
    append_turn(conv_id, "assistant", final_state.get("answer", ""))

    # ── Build response ────────────────────────────────────────────────────
    latency_ms = int((time.perf_counter() - t_start) * 1000)

    citations = [
        CitationOut(
            id=c["id"],
            source=c["source"],
            page=c.get("page"),
            snippet=c["snippet"],
            score=c["score"],
            origin=c["origin"],
        )
        for c in final_state.get("citations", [])
    ]

    trace_out = (
        [
            TraceStepOut(step=s["step"], node=s["node"], detail=s["detail"], ms=s["ms"])
            for s in final_state.get("trace", [])
        ]
        if payload.include_trace
        else []
    )

    return AskResponse(
        answer=final_state.get("answer", "No answer generated"),
        conversation_id=conv_id,
        route=final_state.get("route", "simple"),
        sub_questions=final_state.get("sub_questions", []),
        citations=citations,
        used_web_search=final_state.get("used_web_search", False),
        confidence=_confidence(final_state),
        grounded=final_state.get("grounded", True),
        trace=trace_out,
        latency_ms=latency_ms,
    )


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Return the full turn history for a conversation."""
    from app.core.memory import get_conversation
    turns = get_conversation(conversation_id)
    if not turns:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "turns": turns}
