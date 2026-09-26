"""
app/graph/nodes.py — One function per LangGraph node.

Every node takes ResearchState and returns a partial state dict.
The trace list is accumulated via the Annotated[list, operator.add] reducer.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.core.llm import get_llm, get_synthesis_llm
from app.graph.prompts import (
    CLARIFY_SYSTEM,
    CONTEXTUALISE_SYSTEM,
    DECOMPOSE_SYSTEM,
    GRADE_SYSTEM,
    REPAIR_SYSTEM,
    ROUTE_SYSTEM,
    SYNTHESIZE_EVIDENCE_TEMPLATE,
    SYNTHESIZE_SYSTEM,
    TRANSFORM_SYSTEM,
    VERIFY_SYSTEM,
)
from app.graph.state import Citation, Evidence, ResearchState, TraceStep, make_trace_step
from app.retrieval.hybrid import hybrid_retrieve

log = logging.getLogger(__name__)
_cfg = get_settings()

_RELEVANCE_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "did", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "new", "of", "on",
    "or", "the", "their", "this", "to", "was", "what", "when", "where", "who",
    "why", "with", "would", "you", "your",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_text(msg) -> str:
    """Safely extract text from AIMessage, handling providers that return lists."""
    content = msg.content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts).strip()
    return str(content).strip()


def _llm_json(system: str, user: str) -> dict:
    """Call the LLM expecting JSON output. Parse and return the dict."""
    llm = get_llm(temperature=0.0)
    msgs = [SystemMessage(content=system), HumanMessage(content=user)]
    msg = llm.invoke(msgs)
    raw = _extract_text(msg)
    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(raw)


def _step_num(state: ResearchState) -> int:
    return len(state.get("trace", [])) + 1


def _t0(state: ResearchState) -> float:
    return state.get("start_ms", time.perf_counter())


def _meaningful_terms(text: str) -> set[str]:
    """Return lexical signals used to reject clearly unrelated retrieval hits."""
    return {
        token for token in re.findall(r"[a-z0-9]{3,}", text.lower())
        if token not in _RELEVANCE_STOP_WORDS
    }


# ─── Node 1: contextualize ────────────────────────────────────────────────────

def contextualize(state: ResearchState) -> dict:
    t = time.perf_counter()
    history = state.get("chat_history", [])
    question = state["question"]

    if not history:
        detail = "No history; question used as-is"
        sq = question
    else:
        history_str = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        prompt = f"Conversation history:\n{history_str}\n\nFollow-up question: {question}"
        try:
            llm = get_llm(temperature=0.0)
            msg = llm.invoke(
                [SystemMessage(content=CONTEXTUALISE_SYSTEM), HumanMessage(content=prompt)]
            )
            sq = _extract_text(msg)
            detail = f"Rewrote to: {sq[:80]}…"
        except Exception as e:
            log.warning("Contextualization failed; using follow-up as-is: %s", e)
            sq = question
            detail = "Contextualization unavailable; question used as-is"

    step = make_trace_step(_step_num(state), "contextualize", detail, t)
    return {"standalone_question": sq, "trace": [step], "start_ms": state.get("start_ms", t)}


# ─── Node 2: route_question ───────────────────────────────────────────────────

def route_question(state: ResearchState) -> dict:
    t = time.perf_counter()
    sq = state["standalone_question"]
    try:
        result = _llm_json(ROUTE_SYSTEM, f"Question: {sq}")
        route = result.get("route", "simple")
    except Exception as e:
        log.warning("Route parsing failed (%s); defaulting to simple", e)
        route = "simple"

    step = make_trace_step(_step_num(state), "route_question", f"→ {route}", t)
    return {"route": route, "trace": [step]}


# ─── Node 3: clarify ──────────────────────────────────────────────────────────

def clarify(state: ResearchState) -> dict:
    t = time.perf_counter()
    sq = state["standalone_question"]
    llm = get_synthesis_llm()
    msg = llm.invoke(
        [SystemMessage(content=CLARIFY_SYSTEM), HumanMessage(content=sq)]
    )
    raw = _extract_text(msg)

    if "---CLARIFICATION---" in raw:
        clarifying_q, best_effort = raw.split("---CLARIFICATION---", 1)
        answer = f"**Clarifying question:** {clarifying_q.strip()}\n\n**Best-effort answer:** {best_effort.strip()}"
    else:
        answer = raw.strip()

    step = make_trace_step(_step_num(state), "clarify", "Returned clarifying question + best-effort answer", t)
    return {"draft_answer": answer, "trace": [step]}


# ─── Node 4: decompose ────────────────────────────────────────────────────────

def decompose(state: ResearchState) -> dict:
    t = time.perf_counter()
    sq = state["standalone_question"]
    try:
        result = _llm_json(DECOMPOSE_SYSTEM, f"Question: {sq}")
        sqs = result.get("sub_questions", [sq])[:_cfg.max_sub_questions]
    except Exception:
        sqs = [sq]

    if not sqs:
        sqs = [sq]

    step = make_trace_step(_step_num(state), "decompose", f"{len(sqs)} sub-questions", t)
    return {
        "sub_questions": sqs,
        "current_sq_index": 0,
        "retry_count": 0,
        "trace": [step],
    }


# ─── Node 5: retrieve ─────────────────────────────────────────────────────────

def retrieve(state: ResearchState) -> dict:
    t = time.perf_counter()
    sqs = state.get("sub_questions") or [state["standalone_question"]]
    idx = state.get("current_sq_index", 0)
    query = sqs[idx] if idx < len(sqs) else state["standalone_question"]

    docs = hybrid_retrieve(query)

    evidence: List[Evidence] = []
    for doc in docs:
        meta = doc.metadata or {}
        evidence.append(
            Evidence(
                text=doc.page_content,
                source=meta.get("filename", "unknown"),
                page=meta.get("page"),
                chunk_index=meta.get("chunk_index"),
                score=float(meta.get("score", 0.0)),
                origin="knowledge_base",
                sub_question_index=idx,
            )
        )

    step = make_trace_step(
        _step_num(state), "retrieve",
        f"sq{idx+1}: {len(docs)} chunks after RRF", t
    )
    return {"evidence": evidence, "trace": [step]}


# ─── Node 6: grade_documents ──────────────────────────────────────────────────

def grade_documents(state: ResearchState) -> dict:
    t = time.perf_counter()
    sqs = state.get("sub_questions") or [state["standalone_question"]]
    idx = state.get("current_sq_index", 0)
    query = sqs[idx] if idx < len(sqs) else state["standalone_question"]

    # Only grade evidence for current sub-question
    current_evidence = [e for e in state.get("evidence", []) if e["sub_question_index"] == idx]

    if not current_evidence:
        step = make_trace_step(_step_num(state), "grade_documents", "No evidence to grade", t)
        return {"graded_relevant": False, "trace": [step]}

    # Vector search always returns nearest neighbours, even when every result is
    # unrelated.  Avoid treating those distractors as KB evidence just because
    # an LLM grader is unavailable or permissive; route such questions to web.
    query_terms = _meaningful_terms(query)
    evidence_terms = _meaningful_terms(" ".join(e["text"] for e in current_evidence))
    overlap = query_terms & evidence_terms
    if query_terms and not overlap:
        step = make_trace_step(
            _step_num(state), "grade_documents",
            "No meaningful query-term overlap; using web fallback", t,
        )
        return {"graded_relevant": False, "trace": [step]}

    evidence_text = "\n\n".join(
        f"[{i+1}] {e['text'][:400]}" for i, e in enumerate(current_evidence)
    )
    prompt = f"Question: {query}\n\nRetrieved chunks:\n{evidence_text}"

    try:
        result = _llm_json(GRADE_SYSTEM, prompt)
        relevant = bool(result.get("relevant", False))
        reason = result.get("reason", "")
    except Exception as e:
        log.warning("Grading failed: %s", e)
        relevant = True  # fail open
        reason = "grading error — defaulting to relevant"

    relevant_count = len(current_evidence) if relevant else 0
    step = make_trace_step(
        _step_num(state), "grade_documents",
        f"{relevant_count}/{len(current_evidence)} relevant — {reason[:60]}", t
    )
    return {"graded_relevant": relevant, "trace": [step]}


# ─── Node 7: transform_query ─────────────────────────────────────────────────

def transform_query(state: ResearchState) -> dict:
    t = time.perf_counter()
    sqs = state.get("sub_questions") or [state["standalone_question"]]
    idx = state.get("current_sq_index", 0)
    query = sqs[idx] if idx < len(sqs) else state["standalone_question"]

    try:
        llm = get_llm(temperature=0.3)
        msg = llm.invoke(
            [SystemMessage(content=TRANSFORM_SYSTEM), HumanMessage(content=query)]
        )
        new_query = _extract_text(msg)
    except Exception as e:
        log.warning("Query rewrite failed; retrying the original query: %s", e)
        new_query = query

    # Update the sub-question list with the rewritten query
    updated_sqs = list(sqs)
    if idx < len(updated_sqs):
        updated_sqs[idx] = new_query

    retry_count = state.get("retry_count", 0) + 1
    step = make_trace_step(
        _step_num(state), "transform_query",
        f"Rewrite #{retry_count}: {new_query[:60]}…", t
    )
    return {
        "sub_questions": updated_sqs,
        "retry_count": retry_count,
        "trace": [step],
    }


# ─── Node 8: web_search ───────────────────────────────────────────────────────

def web_search(state: ResearchState) -> dict:
    t = time.perf_counter()
    sqs = state.get("sub_questions") or [state["standalone_question"]]
    idx = state.get("current_sq_index", 0)
    query = sqs[idx] if idx < len(sqs) else state["standalone_question"]

    web_evidence: List[Evidence] = []
    try:
        import httpx
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={
                "query": query,
                "api_key": _cfg.tavily_api_key,
                "search_depth": "basic",
                "max_results": 4,
            },
            timeout=15.0
        )
        resp.raise_for_status()
        data = resp.json()
        
        for doc in data.get("results", []):
            web_evidence.append(
                Evidence(
                    text=doc.get("content", "")[:800],
                    source=doc.get("url", "web"),
                    page=None,
                    chunk_index=None,
                    score=doc.get("score", 0.0),
                    origin="web_search",
                    sub_question_index=idx,
                )
            )
    except Exception as e:
        log.warning("Tavily search failed: %s", e)

    step = make_trace_step(
        _step_num(state), "web_search",
        f"Tavily returned {len(web_evidence)} results for: {query[:50]}…", t
    )
    return {
        "evidence": web_evidence,
        "used_web_search": True,
        "trace": [step],
    }


# ─── Node 9: synthesize ───────────────────────────────────────────────────────

def synthesize(state: ResearchState) -> dict:
    t = time.perf_counter()
    question = state["standalone_question"]
    all_evidence: List[Evidence] = state.get("evidence", [])

    if not all_evidence:
        step = make_trace_step(_step_num(state), "synthesize", "No evidence — returning not-supported message", t)
        return {
            "draft_answer": "The knowledge base does not support this question, and no relevant web results were found.",
            "trace": [step],
        }

    evidence_block = "\n\n".join(
        f"[{i+1}] (source: {e['source']}, origin: {e['origin']})\n{e['text'][:600]}"
        for i, e in enumerate(all_evidence)
    )
    prompt = SYNTHESIZE_EVIDENCE_TEMPLATE.format(
        question=question, evidence_block=evidence_block
    )

    synthesis_available = True
    try:
        llm = get_synthesis_llm()
        msg = llm.invoke(
            [SystemMessage(content=SYNTHESIZE_SYSTEM), HumanMessage(content=prompt)]
        )
        answer = _extract_text(msg)
    except Exception as e:
        log.warning("Synthesis failed: %s", e)
        synthesis_available = False
        web_evidence = [e for e in all_evidence if e["origin"] == "web_search"]
        if web_evidence:
            snippets = "\n".join(
                f"- {e['source']}: {e['text'][:500]}" for e in web_evidence[:2]
            )
            answer = (
                "The web search returned the following information, but the AI provider is "
                "currently unavailable to synthesize it:\n" + snippets
            )
        else:
            answer = (
                "I cannot generate a grounded answer right now because the AI provider is "
                "unavailable. Please retry after the provider is available."
            )

    step = make_trace_step(
        _step_num(state), "synthesize",
        f"Generated answer from {len(all_evidence)} evidence chunks", t
    )
    return {"draft_answer": answer, "grounded": synthesis_available, "trace": [step]}


# ─── Node 10: verify ─────────────────────────────────────────────────────────

def verify(state: ResearchState) -> dict:
    t = time.perf_counter()
    draft = state.get("draft_answer", "")
    all_evidence: List[Evidence] = state.get("evidence", [])

    evidence_block = "\n\n".join(
        f"[{i+1}] {e['text'][:400]}" for i, e in enumerate(all_evidence)
    )
    prompt = f"Draft answer:\n{draft}\n\nEvidence:\n{evidence_block}"
    try:
        result = _llm_json(VERIFY_SYSTEM, prompt)
        grounded = bool(result.get("grounded", True))
        unsupported = result.get("unsupported_claims", [])
    except Exception as e:
        log.warning("Verify parsing failed: %s", e)
        grounded = True
        unsupported = []

    step = make_trace_step(
        _step_num(state), "verify",
        f"grounded={grounded}" + (f"; unsupported: {unsupported[:2]}" if not grounded else ""), t
    )

    if not grounded and unsupported:
        # One repair attempt
        llm = get_synthesis_llm()
        repair_prompt = (
            f"Draft answer:\n{draft}\n\n"
            f"Unsupported claims to remove/flag: {json.dumps(unsupported)}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            "Rewrite the answer:"
        )
        repaired_msg = llm.invoke(
            [SystemMessage(content=REPAIR_SYSTEM), HumanMessage(content=repair_prompt)]
        )
        repaired = _extract_text(repaired_msg)
        return {"draft_answer": repaired, "grounded": False, "trace": [step]}

    return {"grounded": grounded, "trace": [step]}


# ─── Node 11: finalize ───────────────────────────────────────────────────────

def finalize(state: ResearchState) -> dict:
    t = time.perf_counter()
    draft = state.get("draft_answer", "")
    all_evidence: List[Evidence] = state.get("evidence", [])

    citations: List[Citation] = []
    for i, e in enumerate(all_evidence):
        citations.append(
            Citation(
                id=i + 1,
                source=e["source"],
                page=e.get("page"),
                snippet=e["text"][:120] + "…",
                score=e["score"],
                origin=e["origin"],
            )
        )

    step = make_trace_step(
        _step_num(state), "finalize",
        f"{len(citations)} citations attached", t
    )
    return {
        "answer": draft,
        "citations": citations,
        "trace": [step],
    }
