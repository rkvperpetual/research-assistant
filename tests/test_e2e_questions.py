"""
tests/test_e2e_questions.py — End-to-end smoke tests against the running API.

These tests require a running server (localhost:8000) and a seeded knowledge base.
Run with:
    pytest tests/test_e2e_questions.py -v -s

Set E2E_API_URL env var to point at a different host (e.g. the deployed URL).
"""
from __future__ import annotations

import os
import pytest
import httpx

BASE_URL = os.getenv("E2E_API_URL", "http://localhost:8000")

pytestmark = pytest.mark.e2e  # tag so you can skip with: pytest -m "not e2e"


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=120) as c:
        yield c


def ask(client: httpx.Client, question: str, conv_id: str | None = None) -> dict:
    payload = {"question": question, "include_trace": True}
    if conv_id:
        payload["conversation_id"] = conv_id
    resp = client.post("/ask", json=payload)
    resp.raise_for_status()
    return resp.json()


# ── Benchmark tests ───────────────────────────────────────────────────────────

class TestBenchmark:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["qdrant_reachable"]

    def test_q1_compound_eu_nist(self, client):
        """Compound: requires 2+ retrievals and synthesis."""
        data = ask(client, "Compare how the EU AI Act and the NIST RMF each handle risk classification, and say which one is more prescriptive.")
        assert data["answer"], "Expected a non-empty answer"
        assert data["route"] in ("compound", "simple"), f"Got route={data['route']}"
        assert len(data["trace"]) >= 3, "Expected at least 3 trace steps"

    def test_q2_ambiguous_leave(self, client):
        """Ambiguous: under-specified question."""
        data = ask(client, "What's the policy on leave?")
        assert data["answer"]
        # Should either clarify or cover all interpretations
        assert len(data["answer"]) > 20

    def test_q5_not_in_kb_triggers_web_search(self, client):
        """Out-of-KB: should trigger web search or admit ignorance."""
        data = ask(client, "Who is the current CEO of Anthropic?")
        assert data["answer"]
        # Should use web search or admit it doesn't know
        either_web_or_honest = (
            data["used_web_search"]
            or "knowledge base" in data["answer"].lower()
            or "don't" in data["answer"].lower()
            or "cannot" in data["answer"].lower()
        )
        assert either_web_or_honest

    def test_q7_trap_unsupported(self, client):
        """Trap: nothing in KB, should not invent an answer."""
        data = ask(client, "What is Autom8AI's refund policy?")
        assert data["answer"]
        honest = (
            "knowledge base" in data["answer"].lower()
            or "not cover" in data["answer"].lower()
            or "not supported" in data["answer"].lower()
            or "cannot" in data["answer"].lower()
            or data["confidence"] == "low"
        )
        assert honest, f"Expected honest 'not supported' response, got: {data['answer'][:200]}"

    def test_conversation_followup(self, client):
        """Follow-up: pronoun resolution across turns."""
        data1 = ask(client, "What's the policy on leave?")
        conv_id = data1["conversation_id"]

        data2 = ask(client, "And what about parental leave?", conv_id=conv_id)
        assert data2["conversation_id"] == conv_id
        assert data2["answer"]

    def test_trace_has_enough_steps_on_compound(self, client):
        """Compound question must show ≥ 4 distinct nodes in the trace."""
        data = ask(client, "Compare how the EU AI Act and the NIST RMF each handle risk classification, and say which is more prescriptive.")
        nodes = [s["node"] for s in data.get("trace", [])]
        assert len(set(nodes)) >= 4, f"Expected ≥4 distinct nodes, got: {nodes}"
