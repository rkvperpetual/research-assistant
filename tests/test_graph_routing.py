"""
tests/test_graph_routing.py — Unit tests for the graph routing node.

Mocks the LLM so no API calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_state(question: str, history=None) -> dict:
    return {
        "question": question,
        "standalone_question": question,
        "conversation_id": "test-conv",
        "include_trace": True,
        "chat_history": history or [],
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
        "start_ms": 0.0,
    }


class TestRouting:
    """Test that route_question classifies questions correctly."""

    def _mock_llm_route(self, route: str):
        mock_response = MagicMock()
        mock_response.content = f'{{"route": "{route}"}}'
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    @patch("app.graph.nodes.get_llm")
    def test_simple_question(self, mock_get_llm):
        mock_get_llm.return_value = self._mock_llm_route("simple")
        from app.graph.nodes import route_question

        state = _make_state("What is the EU AI Act?")
        result = route_question(state)
        assert result["route"] == "simple"

    @patch("app.graph.nodes.get_llm")
    def test_compound_question(self, mock_get_llm):
        mock_get_llm.return_value = self._mock_llm_route("compound")
        from app.graph.nodes import route_question

        state = _make_state(
            "Compare how the EU AI Act and NIST RMF handle risk classification, "
            "and say which is more prescriptive."
        )
        result = route_question(state)
        assert result["route"] == "compound"

    @patch("app.graph.nodes.get_llm")
    def test_ambiguous_question(self, mock_get_llm):
        mock_get_llm.return_value = self._mock_llm_route("ambiguous")
        from app.graph.nodes import route_question

        state = _make_state("What's the policy on leave?")
        result = route_question(state)
        assert result["route"] == "ambiguous"

    @patch("app.graph.nodes.get_llm")
    def test_chitchat(self, mock_get_llm):
        mock_get_llm.return_value = self._mock_llm_route("chitchat")
        from app.graph.nodes import route_question

        state = _make_state("Hello, how are you?")
        result = route_question(state)
        assert result["route"] == "chitchat"

    @patch("app.graph.nodes.get_llm")
    def test_route_bad_json_defaults_to_simple(self, mock_get_llm):
        """Router should fail open to 'simple' if LLM returns invalid JSON."""
        mock_response = MagicMock()
        mock_response.content = "I cannot classify this."
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        from app.graph.nodes import route_question

        state = _make_state("Some question")
        result = route_question(state)
        assert result["route"] == "simple"


class TestDecompose:
    @patch("app.graph.nodes.get_llm")
    def test_decompose_splits_question(self, mock_get_llm):
        mock_response = MagicMock()
        mock_response.content = '{"sub_questions": ["Q1?", "Q2?"]}'
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        from app.graph.nodes import decompose

        state = _make_state("Compound question with two parts")
        result = decompose(state)
        assert len(result["sub_questions"]) == 2
        assert result["current_sq_index"] == 0

    @patch("app.graph.nodes.get_llm")
    def test_decompose_caps_at_max(self, mock_get_llm):
        mock_response = MagicMock()
        # Return 6 sub-questions (should be capped at 4)
        sqs = [f"Q{i}?" for i in range(6)]
        mock_response.content = f'{{"sub_questions": {sqs}}}'
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        from app.graph.nodes import decompose

        state = _make_state("Complex question")
        result = decompose(state)
        assert len(result["sub_questions"]) <= 4


class TestProviderFallbacks:
    @patch("app.graph.nodes.get_synthesis_llm")
    def test_synthesize_returns_fallback_when_provider_fails(self, mock_get_llm):
        mock_get_llm.return_value.invoke.side_effect = RuntimeError("provider unavailable")
        from app.graph.nodes import synthesize

        state = _make_state("What is the leave policy?")
        state["evidence"] = [{
            "text": "Leave policy evidence",
            "source": "handbook.md",
            "page": None,
            "chunk_index": 0,
            "score": 1.0,
            "origin": "knowledge_base",
            "sub_question_index": 0,
        }]
        result = synthesize(state)

        assert "cannot generate" in result["draft_answer"].lower()
        assert result["grounded"] is False

    @patch("app.graph.nodes.get_synthesis_llm")
    def test_synthesize_returns_web_evidence_when_provider_fails(self, mock_get_llm):
        mock_get_llm.return_value.invoke.side_effect = RuntimeError("provider unavailable")
        from app.graph.nodes import synthesize

        state = _make_state("Who is the CEO?")
        state["evidence"] = [{
            "text": "Example Corp named Jane Doe as its CEO.",
            "source": "https://example.com/news",
            "page": None,
            "chunk_index": None,
            "score": 0.0,
            "origin": "web_search",
            "sub_question_index": 0,
        }]
        result = synthesize(state)

        assert "Jane Doe" in result["draft_answer"]
        assert result["grounded"] is False


class TestRelevanceGuard:
    @patch("app.graph.nodes.get_llm")
    def test_unrelated_evidence_routes_to_web_without_llm_grading(self, mock_get_llm):
        from app.graph.nodes import grade_documents

        state = _make_state("Who is the new CEO of Apple?")
        state["evidence"] = [{
            "text": "EU AI Act penalties and engineering meeting notes.",
            "source": "unrelated.md",
            "page": None,
            "chunk_index": 0,
            "score": 0.1,
            "origin": "knowledge_base",
            "sub_question_index": 0,
        }]
        result = grade_documents(state)

        assert result["graded_relevant"] is False
        assert "web fallback" in result["trace"][0]["detail"]
        mock_get_llm.assert_not_called()


class TestRRF:
    def test_rrf_ranks_known_result_top(self):
        from langchain_core.documents import Document
        from app.retrieval.hybrid import _rrf_fuse

        dense = [
            Document(page_content="EU AI Act risk classification", metadata={}),
            Document(page_content="NIST RMF framework overview", metadata={}),
        ]
        sparse = [
            ("NIST RMF framework overview", 5.0),
            ("Something unrelated", 2.0),
        ]
        fused = _rrf_fuse(dense, sparse, top_k=3)
        # NIST doc should be boosted by both lists
        assert any("NIST" in d.page_content for d in fused[:2])
