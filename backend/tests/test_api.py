"""Tests for the FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import SearchResponse


def _patch_memory_paths(monkeypatch, tmp_path: Path):
    from app.services import memory_store

    monkeypatch.setattr(memory_store, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(memory_store, "TOPICS_FILE", tmp_path / "user_topics.json")
    monkeypatch.setattr(memory_store, "SAVED_PAPERS_FILE", tmp_path / "saved_papers.json")
    monkeypatch.setattr(memory_store, "HISTORY_FILE", tmp_path / "reading_history.json")
    monkeypatch.setattr(memory_store, "CHAT_SESSIONS_FILE", tmp_path / "chat_sessions.json")
    return memory_store


@pytest.fixture(autouse=True)
def isolated_memory(monkeypatch, tmp_path):
    _patch_memory_paths(monkeypatch, tmp_path)


@pytest.fixture()
def client():
    """FastAPI TestClient for synchronous endpoint testing."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        """Health check should return 200 with status healthy."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "paper-radar-agent"

    def test_health_response_format(self, client):
        """Health response should have exactly the expected keys."""
        response = client.get("/health")
        data = response.json()

        assert set(data.keys()) == {"status", "service"}


class TestProviderEndpoint:
    def test_provider_status_returns_current_mode(self, client):
        response = client.get("/api/provider")

        assert response.status_code == 200
        data = response.json()
        assert data["provider"]
        assert data["mode"] in {"mock", "real"}
        assert "label" in data


class TestMemoryEndpoints:
    def test_topics_post_get_and_broken_json_fallback(self, client, monkeypatch, tmp_path):
        _patch_memory_paths(monkeypatch, tmp_path)

        response = client.get("/api/memory/topics")
        assert response.status_code == 200
        assert response.json()["topics"] == []
        assert (tmp_path / "user_topics.json").exists()

        response = client.post(
            "/api/memory/topics",
            json={"text": "LLM Agent 长期记忆机制", "topics": ["Agentic RAG"]},
        )
        assert response.status_code == 200
        assert "Agentic RAG" in response.json()["topics"]

        (tmp_path / "user_topics.json").write_text("{broken", encoding="utf-8")
        response = client.get("/api/memory/topics")
        assert response.status_code == 200
        assert response.json()["topics"] == []

    def test_saved_papers_and_history(self, client, monkeypatch, tmp_path):
        memory_store = _patch_memory_paths(monkeypatch, tmp_path)

        response = client.post(
            "/api/memory/saved-papers",
            json={
                "title": "Memory-Augmented Language Agents",
                "url": "https://example.com/paper",
                "source": "arxiv",
            },
        )
        assert response.status_code == 200

        response = client.get("/api/memory/saved-papers")
        assert response.status_code == 200
        assert len(response.json()["papers"]) == 1

        memory_store.add_history({"query": "paper radar"})
        response = client.get("/api/memory/history")
        assert response.status_code == 200
        assert response.json()["history"][0]["query"] == "paper radar"


class TestSearchEndpoint:
    """Tests for POST /api/search."""

    @patch("app.agents.graph.run_search", new_callable=AsyncMock)
    def test_search_returns_response(self, mock_run_search, client):
        """A valid search request should return a SearchResponse."""
        mock_run_search.return_value = SearchResponse(
            query="transformer architecture",
            answer="Transformers use self-attention mechanisms [1].",
            citations=[
                {"index": 1, "title": "Attention Is All You Need", "url": "https://arxiv.org/abs/1706.03762"}
            ],
            papers=[
                {
                    "title": "Attention Is All You Need",
                    "authors": ["Ashish Vaswani"],
                    "abstract": "We propose the Transformer architecture.",
                    "url": "https://arxiv.org/abs/1706.03762",
                    "source": "arxiv",
                    "published": "2017-06-12",
                }
            ],
            steps=[
                {"node": "router", "status": "completed", "detail": "paper_search", "duration_ms": 50}
            ],
            rewrite_count=0,
        )

        response = client.post(
            "/api/search",
            json={"query": "transformer architecture", "sources": ["arxiv"], "max_results": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "transformer architecture"
        assert "Transformers" in data["answer"]
        assert len(data["citations"]) == 1
        assert len(data["papers"]) == 1
        assert data["rewrite_count"] == 0
        assert data["session_id"]

    @patch("app.agents.graph.run_search", new_callable=AsyncMock)
    def test_search_calls_run_search_with_correct_args(self, mock_run_search, client):
        """run_search should be called with the request parameters."""
        mock_run_search.return_value = SearchResponse(
            query="BERT",
            answer="BERT is a language model.",
        )

        client.post(
            "/api/search",
            json={"query": "BERT", "sources": ["arxiv", "pubmed"], "max_results": 15},
        )

        mock_run_search.assert_called_once_with(
            query="BERT",
            sources=["arxiv", "pubmed"],
            max_results=15,
            session_id=None,
        )

    def test_search_empty_query_returns_422(self, client):
        """An empty query string should return 422 validation error."""
        response = client.post(
            "/api/search",
            json={"query": "", "sources": ["arxiv"]},
        )

        assert response.status_code == 422

    def test_search_missing_query_returns_422(self, client):
        """A request without a query field should return 422."""
        response = client.post(
            "/api/search",
            json={"sources": ["arxiv"]},
        )

        assert response.status_code == 422

    @patch("app.agents.graph.run_search", new_callable=AsyncMock)
    def test_search_default_sources(self, mock_run_search, client):
        """When sources are not specified, defaults should be used."""
        mock_run_search.return_value = SearchResponse(
            query="test",
            answer="test answer",
        )

        response = client.post(
            "/api/search",
            json={"query": "test query"},
        )

        assert response.status_code == 200
        call_kwargs = mock_run_search.call_args
        assert call_kwargs.kwargs["sources"] == ["arxiv", "pubmed", "openalex"]

    def test_search_max_results_out_of_range(self, client):
        """max_results outside 1-99 should return 422."""
        response = client.post(
            "/api/search",
            json={"query": "test", "max_results": 100},
        )
        assert response.status_code == 422


class TestChatSessionEndpoints:
    def test_create_list_get_and_delete_chat_session(self, client):
        response = client.post("/api/chat/sessions", json={"title": "长期记忆学习"})
        assert response.status_code == 200
        session = response.json()

        response = client.get("/api/chat/sessions")
        assert response.status_code == 200
        assert response.json()["sessions"][0]["title"] == "长期记忆学习"

        response = client.get(f"/api/chat/sessions/{session['id']}")
        assert response.status_code == 200
        assert response.json()["id"] == session["id"]

        response = client.delete(f"/api/chat/sessions/{session['id']}")
        assert response.status_code == 200
        assert response.json()["deleted"] is True

        response = client.post(
            "/api/search",
            json={"query": "test", "max_results": 0},
        )
        assert response.status_code == 422


class TestCORSMiddleware:
    """Test that CORS headers are present."""

    def test_cors_allows_localhost(self, client):
        """The frontend origin should be allowed."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # FastAPI CORS middleware should respond to preflight
        assert response.status_code in (200, 204)
