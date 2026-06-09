"""Tests for the lightweight JSON memory store."""

from __future__ import annotations

from pathlib import Path


def _patch_memory_paths(monkeypatch, tmp_path: Path):
    from app.services import memory_store

    monkeypatch.setattr(memory_store, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(memory_store, "TOPICS_FILE", tmp_path / "user_topics.json")
    monkeypatch.setattr(memory_store, "SAVED_PAPERS_FILE", tmp_path / "saved_papers.json")
    monkeypatch.setattr(memory_store, "HISTORY_FILE", tmp_path / "reading_history.json")
    monkeypatch.setattr(memory_store, "CHAT_SESSIONS_FILE", tmp_path / "chat_sessions.json")
    return memory_store


def test_topics_handle_missing_and_broken_json(monkeypatch, tmp_path):
    memory_store = _patch_memory_paths(monkeypatch, tmp_path)

    assert memory_store.list_topics()["topics"] == []
    assert (tmp_path / "user_topics.json").exists()

    (tmp_path / "user_topics.json").write_text("{broken", encoding="utf-8")
    assert memory_store.list_topics()["topics"] == []


def test_upsert_topics_deduplicates(monkeypatch, tmp_path):
    memory_store = _patch_memory_paths(monkeypatch, tmp_path)

    result = memory_store.upsert_topics(["LLM Agent", "llm agent", "Agentic RAG"])

    assert result["topics"] == ["LLM Agent", "Agentic RAG"]
    assert memory_store.list_topics()["topics"] == ["LLM Agent", "Agentic RAG"]


def test_save_paper_and_history(monkeypatch, tmp_path):
    memory_store = _patch_memory_paths(monkeypatch, tmp_path)

    memory_store.save_paper({"title": "Memory Agents", "url": "https://example.com/a"})
    memory_store.save_paper({"title": "Memory Agents", "url": "https://example.com/a"})
    memory_store.add_history({"query": "long-term memory in LLM agents"})

    assert len(memory_store.list_saved_papers()["papers"]) == 1
    assert memory_store.list_history()["history"][0]["query"] == "long-term memory in LLM agents"


def test_chat_sessions_append_and_compress(monkeypatch, tmp_path):
    memory_store = _patch_memory_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(memory_store, "RECENT_MESSAGE_LIMIT", 4)

    session_id = None
    for index in range(4):
        session = memory_store.append_chat_turn(
            session_id=session_id,
            query=f"第 {index} 轮：总分框架和长期记忆要求",
            response={
                "query": f"第 {index} 轮",
                "answer": "需要保留任务、框架、总分型讲解和关键结论。",
                "papers": [],
                "citations": [],
                "steps": [],
            },
        )
        session_id = session["id"]

    loaded = memory_store.get_chat_session(session_id)
    assert loaded is not None
    assert loaded["title"].startswith("第 0 轮")
    assert len(loaded["messages"]) == 4
    assert loaded["memory"]["compressed_message_count"] == 4
    assert "总分框架" in loaded["memory"]["summary"]
    assert any("长期记忆" in note for note in loaded["memory"]["important_notes"])

    sessions = memory_store.list_chat_sessions()["sessions"]
    assert sessions[0]["id"] == session_id
    assert sessions[0]["message_count"] == 8
