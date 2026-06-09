"""Tests for PaperRadar report generation."""

from __future__ import annotations

from pathlib import Path
import json
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage


REQUIRED_SECTIONS = [
    "## 1. 方向概览",
    "## 2. 方法路线分类",
    "## 3. 代表论文推荐",
    "## 4. 近年趋势",
    "## 5. 研究空白",
    "## 6. 两周阅读路线",
    "## 7. 可做小项目建议",
    "## 8. 参考来源",
]


def _patch_memory_paths(monkeypatch, tmp_path: Path):
    from app.services import memory_store

    monkeypatch.setattr(memory_store, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(memory_store, "TOPICS_FILE", tmp_path / "user_topics.json")
    monkeypatch.setattr(memory_store, "SAVED_PAPERS_FILE", tmp_path / "saved_papers.json")
    monkeypatch.setattr(memory_store, "HISTORY_FILE", tmp_path / "reading_history.json")
    monkeypatch.setattr(memory_store, "CHAT_SESSIONS_FILE", tmp_path / "chat_sessions.json")
    return memory_store


@patch("app.agents.nodes.generator.invoke_with_retry")
def test_generate_paper_radar_report(mock_invoke, monkeypatch, tmp_path, sample_state, sample_papers):
    from app.agents.nodes.generator import generate_answer

    memory_store = _patch_memory_paths(monkeypatch, tmp_path)
    mock_invoke.return_value = AIMessage(
        content="\n\n".join(f"{section}\n内容 [1]" for section in REQUIRED_SECTIONS)
    )

    state = {
        **sample_state,
        "classification": "paper_radar",
        "graded_documents": sample_papers[:2],
        "query": "LLM Agent 长期记忆机制论文雷达",
    }
    result = generate_answer(state)

    for section in REQUIRED_SECTIONS:
        assert section in result["answer"]
    assert len(result["citations"]) == 3
    assert result["steps"][-1]["node"] == "generator"
    assert "PaperRadar" in result["steps"][-1]["detail"]
    assert memory_store.list_history()["history"][0]["classification"] == "paper_radar"


def test_mock_provider_outputs_required_sections(monkeypatch):
    from app.config import settings
    from app.services.llm_provider import extract_text, invoke_with_retry

    monkeypatch.setattr(settings, "llm_provider", "mock")
    response = invoke_with_retry(
        [
            HumanMessage(
                content=(
                    "你是 PaperRadar-Agent。\n"
                    "Query: long-term memory in LLM agents\n\n"
                    "请生成 PaperRadar 中文报告："
                )
            )
        ]
    )
    answer = extract_text(response)

    for section in REQUIRED_SECTIONS:
        assert section in answer


def test_missing_radar_sections_are_not_fabricated(sample_papers):
    from app.agents.nodes.generator import _ensure_radar_sections

    answer = "## 1. 方向概览\n已有内容 [1]"
    result = _ensure_radar_sections(answer, sample_papers[:1])

    assert "## 5. 研究空白" in result
    assert "当前检索材料" in result
    assert "记忆写入标准仍不统一" not in result


def test_radar_report_body_covers_all_output_documents(sample_papers):
    from app.agents.nodes.generator import _ensure_radar_document_coverage

    answer = "\n\n".join(f"{section}\n内容 [1]" for section in REQUIRED_SECTIONS)
    result = _ensure_radar_document_coverage(answer, sample_papers)

    assert "### 检索覆盖清单" in result
    for index, paper in enumerate(sample_papers, start=1):
        assert f"[{index}]" in result
        assert paper["title"] in result


def test_search_answer_body_covers_all_output_documents(sample_papers):
    from app.agents.nodes.generator import _ensure_search_document_coverage

    result = _ensure_search_document_coverage("已有回答只引用 [1]。", sample_papers)

    assert "## 检索到的论文" in result
    for index, paper in enumerate(sample_papers, start=1):
        assert f"[{index}]" in result
        assert paper["title"] in result


def test_mock_paper_radar_uses_prompt_papers():
    from app.services.llm_provider import _mock_paper_radar

    prompt = """
用户问题：测试方向论文雷达

检索到的论文：
[1] Paper A
URL: https://example.com/a
Published: 2024-01-01
Abstract: Paper A studies the first method.

[2] Paper B
URL: https://example.com/b
Published: 2025-01-01
Abstract: Paper B studies the second method.

[3] Paper C
URL: https://example.com/c
Published: 2026-01-01
Abstract: Paper C studies the third method.
"""

    result = _mock_paper_radar(prompt)

    assert "Paper A" in result
    assert "Paper B" in result
    assert "Paper C" in result
    assert "[3]" in result


def test_mock_paper_search_uses_prompt_papers():
    from app.services.llm_provider import _mock_paper_search

    prompt = """
Query: graph contrastive learning recommender systems

[1] Paper A
URL: https://example.com/a
Published: 2024-01-01
Abstract: Paper A studies the first method.

[2] Paper B
URL: https://example.com/b
Published: 2025-01-01
Abstract: Paper B studies the second method.
"""

    result = _mock_paper_search(prompt)

    assert "Paper A" in result
    assert "Paper B" in result
    assert "[2]" in result


def test_memory_context_separates_short_and_long(monkeypatch, tmp_path, sample_state):
    from app.agents.nodes.generator import _memory_context_for_prompt

    memory_store = _patch_memory_paths(monkeypatch, tmp_path)
    memory_store.upsert_topics(["Agentic RAG"])
    memory_store.add_history({"query": "长期记忆论文"})

    context = json.loads(
        _memory_context_for_prompt(
            {
                **sample_state,
                "query": "长短期记忆怎么体现",
                "classification": "paper_radar",
                "graded_documents": [{"title": "Memory Agents"}],
            }
        )
    )

    assert context["short_term"]["query"] == "长短期记忆怎么体现"
    assert context["short_term"]["graded_documents_count"] == 1
    assert context["long_term"]["topics"] == ["Agentic RAG"]
    assert context["long_term"]["history"][0]["query"] == "长期记忆论文"
