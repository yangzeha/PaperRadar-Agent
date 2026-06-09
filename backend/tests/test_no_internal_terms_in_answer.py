"""Tests that final answers hide internal provider/runtime terms."""

from app.agents.nodes.synthesizer import remove_internal_terms


def test_synthesizer_removes_internal_terms():
    answer = (
        "当前 mock/fallback 模式不会编造。"
        "本次结果只基于 mock，需要打开原文确认。"
    )

    cleaned = remove_internal_terms(answer)
    lowered = cleaned.lower()

    assert "mock" not in lowered
    assert "fallback" not in lowered
    assert "需要打开原文确认" not in cleaned


def test_radar_quality_fallback_removes_internal_terms():
    from app.agents.nodes.generator import _ensure_radar_quality

    answer = """
# PaperRadar：Agentic RAG

## 1. 方向概览
当前 mock/fallback 模式。
"""
    result = _ensure_radar_quality(answer, "Agentic RAG", [])
    lowered = result.lower()

    assert "mock" not in lowered
    assert "fallback" not in lowered
    assert "## 5. 研究空白" in result
