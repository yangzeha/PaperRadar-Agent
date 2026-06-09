"""Tests for PaperRadar report quality constraints."""

from __future__ import annotations

import re


def _paper(index: int, title: str, abstract: str, published: str = "2025-01-01") -> dict:
    return {
        "title": title,
        "authors": [f"Author {index}"],
        "abstract": abstract,
        "url": f"https://example.com/{index}",
        "source": "openalex",
        "published": published,
        "relevance_score": 0.8,
        "relevance_tier": "core",
    }


def _documents() -> list[dict]:
    from app.services.paper_classifier import classify_documents

    docs = [
        _paper(1, "Agentic RAG: A Survey and Taxonomy", "survey taxonomy overview RAG agentic"),
        _paper(2, "Planning and Reasoning for Iterative Retrieval", "planning reasoning query rewrite iterative retrieval"),
        _paper(3, "Hierarchical Multi-Agent RAG", "multi-agent hierarchical collaboration planner critic"),
        _paper(4, "Benchmarking Citation Accuracy in RAG", "benchmark evaluation metrics hallucination citation"),
        _paper(5, "Multimodal RAG for Vision-Language QA", "multimodal vision MLLM image evidence alignment"),
        _paper(6, "Medical Agentic RAG for Clinical QA", "medical healthcare domain-specific clinical retrieval"),
    ]
    return classify_documents(docs, query="Agentic RAG")


def test_build_radar_report_has_required_quality():
    from app.agents.nodes.generator import _build_radar_report

    answer = _build_radar_report(
        "Agentic RAG 方向论文雷达：趋势、代表论文、研究空白和两周阅读路线",
        _documents(),
    )

    for heading in [
        "# PaperRadar",
        "## 1. 方向概览",
        "## 2. 方法路线分类",
        "## 3. 代表论文推荐",
        "## 4. 近年趋势",
        "## 5. 研究空白",
        "## 6. 两周阅读路线",
        "## 7. 可做小项目建议",
        "## 8. 参考来源",
    ]:
        assert heading in answer

    assert "|---" not in answer
    assert "| 路线 |" not in answer
    assert "| 推荐级 |" not in answer
    assert "| 时间 |" not in answer
    assert len(re.findall(r"### 路线 [A-Z]", answer)) >= 4
    assert re.search(r"### (必读|重点) \d+", answer)
    for day in ("Day 1-2", "Day 3-4", "Day 5-7", "Day 8-10", "Day 11-12", "Day 13-14"):
        assert f"### {day}" in answer
    assert len(re.findall(r"### Gap \d+", answer)) >= 5
    assert len(re.findall(r"### 项目 \d+", answer)) >= 3
    assert len(set(re.findall(r"\[(\d+)\]", answer))) >= 5
    assert "论文数量一致性检查器" not in answer
    assert "年份过滤核查器" not in answer


def test_generated_gap_and_projects_are_agentic_rag_specific():
    from app.agents.nodes.generator import _build_radar_report

    answer = _build_radar_report("Agentic RAG", _documents())

    assert "Query Rewrite" in answer or "query rewrite" in answer
    assert "LangGraph" in answer
    assert "Citation Grounding" in answer or "citation" in answer.lower()
    assert "Graph Contrastive Learning / SSL Recommendation" not in answer
    assert "LightGCL" not in answer
    assert "XSimGCL" not in answer
    assert "长期记忆污染" in answer


def test_radar_quality_rejects_prompt_leakage():
    from app.agents.nodes.generator import _build_radar_report, _radar_answer_quality_ok

    documents = _documents()
    answer = _build_radar_report("Agentic RAG", documents)
    leaked = answer + "\n\ncurrent_paper_titles: ['private prompt detail']"

    assert _radar_answer_quality_ok(answer, "Agentic RAG", documents)
    assert not _radar_answer_quality_ok(leaked, "Agentic RAG", documents)


def test_graph_contrastive_recommender_report_stays_on_topic():
    from app.agents.nodes.generator import _build_radar_report
    from app.services.paper_classifier import classify_documents

    docs = classify_documents(
        [
            _paper(
                1,
                "Graph Contrastive Learning for Optimizing Sparse Data in Recommender Systems with LightGCL",
                "Graph neural networks are powerful for recommendation but struggle under data sparsity and noise. LightGCL uses graph contrastive learning for recommendation.",
                "2025-05-28",
            ),
            _paper(
                2,
                "LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation",
                "Graph neural network recommender systems integrate contrastive learning for recommendation with simple graph views.",
                "2023-02-16",
            ),
            _paper(
                3,
                "Contrastive Self-supervised Learning in Recommender Systems: A Survey",
                "This survey reviews contrastive self-supervised learning in recommender systems.",
                "2023-03-17",
            ),
            _paper(
                4,
                "XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation",
                "XSimGCL studies simple graph contrastive learning for recommendation.",
                "2022-09-06",
            ),
        ],
        query="图对比学习推荐算法论文雷达",
    )

    answer = _build_radar_report("图对比学习推荐算法论文雷达", docs)
    lowered = answer.lower()

    assert "图对比学习" in answer
    assert "推荐" in answer
    assert "Graph Contrastive Learning / SSL Recommendation" in answer
    assert "LightGCL" in answer
    assert "XSimGCL" in answer
    assert "Agentic RAG 研究的是" not in answer
    assert "普通 RAG" not in answer
    assert "query rewrite" not in lowered
    assert "LangGraph Agentic RAG" not in answer
    assert "长期记忆污染" not in answer
