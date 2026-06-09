"""Smoke test for graph-contrastive recommender PaperRadar quality.

Run from ``backend``:
    python scripts/smoke_graph_rec_quality.py
"""

from __future__ import annotations

import re
import sys

from app.agents.nodes.generator import _build_radar_report
from app.services.paper_roles import assign_paper_roles


QUERY = "图对比学习推荐系统算法论文雷达"


def _paper(index: int, title: str, abstract: str, published: str = "2025-01-01") -> dict:
    return {
        "title": title,
        "authors": [f"Author {index}"],
        "abstract": abstract,
        "url": f"https://example.com/{index}",
        "source": "openalex",
        "published": published,
        "relevance_score": 0.9 - index * 0.01,
        "relevance_tier": "core",
    }


def _documents() -> list[dict]:
    return assign_paper_roles(
        [
            _paper(
                1,
                "Contrastive Self-supervised Learning in Recommender Systems: A Survey",
                "A survey of contrastive self-supervised learning in recommender systems.",
                "2023-03-17",
            ),
            _paper(
                2,
                "Graph Neural Networks in Recommender Systems: A Survey",
                "A survey of graph neural network recommender systems and collaborative filtering.",
                "2020-11-01",
            ),
            _paper(
                3,
                "LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation",
                "LightGCL is a simple and effective graph contrastive learning method for recommendation.",
                "2023-02-16",
            ),
            _paper(
                4,
                "XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation",
                "XSimGCL studies extremely simple graph contrastive learning for recommendation.",
                "2022-09-06",
            ),
            _paper(
                5,
                "Graph Contrastive Learning for Optimizing Sparse Data in Recommender Systems with LightGCL",
                "Applies LightGCL to sparse data in recommender systems.",
                "2025-05-28",
            ),
            _paper(
                6,
                "Let Invariant Rationale Discovery Inspire Graph Contrastive Learning",
                "Invariant rationale discovery can improve graph contrastive learning and explanations.",
                "2024-08-01",
            ),
            _paper(
                7,
                "Knowledge Graph-based Recommender Systems: A Survey",
                "A survey of knowledge graph based recommender systems.",
                "2024-01-01",
            ),
            _paper(
                8,
                "A Survey on Multi-Objective Recommender Systems",
                "A survey of multi-objective recommender systems.",
                "2024-06-01",
            ),
            _paper(
                9,
                "Graph Learning Based Recommender Systems: A Review",
                "A review of graph learning based recommender systems.",
                "2024-03-01",
            ),
        ],
        QUERY,
    )


def _section(answer: str, index: int) -> str:
    next_index = index + 1
    pattern = rf"(?ms)^##\s*{index}[.、]?\s+.*?(?=^##\s*{next_index}[.、]?\s+|\Z)"
    match = re.search(pattern, answer)
    return match.group(0) if match else ""


def _top_recommendation_titles(section: str, limit: int = 3) -> str:
    cards = re.findall(r"(?ms)^###\s+(?:必读|重点|背景)\s+\d+：.*?(?=^###\s+|\Z)", section)
    return "\n".join(cards[:limit])


def validate(answer: str) -> list[str]:
    failures: list[str] = []
    section2 = _section(answer, 2)
    section3 = _section(answer, 3)
    section8 = _section(answer, 8)

    required_section2 = [
        "Survey / Foundations",
        "Graph Neural Recommendation / Collaborative Filtering",
        "Graph Contrastive Learning / SSL Recommendation",
        "Simplified / Efficient GCL",
    ]
    for text in required_section2:
        if text not in section2:
            failures.append(f"Section 2 missing route: {text}")

    forbidden_rag_routes = [
        "Planning / Reasoning / Iterative Retrieval",
        "Multimodal RAG",
        "Multi-Agent / Hierarchical RAG",
        "LangGraph",
        "citation grounding",
        "query rewrite",
    ]
    for text in forbidden_rag_routes:
        if text.lower() in section2.lower() or text.lower() in section3.lower():
            failures.append(f"Graph recommender report contains RAG residue: {text}")

    for text in [
        "Contrastive Self-supervised Learning in Recommender Systems",
        "LightGCL",
        "XSimGCL",
    ]:
        if text not in section3:
            failures.append(f"Section 3 missing required paper: {text}")

    top_three = _top_recommendation_titles(section3, 3)
    for text in ["Knowledge Graph", "Multi-Objective", "Invariant Rationale"]:
        if text.lower() in top_three.lower():
            failures.append(f"Non-core paper appears in top three recommendations: {text}")

    top_four = _top_recommendation_titles(section3, 4)
    for text in [
        "Contrastive Self-supervised Learning in Recommender Systems",
        "Graph Neural Networks",
        "LightGCL",
        "XSimGCL",
    ]:
        if text.lower() not in top_four.lower():
            failures.append(f"Top four recommendations should include: {text}")

    for text in ["当前检索覆盖度", "建议补充检索关键词"]:
        if text not in answer:
            failures.append(f"Report missing coverage block text: {text}")

    for text in ["核心论文", "综述与背景", "相关但非核心"]:
        if text not in section8:
            failures.append(f"Section 8 missing reference group: {text}")

    bad_templates = [
        "与“",
        "所覆盖的",
        "Agentic RAG 研究的是",
        "论文数量一致性检查器",
        "年份过滤核查器",
        "摘要证据表",
    ]
    for text in bad_templates:
        if text in answer:
            failures.append(f"Report contains banned template residue: {text}")

    return failures


def main() -> int:
    answer = _build_radar_report(QUERY, _documents())
    failures = validate(answer)
    if failures:
        print("Graph recommender quality smoke failed:")
        for failure in failures:
            print(f"- {failure}")
        print("\nSection 2:\n", _section(answer, 2)[:2000])
        print("\nSection 3:\n", _section(answer, 3)[:2000])
        print("\nSection 8:\n", _section(answer, 8)[:2000])
        return 1

    print("Graph recommender quality smoke passed.")
    print(_section(answer, 2)[:900])
    print(_section(answer, 3)[:900])
    print(_section(answer, 8)[:900])
    return 0


if __name__ == "__main__":
    sys.exit(main())
