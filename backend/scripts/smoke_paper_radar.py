"""Smoke test for PaperRadar mode under the deterministic mock provider."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.agents.nodes.generator import generate_answer
from app.agents.nodes.synthesizer import synthesize_response
from app.config import settings


REQUIRED_SECTIONS = [
    "方向概览",
    "方法路线分类",
    "代表论文推荐",
    "近年趋势",
    "研究空白",
    "两周阅读路线",
    "可做小项目建议",
    "参考来源",
]


def main() -> None:
    settings.llm_provider = "mock"
    query = "long-term memory in LLM agents"
    sample_docs = [
        {
            "title": "Memory-Augmented Language Agents",
            "authors": ["Example Author"],
            "abstract": (
                "This paper studies long-term memory mechanisms for language agents, "
                "including memory writing, retrieval, reflection, and task reuse."
            ),
            "url": "https://arxiv.org/abs/2401.00001",
            "source": "arxiv",
            "published": "2024-01-01",
            "relevance_score": 0.91,
        },
        {
            "title": "Agentic RAG for Grounded Research Assistants",
            "authors": ["Example Author"],
            "abstract": (
                "This work combines retrieval-augmented generation, grading, rewriting, "
                "and hallucination checks in a graph-based agent workflow."
            ),
            "url": "https://arxiv.org/abs/2402.00002",
            "source": "arxiv",
            "published": "2024-02-01",
            "relevance_score": 0.88,
        },
    ]
    state = {
        "query": query,
        "sources": ["arxiv"],
        "max_results": 2,
        "documents": sample_docs,
        "graded_documents": sample_docs,
        "classification": "paper_radar",
        "rewrite_count": 0,
        "answer": "",
        "hallucination_score": 0.0,
        "steps": [],
        "citations": [],
        "memory_context": {},
    }

    generated = generate_answer(state)
    final_state = synthesize_response(generated)
    answer = final_state["answer"]

    missing = [section for section in REQUIRED_SECTIONS if section not in answer]
    if missing:
        raise AssertionError(f"Missing PaperRadar sections: {missing}")

    print("PaperRadar mock smoke passed.")
    print(answer)


if __name__ == "__main__":
    main()
