"""Tests for graph routing and final paper-count controls."""

from __future__ import annotations


def _paper(index: int) -> dict:
    return {
        "title": f"Paper {index}",
        "authors": [f"Author {index}"],
        "abstract": f"Abstract {index}",
        "url": f"https://example.com/{index}",
        "source": "arxiv",
        "published": "2026-01-01",
        "relevance_score": None,
    }


def _scored_paper(index: int, score: float) -> dict:
    return {**_paper(index), "relevance_score": score}


def test_hallucination_score_at_threshold_retries_generation(monkeypatch):
    from app.agents.graph import _route_after_hallucination
    from app.config import settings

    monkeypatch.setattr(settings, "hallucination_threshold", 0.3)
    monkeypatch.setattr(settings, "max_hallucination_retries", 5)

    assert (
        _route_after_hallucination(
            {
                "hallucination_score": 0.29,
                "steps": [{"node": "generator"}],
            }
        )
        == "synthesizer"
    )
    assert (
        _route_after_hallucination(
            {
                "hallucination_score": 0.3,
                "steps": [{"node": "generator"}],
            }
        )
        == "generator"
    )
    assert (
        _route_after_hallucination(
            {
                "hallucination_score": 0.7,
                "steps": [{"node": "generator"}, {"node": "generator"}],
            }
        )
        == "generator"
    )
    assert (
        _route_after_hallucination(
            {
                "hallucination_score": 0.7,
                "steps": [{"node": "generator"}] * 6,
            }
        )
        == "synthesizer"
    )


def test_output_documents_fill_to_requested_count():
    from app.agents.document_selection import select_output_documents

    documents = [_paper(i) for i in range(1, 11)]
    state = {
        "max_results": 10,
        "graded_documents": documents[:4],
        "documents": documents,
    }

    selected = select_output_documents(state)

    assert len(selected) == 10
    assert [paper["title"] for paper in selected[:4]] == [
        "Paper 1",
        "Paper 2",
        "Paper 3",
        "Paper 4",
    ]
    assert selected[-1]["title"] == "Paper 10"


def test_output_documents_do_not_fill_with_zero_relevance():
    from app.agents.document_selection import select_output_documents

    state = {
        "max_results": 5,
        "graded_documents": [_scored_paper(1, 3.0), _scored_paper(2, 2.0)],
        "documents": [
            _scored_paper(1, 3.0),
            _scored_paper(2, 2.0),
            _scored_paper(3, 0.0),
            _scored_paper(4, 0.0),
        ],
    }

    selected = select_output_documents(state)

    assert [paper["title"] for paper in selected] == ["Paper 1", "Paper 2"]


def test_synthesizer_keeps_unreferenced_reference_entries():
    from app.agents.nodes.synthesizer import synthesize_response

    citations = [
        {"index": i, "title": f"Paper {i}", "url": f"https://example.com/{i}"}
        for i in range(1, 11)
    ]
    result = synthesize_response(
        {
            "answer": "Only the first paper is cited [1]. This dangling ref is removed [99].",
            "citations": citations,
            "steps": [],
        }
    )

    assert len(result["citations"]) == 10
    assert "[99]" not in result["answer"]
