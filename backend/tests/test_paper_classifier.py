"""Tests for rule-based PaperRadar paper classification."""

from app.services.paper_classifier import classify_paper


def test_classifies_survey_paper():
    result = classify_paper(
        "Retrieval-Augmented Generation for Large Language Models: A Survey",
        "This survey reviews RAG methods, evaluation, and applications.",
        "2024-01-01",
        "Agentic RAG",
    )

    assert result["route"] == "Survey / Taxonomy / SoK"
    assert result["year"] == 2024


def test_classifies_multi_agent_paper():
    result = classify_paper(
        "Hierarchical Multi-Agent Retrieval for Scientific QA",
        "A planner coordinates multi-agent collaboration between retrievers and critics.",
        "2025",
        "Agentic RAG",
    )

    assert result["route"] == "Multi-Agent / Hierarchical"


def test_classifies_benchmark_paper():
    result = classify_paper(
        "Benchmarking Citation Accuracy for Retrieval-Augmented Generation",
        "We propose evaluation metrics for hallucination and citation grounding.",
        "2026",
        "Agentic RAG evaluation",
    )

    assert result["route"] == "Evaluation / Benchmark"


def test_classifies_multimodal_paper():
    result = classify_paper(
        "Multimodal RAG with Vision-Language Evidence Alignment",
        "The system uses image and visual evidence for MLLM retrieval.",
        "2024",
        "Multimodal Agentic RAG",
    )

    assert result["route"] == "Multimodal RAG"
