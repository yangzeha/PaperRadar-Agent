"""Hallucination checker node — verifies the answer is grounded in source documents."""

import logging
import re
import time

from langchain_core.messages import HumanMessage

from app.agents.document_selection import select_output_documents
from app.agents.state import AgentState
from app.services.llm_provider import extract_text, invoke_with_retry

logger = logging.getLogger(__name__)

_HALLUCINATION_PROMPT = (
    "You are a hallucination detector. Compare the generated answer against "
    "the source documents and rate how grounded the answer is.\n\n"
    "Score from 0.0 to 1.0:\n"
    "- 0.0 = every claim is directly supported by the sources\n"
    "- 0.5 = some claims lack source support\n"
    "- 1.0 = the answer is entirely fabricated\n\n"
    "Source documents:\n{sources_text}\n\n"
    "Generated answer:\n{answer}\n\n"
    "Respond with ONLY a decimal number between 0.0 and 1.0."
)


def _format_sources(documents: list[dict]) -> str:
    """Concatenate graded document abstracts for the checker prompt."""
    if not documents:
        return "No source documents provided."
    parts = []
    for i, doc in enumerate(documents, start=1):
        title = doc.get("title", "Untitled")
        abstract = doc.get("abstract", "")[:1000]
        parts.append(f"[{i}] {title}: {abstract}")
    return "\n".join(parts)


def check_hallucination(state: AgentState) -> AgentState:
    """Check whether the generated answer is grounded in the source papers."""
    start = time.perf_counter()

    answer = state.get("answer", "")
    source_documents = select_output_documents(state)
    classification = state.get("classification", "paper_search")

    # Skip hallucination check for general (non-research) queries
    if classification == "general" or not source_documents:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        steps = list(state.get("steps", []))
        steps.append({
            "node": "hallucination_checker",
            "status": "skipped",
            "detail": "No source documents to check against",
            "duration_ms": elapsed_ms,
        })
        return {**state, "hallucination_score": 0.0, "steps": steps}

    sources_text = _format_sources(source_documents)
    prompt = _HALLUCINATION_PROMPT.format(sources_text=sources_text, answer=answer)

    response = invoke_with_retry([HumanMessage(content=prompt)])
    raw_score = extract_text(response).strip()

    # Parse the score — extract first decimal number from response
    match = re.search(r"(\d+\.?\d*)", raw_score)
    if match:
        score = min(max(float(match.group(1)), 0.0), 1.0)
    else:
        logger.warning("Could not parse hallucination score from: '%s'", raw_score)
        score = 1.0  # conservative default: uncertain should trigger regeneration

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info("Hallucination score: %.2f in %dms", score, elapsed_ms)

    steps = list(state.get("steps", []))
    steps.append({
        "node": "hallucination_checker",
        "status": "completed",
        "detail": f"Hallucination score: {score:.2f}",
        "duration_ms": elapsed_ms,
    })

    return {**state, "hallucination_score": score, "steps": steps}
