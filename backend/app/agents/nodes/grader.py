"""Grader node — LLM judges whether each retrieved document is relevant to the query."""

import logging
import re
import time

from langchain_core.messages import HumanMessage

from app.agents.state import AgentState
from app.services.paper_classifier import classify_document
from app.services.llm_provider import extract_text, invoke_with_retry

logger = logging.getLogger(__name__)

_GRADER_PROMPT = (
    "You are a relevance grader for a research assistant. "
    "Given a user query and paper abstracts, assign each paper a relevance score.\n\n"
    "For EACH paper, respond on one line with this exact format:\n"
    "1: score=0.82 tier=core reason=directly studies the query topic\n"
    "2: score=0.43 tier=background reason=related RAG background but not agentic\n"
    "3: score=0.12 tier=exclude reason=unrelated topic\n\n"
    "Scoring rules:\n"
    "- score >= 0.55: tier=core, may appear in the report body.\n"
    "- 0.35 <= score < 0.55: tier=background, may support context only.\n"
    "- score < 0.35: tier=exclude, do not use in the report body.\n"
    "Do not mark a paper core just because it mentions LLMs; it must help answer the user topic.\n\n"
    "Query: {query}\n\n"
    "{papers_block}\n\n"
    "Relevance scores:"
)


def _local_score(document: dict) -> float:
    score = document.get("relevance_score")
    try:
        numeric = float(score)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(numeric / 10.0, 1.0))


def _parse_verdict(verdict_text: str, index: int) -> tuple[float | None, str | None, str]:
    line_match = re.search(rf"(?m)^\s*{index}\s*:\s*(.+)$", verdict_text)
    if not line_match:
        return None, None, ""

    line = line_match.group(1).strip()
    score_match = re.search(r"score\s*=\s*(0(?:\.\d+)?|1(?:\.0+)?)", line)
    tier_match = re.search(r"tier\s*=\s*(core|background|exclude)", line)

    if score_match:
        score = min(max(float(score_match.group(1)), 0.0), 1.0)
    elif re.search(r"\byes\b", line):
        score = 0.65
    elif re.search(r"\bno\b", line):
        score = 0.0
    else:
        score = None

    tier = tier_match.group(1) if tier_match else None
    reason_match = re.search(r"reason\s*=\s*(.+)$", line)
    reason = reason_match.group(1).strip() if reason_match else line
    return score, tier, reason


def _tier_for_score(score: float) -> str:
    if score >= 0.55:
        return "core"
    if score >= 0.35:
        return "background"
    return "exclude"


def grade_documents(state: AgentState) -> AgentState:
    """Grade each retrieved document for relevance in a single batched LLM call."""
    start = time.perf_counter()

    query = state["query"]
    documents = state.get("documents", [])

    if not documents:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        steps = list(state.get("steps", []))
        steps.append({
            "node": "grader",
            "status": "completed",
            "detail": "0/0 documents relevant",
            "duration_ms": elapsed_ms,
        })
        return {**state, "graded_documents": [], "background_documents": [], "steps": steps}

    # Build a single prompt with all papers. Keep abstracts shorter when the
    # user asks for many papers so the grading prompt stays within model limits.
    abstract_limit = 1000
    if len(documents) > 50:
        abstract_limit = 300
    elif len(documents) > 20:
        abstract_limit = 500

    papers_block = ""
    for i, doc in enumerate(documents, 1):
        title = doc.get("title", "Untitled")
        published = doc.get("published") or "unknown"
        source = doc.get("source") or "unknown"
        abstract = doc.get("abstract", "")[:abstract_limit]
        papers_block += (
            f"Paper {i}: {title}\n"
            f"Published: {published}\n"
            f"Source: {source}\n"
            f"Abstract: {abstract}\n\n"
        )

    prompt = _GRADER_PROMPT.format(query=query, papers_block=papers_block)
    response = invoke_with_retry([HumanMessage(content=prompt)])
    verdict_text = extract_text(response).strip().lower()

    core: list[dict] = []
    background: list[dict] = []
    for i, doc in enumerate(documents, 1):
        parsed_score, parsed_tier, reason = _parse_verdict(verdict_text, i)
        local = _local_score(doc)
        score = max(parsed_score if parsed_score is not None else 0.0, local)
        tier = parsed_tier or _tier_for_score(score)
        if parsed_score is None and local > 0:
            tier = _tier_for_score(local)

        enriched = classify_document(
            {
                **doc,
                "relevance_score": round(score, 3),
                "relevance_tier": tier,
                "relevance_reason": reason or doc.get("relevance_reason", ""),
            },
            query=query,
        )

        if tier == "core" and score >= 0.55:
            core.append(enriched)
            logger.debug("CORE: %s", doc.get("title", "")[:80])
        elif tier in {"core", "background"} and score >= 0.35:
            enriched["relevance_tier"] = "background"
            background.append(enriched)
            logger.debug("BACKGROUND: %s", doc.get("title", "")[:80])
        else:
            logger.debug("EXCLUDE: %s", doc.get("title", "")[:80])

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "Graded %d docs → %d relevant in %dms",
        len(documents),
        len(core),
        elapsed_ms,
    )

    steps = list(state.get("steps", []))
    steps.append({
        "node": "grader",
        "status": "completed",
        "detail": (
            f"{len(core)}/{len(documents)} core, "
            f"{len(background)} background documents"
        ),
        "duration_ms": elapsed_ms,
    })

    return {
        **state,
        "graded_documents": core,
        "background_documents": background,
        "steps": steps,
    }
