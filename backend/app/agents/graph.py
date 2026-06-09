"""LangGraph state graph for the ScholarAgent research pipeline.

Graph topology:

  START -> router
  router -> retriever       (if paper_search)
  router -> generator       (if general)
  retriever -> grader
  grader -> generator       (if docs are relevant OR retries exhausted)
  grader -> rewriter        (if docs irrelevant AND retries remaining)
  rewriter -> retriever     (loop back for another search)
  generator -> hallucination_checker
  hallucination_checker -> synthesizer  (if score < threshold)
  hallucination_checker -> generator    (if score >= threshold, retry up to max_hallucination_retries)
  synthesizer -> END
"""

import logging

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph

from app.agents.document_selection import select_output_documents
from app.agents.nodes.generator import generate_answer
from app.agents.nodes.grader import grade_documents
from app.agents.nodes.hallucination_checker import check_hallucination
from app.agents.nodes.retriever import retrieve_papers
from app.agents.nodes.rewriter import rewrite_query
from app.agents.nodes.router import route_query
from app.agents.nodes.synthesizer import synthesize_response
from app.agents.state import AgentState
from app.config import settings
from app.models.schemas import AgentStep, Citation, PaperResult, SearchResponse
from app.services.memory_store import get_chat_session

logger = logging.getLogger(__name__)

REPORT_TEMPLATE_VERSION = "paper-radar-card-v3"


def _content_to_text(content: object) -> str:
    """Extract readable text from LangChain/OpenAI-style message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block.get("text") or ""))
                elif "content" in block:
                    parts.append(_content_to_text(block.get("content")))
            elif block is not None:
                parts.append(str(block))
        return "\n".join(part for part in parts if part).strip()
    if content is None:
        return ""
    return str(content)


def _message_to_text(message: object) -> str:
    if isinstance(message, BaseMessage):
        return _content_to_text(message.content)
    if isinstance(message, dict):
        if "content" in message:
            return _content_to_text(message.get("content"))
        kwargs = message.get("kwargs")
        if isinstance(kwargs, dict):
            return _content_to_text(kwargs.get("content"))
    return _content_to_text(message)


def _adapt_studio_input(state: AgentState) -> AgentState:
    """Normalize LangGraph Studio chat input into this graph's query state."""
    query = str(state.get("query") or "").strip()
    if not query:
        messages = state.get("messages", [])
        for message in reversed(messages):
            query = _message_to_text(message).strip()
            if query:
                break

    return {
        **state,
        "query": query,
        "original_query": state.get("original_query") or query,
        "sources": state.get("sources") or ["arxiv", "pubmed", "openalex"],
        "max_results": state.get("max_results") or settings.max_papers,
        "documents": state.get("documents", []),
        "graded_documents": state.get("graded_documents", []),
        "background_documents": state.get("background_documents", []),
        "rewrite_count": state.get("rewrite_count", 0),
        "answer": state.get("answer", ""),
        "hallucination_score": state.get("hallucination_score", 0.0),
        "steps": state.get("steps", []),
        "citations": state.get("citations", []),
        "memory_context": state.get("memory_context", {}),
    }


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def _route_after_router(state: AgentState) -> str:
    """Decide whether to search papers or go straight to generation."""
    classification = state.get("classification", "paper_search")
    if classification == "general":
        return "generator"
    return "retriever"


def _route_after_grader(state: AgentState) -> str:
    """If graded docs are empty and we have retries left, rewrite; otherwise generate."""
    graded = state.get("graded_documents", [])
    rewrite_count = state.get("rewrite_count", 0)
    max_retries = settings.max_rewrite_retries

    if graded:
        return "generator"

    if rewrite_count < max_retries:
        logger.info(
            "No relevant docs — rewriting query (attempt %d/%d)",
            rewrite_count + 1,
            max_retries,
        )
        return "rewriter"

    # Retries exhausted — generate with whatever we have
    logger.warning("Rewrite retries exhausted (%d/%d) — generating anyway", rewrite_count, max_retries)
    return "generator"


def _route_after_hallucination(state: AgentState) -> str:
    """Route by hallucination score.

    A score below the threshold is considered grounded enough to synthesize.
    A score at or above the threshold goes back to the generator up to the
    configured retry cap, preventing an infinite generator/checker loop.
    """
    score = state.get("hallucination_score", 0.0)
    threshold = settings.hallucination_threshold
    max_retries = settings.max_hallucination_retries

    # Count how many times generator has already run
    generator_runs = sum(
        1 for step in state.get("steps", []) if step.get("node") == "generator"
    )

    if score < threshold:
        return "synthesizer"

    hallucination_retries = max(generator_runs - 1, 0)
    if hallucination_retries < max_retries:
        logger.info(
            "Hallucination score %.2f >= threshold %.2f — retrying generation (%d/%d)",
            score,
            threshold,
            hallucination_retries + 1,
            max_retries,
        )
        return "generator"

    logger.warning(
        "Hallucination score %.2f is still >= threshold %.2f after %d retries — synthesizing latest answer",
        score,
        threshold,
        max_retries,
    )
    return "synthesizer"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the ScholarAgent state graph.

    Returns the compiled graph ready to be invoked.
    """
    graph = StateGraph(AgentState)

    # -- Add nodes --
    graph.add_node("input_adapter", _adapt_studio_input)
    graph.add_node("router", route_query)
    graph.add_node("retriever", retrieve_papers)
    graph.add_node("grader", grade_documents)
    graph.add_node("rewriter", rewrite_query)
    graph.add_node("generator", generate_answer)
    graph.add_node("hallucination_checker", check_hallucination)
    graph.add_node("synthesizer", synthesize_response)

    # -- Add edges --
    graph.add_edge(START, "input_adapter")
    graph.add_edge("input_adapter", "router")

    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {"retriever": "retriever", "generator": "generator"},
    )

    graph.add_edge("retriever", "grader")

    graph.add_conditional_edges(
        "grader",
        _route_after_grader,
        {"generator": "generator", "rewriter": "rewriter"},
    )

    graph.add_edge("rewriter", "retriever")

    graph.add_edge("generator", "hallucination_checker")

    graph.add_conditional_edges(
        "hallucination_checker",
        _route_after_hallucination,
        {"synthesizer": "synthesizer", "generator": "generator"},
    )

    graph.add_edge("synthesizer", END)

    return graph.compile()


# Exported for LangGraph Studio via langgraph.json.
graph = build_graph()


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

async def run_search(
    query: str,
    sources: list[str] | None = None,
    max_results: int | None = None,
    session_id: str | None = None,
) -> SearchResponse:
    """Execute the full ScholarAgent pipeline and return a SearchResponse.

    This is the main entry point called by the API layer.
    """
    if sources is None:
        sources = ["arxiv", "pubmed", "openalex"]
    if max_results is None:
        max_results = settings.max_papers

    memory_context = {}
    if session_id:
        current_session = get_chat_session(session_id)
        if current_session:
            memory_context["current_session"] = current_session

    initial_state: AgentState = {
        "query": query,
        "original_query": query,
        "sources": sources,
        "max_results": max_results,
        "documents": [],
        "graded_documents": [],
        "background_documents": [],
        "rewrite_count": 0,
        "answer": "",
        "hallucination_score": 0.0,
        "steps": [],
        "citations": [],
        "memory_context": memory_context,
    }

    compiled_graph = build_graph()

    # Use ainvoke for async context (called from async FastAPI endpoint)
    final_state = await compiled_graph.ainvoke(initial_state)

    # --- Build SearchResponse from final state ---
    papers = [PaperResult(**doc) for doc in select_output_documents(final_state)]

    citations = [
        Citation(**c)
        for c in final_state.get("citations", [])
    ]

    steps = [
        AgentStep(**s)
        for s in final_state.get("steps", [])
    ]

    return SearchResponse(
        query=final_state.get("original_query", query),
        answer=final_state.get("answer", ""),
        citations=citations,
        papers=papers,
        steps=steps,
        rewrite_count=final_state.get("rewrite_count", 0),
        classification=final_state.get("classification"),
        session_id=session_id,
        report_template_version=REPORT_TEMPLATE_VERSION,
    )
