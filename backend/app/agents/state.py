"""Agent state definition for the ScholarAgent LangGraph pipeline."""

from typing import Annotated, Any, TypedDict

from langgraph.graph import add_messages


def _normalize_serialized_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return message
    if "role" in message and "content" in message:
        return message

    message_id = message.get("id")
    kwargs = message.get("kwargs")
    if not isinstance(message_id, list) or not isinstance(kwargs, dict):
        return message

    message_type = message_id[-1] if message_id else ""
    role_by_type = {
        "HumanMessage": "user",
        "AIMessage": "assistant",
        "SystemMessage": "system",
        "ToolMessage": "tool",
    }
    role = role_by_type.get(str(message_type))
    if role is None:
        return message

    return {"role": role, "content": kwargs.get("content", "")}


def _normalize_message_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_normalize_serialized_message(message) for message in value]
    return [_normalize_serialized_message(value)]


def merge_studio_messages(left: Any, right: Any) -> list[Any]:
    """Merge messages while accepting LangGraph Studio serialized messages."""
    return add_messages(
        _normalize_message_list(left),
        _normalize_message_list(right),
    )


class AgentState(TypedDict, total=False):
    """Shared state that flows through every node in the research graph.

    Fields marked ``total=False`` are optional — nodes progressively
    populate them as the graph executes.
    """

    # --- Input ---
    query: str
    original_query: str
    messages: Annotated[list[Any], merge_studio_messages]
    sources: list[str]          # e.g. ["arxiv", "pubmed"]
    max_results: int

    # --- Retrieval ---
    documents: list[dict]       # raw PaperResult dicts from PaperFetcher
    graded_documents: list[dict]  # only docs that pass relevance grading
    background_documents: list[dict]  # partly relevant docs for context/background

    # --- Generation ---
    answer: str
    citations: list[dict]       # Citation dicts with index, title, url
    hallucination_score: float  # 0.0 = fully grounded, 1.0 = hallucinated

    # --- Memory ---
    memory_context: dict        # user topics, saved papers, and reading history

    # --- Control ---
    classification: str         # paper_search, paper_radar, reading_plan, project_idea, general
    rewrite_count: int          # number of query rewrites performed (default 0)
    steps: list[dict]           # AgentStep dicts — execution trace
