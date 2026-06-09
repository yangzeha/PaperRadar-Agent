"""Router node: classifies the user query into a research task type."""

from __future__ import annotations

import logging
import re
import time

from langchain_core.messages import HumanMessage

from app.agents.state import AgentState
from app.services.llm_provider import extract_text, invoke_with_retry

logger = logging.getLogger(__name__)

_VALID_TASKS = {
    "paper_search",
    "paper_qa",
    "paper_radar",
    "reading_plan",
    "project_idea",
    "general",
}


def _contains_any(text: str, words: list[str]) -> bool:
    return any(word in text for word in words)


def _is_general_greeting(text: str) -> bool:
    english_greeting = re.search(r"\b(?:hello|hi|hey)\b", text) is not None
    chinese_greeting = _contains_any(
        text,
        [
            "你好",
            "您好",
            "在吗",
            "谢谢",
            "早上好",
            "晚上好",
            "晚安",
        ],
    )
    return english_greeting or chinese_greeting


_ROUTER_PROMPT = (
    "You are a query classifier for a research assistant. "
    "Given the user query below, respond with EXACTLY one label:\n"
    "- 'paper_radar' for trend analysis, topic tracking, research-gap reports, "
    "paper radar reports, or direction overview requests.\n"
    "- 'reading_plan' for reading routes, study plans, or two-week learning plans.\n"
    "- 'project_idea' for mini-project, experiment, or resume-project ideas.\n"
    "- 'paper_qa' for questions that require answering from academic papers.\n"
    "- 'general' for casual conversation, greetings, or non-research messages.\n\n"
    "Legacy label 'paper_search' is also acceptable for plain literature search.\n\n"
    "Query: {query}\n\n"
    "Classification:"
)


def _heuristic_classification(query: str) -> str | None:
    text = query.lower().strip()

    if _contains_any(
        text,
        [
            "论文雷达",
            "趋势",
            "研究空白",
            "选题",
            "方向",
            "radar",
            "trend",
            "gap",
            "topic tracking",
            "research direction",
            "long-term memory in llm agents",
        ],
    ):
        return "paper_radar"

    if _contains_any(
        text,
        [
            "两周",
            "阅读路线",
            "学习路线",
            "reading plan",
            "reading route",
            "study plan",
            "two-week",
            "2 week",
        ],
    ):
        return "reading_plan"

    if _contains_any(
        text,
        [
            "小项目",
            "项目建议",
            "简历项目",
            "project idea",
            "mini-project",
        ],
    ):
        return "project_idea"

    if _contains_any(
        text,
        [
            "paper qa",
            "paper_qa",
            "what does the paper say",
            "how does the paper",
            "explain this paper",
            "summarize this paper",
            "根据论文",
            "根据这篇论文",
            "这篇论文",
            "文中",
            "文里",
            "论文问答",
            "论文里",
        ],
    ):
        return "paper_qa"

    if _is_general_greeting(text) and not _contains_any(
        text,
        [
            "paper",
            "literature",
            "survey",
            "arxiv",
            "pubmed",
            "论文",
            "文献",
            "综述",
        ],
    ):
        return "general"

    if _contains_any(
        text,
        [
            "paper",
            "literature",
            "survey",
            "arxiv",
            "pubmed",
            "论文",
            "文献",
            "综述",
        ],
    ):
        return "paper_search"

    return None


def _normalize_classification(raw: str, query: str) -> str:
    value = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if value in _VALID_TASKS:
        return value
    if "paper_radar" in value or "radar" in value or "trend" in value:
        return "paper_radar"
    if "reading" in value or "study" in value:
        return "reading_plan"
    if "project" in value or "experiment" in value:
        return "project_idea"
    if "qa" in value or "question" in value or "answer" in value:
        return "paper_qa"
    if "general" in value or "chat" in value or "greeting" in value:
        return "general"
    if "paper" in value or "search" in value or "literature" in value:
        return "paper_search"
    return _heuristic_classification(query) or "general"


def route_query(state: AgentState) -> AgentState:
    """Classify the user query intent and record the routing decision."""
    start = time.perf_counter()
    query = state["query"]

    heuristic = _heuristic_classification(query)
    if heuristic is not None:
        classification = heuristic
    else:
        prompt = _ROUTER_PROMPT.format(query=query)
        response = invoke_with_retry([HumanMessage(content=prompt)])
        classification = _normalize_classification(extract_text(response), query)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info("Query routed as '%s' in %dms", classification, elapsed_ms)

    steps = list(state.get("steps", []))
    steps.append(
        {
            "node": "router",
            "status": "completed",
            "detail": f"Classified as '{classification}'",
            "duration_ms": elapsed_ms,
        }
    )

    return {**state, "classification": classification, "steps": steps}
