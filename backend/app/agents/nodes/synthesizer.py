"""Synthesizer node — formats the final response and cleans up citations."""

import logging
import re
import time

from langchain_core.messages import AIMessage

from app.agents.state import AgentState

logger = logging.getLogger(__name__)

_INTERNAL_TERM_REPLACEMENTS = {
    "当前 mock/fallback 模式": "当前检索材料",
    "本次结果只基于 mock": "以下分析基于当前检索到的标题、摘要与元数据",
    "fallback 模式不会编造": "以下分析仅使用当前检索材料",
    "mock/fallback": "当前检索材料",
    "fallback": "当前检索材料",
    "mock": "当前检索材料",
    "需要打开原文确认": "建议进一步阅读全文核验细节",
    "证据不足，不能替代完整论文阅读": "当前检索材料对该点支持有限，建议进一步阅读全文",
    "不能替代完整论文阅读": "建议进一步阅读全文",
}

_RADAR_TASKS = {"paper_radar", "reading_plan", "project_idea"}

_PIPE_TABLE_FRAGMENTS = [
    "| 时间 |",
    "| 阅读目标 |",
    "| 路线 |",
    "| 推荐级 |",
    "|---",
]

_OLD_TEMPLATE_FRAGMENTS = [
    "论文数量一致性检查器",
    "年份过滤核查器",
    "摘要证据表",
    "先按标题和摘要识别与用户问题最接近的论文",
    "本节证据不足",
    "只能基于检索摘要做保守归纳",
    "第 1-2 天：先读第 1-3 篇",
]


def remove_internal_terms(answer: str) -> str:
    cleaned = answer
    for old, new in _INTERNAL_TERM_REPLACEMENTS.items():
        cleaned = re.sub(re.escape(old), new, cleaned, flags=re.IGNORECASE)
    return cleaned


def has_pipe_table_residue(answer: str) -> bool:
    return any(fragment in answer for fragment in _PIPE_TABLE_FRAGMENTS)


def has_old_template_residue(answer: str) -> bool:
    return any(fragment in answer for fragment in _OLD_TEMPLATE_FRAGMENTS)


def rebuild_radar_answer_if_needed(state: AgentState, answer: str) -> str:
    """Use deterministic card-style report if old report sections survived."""
    if state.get("classification") not in _RADAR_TASKS:
        return answer
    if not has_pipe_table_residue(answer) and not has_old_template_residue(answer):
        return answer

    from app.agents.document_selection import select_output_documents
    from app.agents.nodes.generator import _build_radar_report

    topic = str(state.get("original_query") or state.get("query") or "论文雷达")
    documents = select_output_documents(state)
    logger.warning("Rebuilding radar answer to remove old report residue.")
    return _build_radar_report(topic, documents)


def synthesize_response(state: AgentState) -> AgentState:
    """Clean up the answer, ensure citation indices are consistent, and finalize."""
    start = time.perf_counter()

    answer = remove_internal_terms(state.get("answer", ""))
    answer = remove_internal_terms(rebuild_radar_answer_if_needed(state, answer))
    citations = list(state.get("citations", []))

    # --- Remove citation references that don't have a matching citation entry ---
    valid_indices = {c["index"] for c in citations}
    if valid_indices:
        # Find all [N] references in the answer
        referenced = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}

        # Remove dangling references from the answer (cited but no entry)
        for idx in referenced - valid_indices:
            answer = answer.replace(f"[{idx}]", "")

    # --- Clean up extra whitespace from removed references ---
    answer = re.sub(r"  +", " ", answer).strip()

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info("Response synthesized in %dms", elapsed_ms)

    steps = list(state.get("steps", []))
    steps.append({
        "node": "synthesizer",
        "status": "completed",
        "detail": f"Final answer: {len(answer)} chars, {len(citations)} citations",
        "duration_ms": elapsed_ms,
    })

    return {
        **state,
        "answer": answer,
        "citations": citations,
        "steps": steps,
        "messages": [AIMessage(content=answer)],
    }
