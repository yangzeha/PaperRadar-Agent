"""Smoke test for PaperRadar report quality.

Run from ``backend``:
    python scripts/smoke_report_quality.py
"""

from __future__ import annotations

import asyncio
import re
import sys

from app.agents.graph import run_search


QUERY = "Agentic RAG 方向论文雷达：趋势、代表论文、研究空白和两周阅读路线"

REQUIRED_HEADINGS = [
    "# PaperRadar",
    "方向概览",
    "方法路线分类",
    "代表论文推荐",
    "近年趋势",
    "研究空白",
    "两周阅读路线",
    "可做小项目建议",
    "参考来源",
]

BANNED_TERMS = [
    "mock",
    "fallback",
    "当前 mock/fallback 模式",
    "本次结果只基于 mock",
    "当前 mock",
    "当前 fallback",
]

PIPE_TABLE_FRAGMENTS = [
    "| 时间 |",
    "| 阅读目标 |",
    "| 路线 |",
    "| 推荐级 |",
    "|---",
]

FORBIDDEN_PROJECTS = [
    "论文数量一致性检查器",
    "年份过滤核查器",
    "摘要证据表",
]


def _section(answer: str, index: int, title: str) -> str:
    next_index = index + 1
    next_pattern = rf"^##\s*{next_index}[.、]" if index < 8 else r"\Z"
    pattern = rf"(?ms)^##\s*{index}[.、]\s*{re.escape(title)}\s*(.*?)(?={next_pattern}|\Z)"
    match = re.search(pattern, answer)
    return match.group(1).strip() if match else ""


def validate(answer: str) -> list[str]:
    failures: list[str] = []

    for heading in REQUIRED_HEADINGS:
        if heading not in answer:
            failures.append(f"Missing heading/content: {heading}")

    for fragment in PIPE_TABLE_FRAGMENTS:
        if fragment in answer:
            failures.append(f"Answer contains raw pipe-table residue: {fragment}")

    route_section = _section(answer, 2, "方法路线分类")
    route_count = len(re.findall(r"###\s*路线\s+[A-Z]", route_section))
    if route_count < 4:
        failures.append(f"Expected at least 4 route cards, got {route_count}.")

    recommendation_section = _section(answer, 3, "代表论文推荐")
    if not re.search(r"###\s*(必读|重点)\s*\d+", recommendation_section):
        failures.append("Representative recommendation section lacks 必读/重点 cards.")

    gap_count = len(re.findall(r"###\s*Gap\s*\d+", answer, flags=re.IGNORECASE))
    if gap_count < 5:
        failures.append(f"Expected at least 5 concrete gaps, got {gap_count}.")

    project_count = len(re.findall(r"###\s*项目\s*\d+", answer))
    if project_count < 3:
        failures.append(f"Expected at least 3 project suggestions, got {project_count}.")

    citation_count = len(set(re.findall(r"\[(\d+)\]", answer)))
    if citation_count < 5:
        failures.append(f"Expected at least 5 citation markers, got {citation_count}.")

    lowered = answer.lower()
    for term in BANNED_TERMS:
        if term.lower() in lowered:
            failures.append(f"Answer contains banned term: {term}")

    reading_section = _section(answer, 6, "两周阅读路线")
    for day in ("Day 1-2", "Day 3-4", "Day 5-7", "Day 8-10", "Day 11-12", "Day 13-14"):
        if f"### {day}" not in reading_section:
            failures.append(f"Reading plan lacks {day}.")

    project_section = _section(answer, 7, "可做小项目建议")
    for title in FORBIDDEN_PROJECTS:
        if title in project_section:
            failures.append(f"Project section contains an internal validation tool: {title}")

    return failures


async def main() -> int:
    response = await run_search(
        query=QUERY,
        sources=["openalex"],
        max_results=6,
    )
    failures = validate(response.answer)
    if failures:
        print("PaperRadar report quality smoke test failed:")
        for failure in failures:
            print(f"- {failure}")
        print("\nAnswer preview:\n")
        print(response.answer[:2000])
        return 1

    print("PaperRadar report quality smoke test passed.")
    print(f"papers={len(response.papers)} citations={len(response.citations)}")
    print(response.answer[:1200])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
