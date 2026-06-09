"""Smoke test that /api/search keeps non-RAG topics on topic.

Run from ``backend`` after the FastAPI server is running:
    python scripts/smoke_api_non_rag_topic_quality.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request


QUERY = "图对比学习推荐系统算法论文雷达"


def _section(answer: str, index: int) -> str:
    next_index = index + 1
    pattern = rf"(?ms)^##\s*{index}[.、]?\s+.*?(?=^##\s*{next_index}[.、]?\s+|\Z)"
    match = re.search(pattern, answer)
    return match.group(0) if match else ""


def _post_search(base_url: str) -> dict:
    payload = {
        "query": QUERY,
        "sources": ["openalex"],
        "max_results": 10,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/search",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def validate(response: dict) -> list[str]:
    failures: list[str] = []
    answer = str(response.get("answer") or "")
    lowered = answer.lower()

    if response.get("report_template_version") != "paper-radar-card-v3":
        failures.append("API did not return report_template_version=paper-radar-card-v3")

    required = [
        "Graph Contrastive Learning / SSL Recommendation",
        "LightGCL",
        "当前检索覆盖度",
        "建议补充检索关键词",
        "核心论文",
        "综述与背景",
        "相关但非核心",
    ]
    for text in required:
        if text not in answer:
            failures.append(f"API answer missing graph-rec content: {text}")

    forbidden = [
        "Planning / Reasoning / Iterative Retrieval",
        "Multimodal RAG",
        "Multi-Agent / Hierarchical RAG",
        "LangGraph 流程图",
        "citation grounding",
        "query rewrite",
        "论文数量一致性检查器",
        "年份过滤核查器",
        "摘要证据表",
        "与“",
        "所覆盖的",
    ]
    for text in forbidden:
        if text.lower() in lowered:
            failures.append(f"API answer contains off-topic/template residue: {text}")

    top_three = "\n".join(
        re.findall(r"(?ms)^###\s+(?:必读|重点|背景)\s+\d+：.*?(?=^###\s+|\Z)", _section(answer, 3))[:3]
    )
    for text in ["Knowledge Graph", "Multi-Objective", "Invariant Rationale"]:
        if text.lower() in top_three.lower():
            failures.append(f"Non-core paper appears in top three recommendations: {text}")

    top_four = "\n".join(
        re.findall(r"(?ms)^###\s+(?:必读|重点|背景)\s+\d+：.*?(?=^###\s+|\Z)", _section(answer, 3))[:4]
    )
    for text in ["Contrastive Self-supervised Learning in Recommender Systems", "LightGCL", "XSimGCL"]:
        if text.lower() not in top_four.lower():
            failures.append(f"Top four recommendations should include: {text}")

    return failures


def main() -> int:
    base_url = os.environ.get("PAPERRADAR_API_BASE", "http://127.0.0.1:8000")
    response = _post_search(base_url)
    failures = validate(response)
    if failures:
        print("Non-RAG API topic quality smoke failed:")
        for failure in failures:
            print(f"- {failure}")
        answer = str(response.get("answer") or "")
        print("\nAnswer preview:\n")
        print(answer[:2000])
        return 1

    print("Non-RAG API topic quality smoke passed.")
    print(f"url={base_url}/api/search")
    print(f"papers={len(response.get('papers') or [])}")
    print(_section(str(response.get("answer") or ""), 2)[:900])
    return 0


if __name__ == "__main__":
    sys.exit(main())
