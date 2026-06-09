"""Smoke test the real /api/search endpoint report shape.

Run from ``backend`` while the FastAPI server is running:
    python scripts/smoke_api_report_quality.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any


QUERY = "Agentic RAG 方向论文雷达：趋势、代表论文、研究空白和两周阅读路线"
EXPECTED_TEMPLATE_VERSION = "paper-radar-card-v3"

BANNED_TERMS = [
    "mock",
    "fallback",
    "当前 mock",
    "当前 fallback",
    "当前 mock/fallback 模式",
    "论文数量一致性检查器",
    "年份过滤核查器",
    "摘要证据表",
    "先按标题和摘要识别与用户问题最接近的论文，再把它们分成可优先阅读和需要二次核查两类",
    "本节证据不足",
    "只能基于检索摘要做保守归纳",
    "第 1-2 天：先读第 1-3 篇",
]

REQUIRED_TERMS = [
    "### 路线 A",
    "### 路线 B",
    "### 必读 1",
    "### Day 1-2",
    "### Day 3-4",
    "### Day 5-7",
    "### Gap 1",
    "### 项目 1",
]

PIPE_TABLE_FRAGMENTS = [
    "| 时间 |",
    "| 阅读目标 |",
    "| 路线 |",
    "| 推荐级 |",
    "|---",
]


def _request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=360) as response:
        return json.loads(response.read().decode("utf-8"))


def _provider_status(base_url: str) -> dict[str, Any]:
    try:
        return _request_json(f"{base_url}/api/provider")
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"provider": "unknown", "model": "unknown", "has_api_key": False, "error": str(exc)}


def _section(answer: str, index: int, title: str) -> str:
    next_index = index + 1
    next_pattern = rf"^##\s*{next_index}[.、]" if index < 8 else r"\Z"
    pattern = rf"(?ms)^##\s*{index}[.、]\s*{re.escape(title)}\s*(.*?)(?={next_pattern}|\Z)"
    match = re.search(pattern, answer)
    return match.group(1).strip() if match else ""


def _validate_response(data: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    answer = str(data.get("answer") or "")

    if data.get("report_template_version") != EXPECTED_TEMPLATE_VERSION:
        failures.append(
            "report_template_version expected "
            f"{EXPECTED_TEMPLATE_VERSION!r}, got {data.get('report_template_version')!r}"
        )

    lowered = answer.lower()
    for term in BANNED_TERMS:
        if term.lower() in lowered:
            failures.append(f"answer contains banned term: {term}")

    for term in REQUIRED_TERMS:
        if term not in answer:
            failures.append(f"answer missing required card marker: {term}")

    route_section = _section(answer, 2, "方法路线分类")
    recommendation_section = _section(answer, 3, "代表论文推荐")
    reading_section = _section(answer, 6, "两周阅读路线")
    for section_name, section in [
        ("第 2 节", route_section),
        ("第 3 节", recommendation_section),
        ("第 6 节", reading_section),
    ]:
        if not section:
            failures.append(f"{section_name} missing or not parseable")
            continue
        for fragment in PIPE_TABLE_FRAGMENTS:
            if fragment in section:
                failures.append(f"{section_name} contains Markdown table residue: {fragment}")

    return failures


def main() -> int:
    base_url = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
    provider = _provider_status(base_url)
    payload = {
        "query": QUERY,
        "sources": ["openalex"],
        "max_results": 10,
    }

    try:
        data = _request_json(f"{base_url}/api/search", payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:2000]
        print("API report quality smoke test failed before validation:")
        print(f"provider={provider.get('provider')}")
        print(f"model={provider.get('model')}")
        print(f"key_present={bool(provider.get('has_api_key'))}")
        print(f"error=HTTP {exc.code}: {body}")
        return 1
    except Exception as exc:
        print("API report quality smoke test failed before validation:")
        print(f"provider={provider.get('provider')}")
        print(f"model={provider.get('model')}")
        print(f"key_present={bool(provider.get('has_api_key'))}")
        print(f"error={exc}")
        return 1

    failures = _validate_response(data)
    if failures:
        print("API report quality smoke test failed:")
        for failure in failures:
            print(f"- {failure}")
        print("\nAnswer preview:\n")
        print(str(data.get("answer") or "")[:2000])
        return 1

    print("API report quality smoke test passed.")
    print(f"provider={provider.get('provider')}")
    print(f"model={provider.get('model')}")
    print(f"key_present={bool(provider.get('has_api_key'))}")
    print(f"report_template_version={data.get('report_template_version')}")
    print(f"papers={len(data.get('papers') or [])} citations={len(data.get('citations') or [])}")
    print(str(data.get("answer") or "")[:1200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
