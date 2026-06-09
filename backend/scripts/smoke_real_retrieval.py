"""Real DeepSeek/Qwen/Gemini + arXiv/PubMed + Chroma smoke test."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings
from app.services.llm_provider import (
    normalize_chat_completions_url,
    resolve_provider_config,
)
from app.services.memory_store import list_history


QUERY = "long-term memory in LLM agents"
REQUIRED_SECTIONS = [
    "方向概览",
    "方法路线分类",
    "代表论文推荐",
    "近年趋势",
    "研究空白",
    "两周阅读路线",
    "可做小项目建议",
    "参考来源",
]
REQUIRED_RAG_MODULES = [
    ("arxiv", "arxiv"),
    ("langchain_chroma", "langchain-chroma"),
    ("chromadb", "chromadb"),
    ("langchain_community", "langchain-community"),
    ("sentence_transformers", "sentence-transformers"),
]


def _missing_modules() -> list[str]:
    missing = []
    for module_name, package_name in REQUIRED_RAG_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    return missing


async def _source_status(query: str) -> tuple[list[str], dict[str, str]]:
    from app.services.paper_fetcher import PaperFetcher

    fetcher = PaperFetcher()
    statuses: dict[str, str] = {}
    sources = ["arxiv"]

    try:
        arxiv_papers = await fetcher.search_arxiv(query, max_results=3)
        if arxiv_papers and any(p.source == "arxiv_openalex" for p in arxiv_papers):
            statuses["arxiv_status"] = "ok_openalex_fallback"
        elif arxiv_papers:
            statuses["arxiv_status"] = "ok"
        else:
            statuses["arxiv_status"] = "failed_zero_results"
    except Exception as exc:
        statuses["arxiv_status"] = f"failed: {str(exc)[:200]}"

    try:
        pubmed_papers = await fetcher.search_pubmed(query, max_results=3)
        if pubmed_papers:
            statuses["pubmed_status"] = "ok"
            sources.append("pubmed")
        else:
            statuses["pubmed_status"] = "skipped_no_results"
    except Exception as exc:
        statuses["pubmed_status"] = f"failed: {str(exc)[:200]}"

    return sources, statuses


async def _run() -> None:
    cfg = resolve_provider_config()
    if cfg.provider == "mock":
        print("SKIP: real retrieval smoke requires LLM_PROVIDER=deepseek, qwen, or gemini.")
        return
    if cfg.provider not in {"deepseek", "qwen", "gemini"}:
        raise SystemExit(f"FAIL: unsupported LLM_PROVIDER={cfg.provider!r}")
    if not cfg.api_key:
        print(f"SKIP: {cfg.provider} provider selected but {cfg.key_env} is not set.")
        return

    missing = _missing_modules()
    if missing:
        print(
            "FAIL: real RAG dependencies are missing. Install with: "
            "python -m pip install -e \".[rag]\". "
            "Minimal packages: chromadb sentence-transformers arxiv "
            "langchain-chroma langchain-community. "
            f"Missing: {', '.join(missing)}"
        )
        raise SystemExit(2)

    sources, source_status = await _source_status(QUERY)
    if not source_status.get("arxiv_status", "").startswith("ok"):
        raise SystemExit(f"FAIL: arXiv preflight failed: {source_status}")

    from app.agents.graph import run_search

    history_before = len(list_history(limit=500).get("history", []))
    tmp_dir = tempfile.mkdtemp(prefix="paper-radar-chroma-")
    try:
        settings.chroma_persist_dir = tmp_dir
        settings.chroma_collection_name = "paper_radar_real_smoke"
        settings.top_k_results = 5
        response = await run_search(
            query=QUERY,
            sources=sources,
            max_results=5,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    history_after = len(list_history(limit=500).get("history", []))

    missing_sections = [
        section for section in REQUIRED_SECTIONS if section not in response.answer
    ]
    result = {
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": (
            normalize_chat_completions_url(cfg.base_url)
            if cfg.provider in {"deepseek", "qwen"}
            else cfg.base_url
        ),
        "key_present": bool(cfg.api_key),
        **source_status,
        "sources_used": sources,
        "classification": response.classification,
        "papers": len(response.papers),
        "citations": len(response.citations),
        "steps": len(response.steps),
        "history_written": history_after > history_before,
        "missing_sections": missing_sections,
        "answer_preview": response.answer[:800],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if len(response.papers) < 3:
        raise SystemExit(f"FAIL: expected at least 3 papers, got {len(response.papers)}.")
    if len(response.citations) < 1:
        raise SystemExit(f"FAIL: expected at least 1 citation, got {len(response.citations)}.")
    if not response.steps:
        raise SystemExit("FAIL: expected LangGraph steps.")
    if missing_sections:
        raise SystemExit(f"FAIL: missing required sections: {missing_sections}.")
    if history_after <= history_before:
        raise SystemExit("FAIL: reading_history.json was not updated.")

    print("PASS: real retrieval smoke completed.")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
