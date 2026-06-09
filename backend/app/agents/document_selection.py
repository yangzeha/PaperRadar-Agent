"""Helpers for choosing the papers shown and cited in the final response."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.config import settings


def _paper_key(document: Mapping[str, Any]) -> str:
    url = str(document.get("url") or "").strip().lower()
    if url:
        return f"url:{url}"
    title = str(document.get("title") or "").strip().lower()
    return f"title:{title}"


def _can_fill_from_ungraded(document: Mapping[str, Any]) -> bool:
    """Only use fallback papers that have at least background relevance."""
    score = document.get("relevance_score")
    if score is None:
        return True
    try:
        value = float(score)
    except (TypeError, ValueError):
        return True
    if value > 1.0:
        value = value / 10.0
    return value >= 0.35


def select_output_documents(state: Mapping[str, Any]) -> list[dict]:
    """Return the papers that should be visible to the user.

    Relevance-graded papers are preferred, but the UI count should reflect the
    user's requested number when retrieval found enough plausible candidates.
    If the grader keeps fewer than requested, fill the remainder only from
    retrieved papers with a nonzero local relevance score.
    """
    raw_max_results = state.get("max_results") or settings.max_papers
    try:
        max_results = int(raw_max_results)
    except (TypeError, ValueError):
        max_results = settings.max_papers
    max_results = max(1, min(max_results, 99))

    selected: list[dict] = []
    seen: set[str] = set()

    for collection_name in ("graded_documents", "background_documents", "documents"):
        for document in state.get(collection_name, []) or []:
            if not isinstance(document, dict):
                continue
            if collection_name in {"background_documents", "documents"} and not _can_fill_from_ungraded(document):
                continue
            key = _paper_key(document)
            if key in seen:
                continue
            selected.append(document)
            seen.add(key)
            if len(selected) >= max_results:
                return selected

    return selected
