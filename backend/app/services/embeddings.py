"""Singleton embedding model backed by HuggingFace sentence-transformers."""

from __future__ import annotations

import logging
import os
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_embeddings_instance: Any | None = None


def get_embeddings() -> Any:
    """Return a cached HuggingFaceEmbeddings instance."""
    global _embeddings_instance

    if _embeddings_instance is None:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "langchain-community and sentence-transformers are required "
                "for real embedding retrieval. Install with: "
                "python -m pip install -e \".[rag]\""
            ) from exc

        logger.info("Loading embedding model: %s", settings.embedding_model)
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            show_progress=False,
        )
        logger.info("Embedding model loaded successfully")

    return _embeddings_instance
