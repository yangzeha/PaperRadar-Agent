"""Smoke test for the configured LLM provider.

Exit code 0 means either PASS or a deliberate SKIP caused by missing API key.
Unexpected provider/runtime errors raise and return a non-zero exit code.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from langchain_core.messages import HumanMessage

from app.config import settings
from app.services.llm_provider import (
    extract_text,
    invoke_with_retry,
    normalize_chat_completions_url,
    resolve_provider_config,
)


def main() -> None:
    provider = settings.llm_provider.strip().lower()
    if provider not in {"mock", "deepseek", "qwen", "gemini"}:
        raise SystemExit(f"FAIL: unsupported LLM_PROVIDER={provider!r}")

    cfg = resolve_provider_config()
    key_present = bool(cfg.api_key)
    normalized_base_url = (
        normalize_chat_completions_url(cfg.base_url)
        if cfg.base_url and provider in {"deepseek", "qwen"}
        else cfg.base_url
    )
    print(
        f"provider={cfg.provider} model={cfg.model} "
        f"base_url={normalized_base_url or '(none)'} key_present={str(key_present).lower()}"
    )

    if provider != "mock" and not key_present:
        print(f"SKIP: {provider} provider selected but {cfg.key_env} is not set.")
        return

    try:
        response = invoke_with_retry(
            [HumanMessage(content="Reply with exactly: provider smoke ok")]
        )
    except Exception as exc:
        raise SystemExit(f"FAIL: provider call failed: {str(exc)[:1000]}") from exc
    text = extract_text(response).strip()
    if not text:
        raise SystemExit("FAIL: provider returned an empty response.")

    print("PASS: provider smoke completed.")
    print(f"response_preview={text[:200]}")


if __name__ == "__main__":
    main()
