"""Tests for provider URL normalization and status helpers."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage


def test_normalize_openai_compatible_urls():
    from app.services.llm_provider import normalize_chat_completions_url

    assert (
        normalize_chat_completions_url("https://api.deepseek.com")
        == "https://api.deepseek.com/chat/completions"
    )
    assert (
        normalize_chat_completions_url("https://api.deepseek.com/v1")
        == "https://api.deepseek.com/v1/chat/completions"
    )
    assert (
        normalize_chat_completions_url("https://api.deepseek.com/chat/completions")
        == "https://api.deepseek.com/chat/completions"
    )
    assert (
        normalize_chat_completions_url("https://dashscope.aliyuncs.com")
        == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    assert (
        normalize_chat_completions_url(
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )


def test_provider_balance_error_falls_back_to_mock(monkeypatch):
    from app.config import settings
    from app.services import llm_provider
    from app.services.llm_provider import extract_text, invoke_with_retry

    monkeypatch.setattr(settings, "llm_provider", "deepseek")
    monkeypatch.setattr(settings, "deepseek_api_key", "fake-key")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "llm_model_id", "")
    monkeypatch.setattr(settings, "llm_base_url", "")
    monkeypatch.setattr(
        llm_provider,
        "_invoke_openai_compatible",
        lambda **_: (_ for _ in ()).throw(
            RuntimeError('HTTP 402: {"error":{"message":"Insufficient Balance"}}')
        ),
    )

    response = invoke_with_retry([HumanMessage(content="User: hello")])

    assert "PaperRadar-Agent" in extract_text(response)


def test_provider_timeout_falls_back_and_is_cached(monkeypatch):
    from app.config import settings
    from app.services import llm_provider
    from app.services.llm_provider import extract_text, invoke_with_retry

    calls = 0

    def fake_invoke(**_):
        nonlocal calls
        calls += 1
        raise RuntimeError("provider request timed out after 1s")

    llm_provider._UNAVAILABLE_UNTIL.clear()
    monkeypatch.setattr(settings, "llm_provider", "deepseek")
    monkeypatch.setattr(settings, "deepseek_api_key", "fake-key")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "llm_model_id", "")
    monkeypatch.setattr(settings, "llm_base_url", "")
    monkeypatch.setattr(settings, "llm_unavailable_cache_seconds", 300)
    monkeypatch.setattr(llm_provider, "_invoke_openai_compatible", fake_invoke)

    first = invoke_with_retry([HumanMessage(content="User: hello")])
    second = invoke_with_retry([HumanMessage(content="User: hello")])

    assert "PaperRadar-Agent" in extract_text(first)
    assert "PaperRadar-Agent" in extract_text(second)
    assert calls == 1
    llm_provider._UNAVAILABLE_UNTIL.clear()


def test_qwen_provider_tries_next_model_on_quota_error(monkeypatch):
    from app.config import settings
    from app.services import llm_provider
    from app.services.llm_provider import extract_text, invoke_with_retry

    attempted_models: list[str] = []

    def fake_invoke(**kwargs):
        attempted_models.append(kwargs["model"])
        if kwargs["model"] == "qwen-plus":
            raise RuntimeError('HTTP 429: {"message":"quota exceeded"}')
        return llm_provider.AIMessage(content="第二个模型成功")

    monkeypatch.setattr(settings, "llm_provider", "qwen")
    monkeypatch.setattr(settings, "dashscope_api_key", "fake-key")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "llm_model_id", "qwen-plus")
    monkeypatch.setattr(settings, "llm_base_url", "")
    monkeypatch.setattr(settings, "qwen_model_sequence", "qwen-plus,qwen-turbo")
    monkeypatch.setattr(llm_provider, "_invoke_openai_compatible", fake_invoke)

    response = invoke_with_retry([HumanMessage(content="User: hello")])

    assert extract_text(response) == "第二个模型成功"
    assert attempted_models == ["qwen-plus", "qwen-turbo"]


def test_qwen_arrearage_error_falls_back_to_mock_after_sequence(monkeypatch):
    from app.config import settings
    from app.services import llm_provider
    from app.services.llm_provider import extract_text, invoke_with_retry

    monkeypatch.setattr(settings, "llm_provider", "qwen")
    monkeypatch.setattr(settings, "dashscope_api_key", "fake-key")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "llm_model_id", "qwen-plus")
    monkeypatch.setattr(settings, "llm_base_url", "")
    monkeypatch.setattr(settings, "qwen_model_sequence", "qwen-plus,qwen-turbo")
    monkeypatch.setattr(
        llm_provider,
        "_invoke_openai_compatible",
        lambda **_: (_ for _ in ()).throw(
            RuntimeError(
                'HTTP 400: {"error":{"message":"Access denied",'
                '"type":"Arrearage","code":"Arrearage"}}'
            )
        ),
    )

    response = invoke_with_retry([HumanMessage(content="User: hello")])

    assert "PaperRadar-Agent" in extract_text(response)


def test_provider_non_account_error_is_not_hidden(monkeypatch):
    from app.config import settings
    from app.services import llm_provider
    from app.services.llm_provider import invoke_with_retry

    llm_provider._UNAVAILABLE_UNTIL.clear()
    monkeypatch.setattr(settings, "llm_provider", "deepseek")
    monkeypatch.setattr(settings, "deepseek_api_key", "fake-key")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "llm_model_id", "")
    monkeypatch.setattr(settings, "llm_base_url", "")
    monkeypatch.setattr(
        llm_provider,
        "_invoke_openai_compatible",
        lambda **_: (_ for _ in ()).throw(RuntimeError("unexpected parser bug")),
    )

    with pytest.raises(RuntimeError, match="unexpected parser bug"):
        invoke_with_retry([HumanMessage(content="User: hello")])
