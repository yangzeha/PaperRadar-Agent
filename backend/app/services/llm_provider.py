"""LLM provider factory for Gemini, DeepSeek, Qwen, and deterministic mock."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from app.config import settings

logger = logging.getLogger(__name__)

_UNAVAILABLE_UNTIL: dict[str, float] = {}


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    api_key: str
    base_url: str
    key_env: str


_PROVIDER_KEY_ENVS = {
    "mock": "",
    "deepseek": "LLM_API_KEY or DEEPSEEK_API_KEY",
    "qwen": "LLM_API_KEY or DASHSCOPE_API_KEY",
    "gemini": "LLM_API_KEY or GOOGLE_API_KEY",
}


def extract_text(response: BaseMessage) -> str:
    """Extract text content from an LLM response."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def resolve_provider_config(
    provider_override: str | None = None,
    model_override: str | None = None,
) -> ProviderConfig:
    """Resolve provider config with generic env vars taking priority."""
    provider = (provider_override or settings.llm_provider).strip().lower()
    generic_model = settings.llm_model_id.strip()
    generic_key = settings.llm_api_key.strip()
    generic_base_url = settings.llm_base_url.strip()

    if provider == "mock":
        return ProviderConfig(
            provider="mock",
            model=model_override or generic_model or "mock",
            api_key="",
            base_url="",
            key_env="",
        )

    if provider == "deepseek":
        return ProviderConfig(
            provider=provider,
            model=model_override or generic_model or settings.deepseek_model,
            api_key=generic_key or settings.deepseek_api_key,
            base_url=generic_base_url or settings.deepseek_base_url,
            key_env="LLM_API_KEY or DEEPSEEK_API_KEY",
        )

    if provider == "qwen":
        return ProviderConfig(
            provider=provider,
            model=model_override or generic_model or settings.qwen_model,
            api_key=settings.dashscope_api_key or generic_key,
            base_url=generic_base_url or settings.qwen_base_url,
            key_env="LLM_API_KEY or DASHSCOPE_API_KEY",
        )

    if provider == "gemini":
        return ProviderConfig(
            provider=provider,
            model=model_override or generic_model or settings.primary_model,
            api_key=generic_key or settings.google_api_key,
            base_url=generic_base_url,
            key_env="LLM_API_KEY or GOOGLE_API_KEY",
        )

    return ProviderConfig(
        provider=provider,
        model=model_override or generic_model,
        api_key=generic_key,
        base_url=generic_base_url,
        key_env="LLM_API_KEY",
    )


def provider_api_key(provider: str | None = None) -> str:
    """Return the configured API key for a provider without exposing it."""
    return resolve_provider_config(provider).api_key


def provider_status() -> dict[str, object]:
    """Return provider metadata for API/UI display."""
    cfg = resolve_provider_config()
    is_mock = cfg.provider == "mock"
    model_sequence = _model_sequence_for_provider(cfg.provider, cfg.model)
    return {
        "provider": cfg.provider,
        "mode": "mock" if is_mock else "real",
        "model": cfg.model,
        "model_sequence": model_sequence,
        "base_url": cfg.base_url,
        "has_api_key": bool(cfg.api_key),
        "label": "当前为 mock 演示模式" if is_mock else f"当前为 {cfg.provider} real 模式",
        "key_env": _PROVIDER_KEY_ENVS.get(cfg.provider, cfg.key_env),
    }


def normalize_chat_completions_url(base_url: str) -> str:
    """Normalize provider base URLs to a chat completions endpoint."""
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if "dashscope.aliyuncs.com" in cleaned and "compatible-mode/v1" not in cleaned:
        cleaned = f"{cleaned}/compatible-mode/v1"
    return f"{cleaned}/chat/completions"


def get_llm(model_override: str | None = None) -> BaseChatModel:
    """Return a Gemini LangChain chat model for the Gemini path."""
    cfg = resolve_provider_config("gemini", model_override)
    if not cfg.api_key:
        raise RuntimeError("No Gemini API key configured. Set LLM_API_KEY or GOOGLE_API_KEY.")

    from langchain_google_genai import ChatGoogleGenerativeAI

    logger.info("Using Gemini model: %s", cfg.model)
    return ChatGoogleGenerativeAI(
        model=cfg.model,
        google_api_key=cfg.api_key,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_tokens,
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item if isinstance(item, str) else str(item.get("text", ""))
            for item in content
            if isinstance(item, (str, dict))
        )
    return str(content)


def _message_role(message: BaseMessage) -> str:
    msg_type = getattr(message, "type", "")
    if msg_type == "human":
        return "user"
    if msg_type == "ai":
        return "assistant"
    if msg_type == "system":
        return "system"
    return "user"


def _messages_to_openai(messages: list[BaseMessage]) -> list[dict[str, str]]:
    return [
        {"role": _message_role(message), "content": _content_to_text(message.content)}
        for message in messages
    ]


def _summarize_http_error(exc: httpx.HTTPStatusError) -> str:
    body = exc.response.text[:1000] if exc.response is not None else ""
    message = f"HTTP {exc.response.status_code}: {body}" if exc.response else str(exc)
    return _redact_secrets(message)


def _redact_secrets(text: str) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-<redacted>", text)
    text = re.sub(r"apikey=[^&\\s']+", "apikey=<redacted>", text, flags=re.IGNORECASE)
    return text


def _invoke_openai_compatible(
    *,
    messages: list[BaseMessage],
    api_key: str,
    base_url: str,
    model: str,
) -> BaseMessage:
    """Call an OpenAI-compatible chat completion endpoint."""
    url = normalize_chat_completions_url(base_url)
    payload = {
        "model": model,
        "messages": _messages_to_openai(messages),
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }
    try:
        response = httpx.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=settings.llm_request_timeout,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(_summarize_http_error(exc)) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"provider request timed out after {settings.llm_request_timeout:g}s"
        ) from exc
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return AIMessage(content=content)


def _split_model_sequence(value: str) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for raw_model in value.split(","):
        model = raw_model.strip()
        if not model or model in seen:
            continue
        seen.add(model)
        models.append(model)
    return models


def _model_sequence_for_provider(provider: str, primary_model: str) -> list[str]:
    if provider == "qwen":
        configured = _split_model_sequence(settings.qwen_model_sequence)
        return _split_model_sequence(",".join([primary_model, *configured]))
    return [primary_model] if primary_model else []


def _should_try_next_model(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in [
            "http 400",
            "http 402",
            "http 404",
            "http 429",
            "insufficient balance",
            "insufficient_quota",
            "resource_exhausted",
            "quota",
            "billing",
            "rate limit",
            "timed out",
            "timeout",
            "model",
            "not exist",
            "not found",
            "expired",
        ]
    )


def _invoke_openai_compatible_with_model_fallback(
    *,
    messages: list[BaseMessage],
    api_key: str,
    base_url: str,
    models: list[str],
) -> BaseMessage:
    errors: list[str] = []
    for index, model in enumerate(models, start=1):
        try:
            logger.info("Trying OpenAI-compatible model %s (%d/%d)", model, index, len(models))
            response = _invoke_openai_compatible(
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                model=model,
            )
            logger.info("OpenAI-compatible model %s succeeded", model)
            return response
        except RuntimeError as exc:
            message = _redact_secrets(str(exc))
            errors.append(f"{model}: {message}")
            if not _should_try_next_model(exc) or index == len(models):
                raise RuntimeError("; ".join(errors)) from exc
            logger.warning("Model %s failed; trying next model: %s", model, message[:200])

    raise RuntimeError("; ".join(errors) or "No OpenAI-compatible models configured.")


def _full_prompt(messages: list[BaseMessage]) -> str:
    return "\n\n".join(_content_to_text(message.content) for message in messages)


def _extract_query(prompt: str) -> str:
    match = re.search(r"Query:\s*(.+?)(?:\n\n|\n[A-Z][A-Za-z ]+:|$)", prompt, re.S)
    if match:
        return match.group(1).strip()
    match = re.search(r"用户问题：\s*(.+?)(?:\n\n|$)", prompt, re.S)
    if match:
        return match.group(1).strip()
    match = re.search(r"User:\s*(.+?)(?:\n\n|$)", prompt, re.S)
    if match:
        return match.group(1).strip()
    return ""


def _mock_router(prompt: str) -> str:
    query = _extract_query(prompt).lower()
    radar_words = [
        "radar",
        "trend",
        "gap",
        "research direction",
        "reading route",
        "reading plan",
        "project idea",
        "paper radar",
        "long-term memory in llm agents",
        "论文雷达",
        "趋势",
        "研究空白",
        "阅读路线",
        "小项目",
        "选题",
    ]
    paper_words = [
        "paper",
        "literature",
        "research",
        "arxiv",
        "pubmed",
        "survey",
        "论文",
        "综述",
        "研究",
    ]
    if any(word in query for word in radar_words):
        return "paper_radar"
    if any(word in query for word in paper_words):
        return "paper_search"
    return "general"


def _mock_grader(prompt: str) -> str:
    paper_count = len(re.findall(r"Paper\s+\d+:", prompt))
    paper_count = paper_count or len(re.findall(r"^\[\d+\]", prompt, re.M)) or 3
    return "\n".join(f"{i}: yes" for i in range(1, paper_count + 1))


def _extract_prompt_papers(prompt: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r"^\[(\d+)\]\s*(?P<title>.+?)\n"
        r"URL:\s*(?P<url>.*?)\n"
        r"Published:\s*(?P<published>.*?)\n"
        r"Abstract:\s*(?P<abstract>.*?)(?=\n\n\[\d+\]|\Z)",
        re.M | re.S,
    )
    papers: list[dict[str, str]] = []
    for match in pattern.finditer(prompt):
        papers.append(
            {
                "index": match.group(1),
                "title": re.sub(r"\s+", " ", match.group("title")).strip(),
                "url": match.group("url").strip(),
                "published": match.group("published").strip() or "年份未知",
                "abstract": re.sub(r"\s+", " ", match.group("abstract")).strip(),
            }
        )
    return papers


def _mock_excerpt(text: str, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return "摘要信息不足，建议打开原文进一步核对。"
    sentence = re.match(r"(.+?[.!?。！？])(?:\s|$)", text)
    value = sentence.group(1) if sentence else text
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "..."


def _mock_reference_span(papers: list[dict[str, str]], limit: int = 4) -> str:
    if not papers:
        return ""
    return " ".join(f"[{paper['index']}]" for paper in papers[:limit])


def _mock_paper_search(prompt: str) -> str:
    query = _extract_query(prompt) or "论文检索"
    papers = _extract_prompt_papers(prompt)
    if not papers:
        return "本次没有收到可用论文摘要，因此无法生成可靠的论文清单。"

    lines = [
        f"下面是围绕“{query}”检索到的 {len(papers)} 篇论文。"
        "我只根据标题、年份和摘要做保守概括，具体贡献需要打开原文确认。",
        "",
        "## 检索到的论文",
    ]
    for paper in papers:
        lines.append(
            f"- [{paper['index']}] **{paper['title']}**"
            f"（{paper['published']}）：{_mock_excerpt(paper['abstract'])}"
        )
    return "\n".join(lines)


def _mock_paper_radar(prompt: str) -> str:
    query = _extract_query(prompt) or "LLM Agent long-term memory"
    papers = _extract_prompt_papers(prompt)
    if not papers:
        papers = [
            {
                "index": "1",
                "title": "示例论文 1",
                "url": "",
                "published": "年份未知",
                "abstract": "当前没有收到真实论文摘要。",
            },
            {
                "index": "2",
                "title": "示例论文 2",
                "url": "",
                "published": "年份未知",
                "abstract": "当前没有收到真实论文摘要。",
            },
        ]

    year_values = sorted(
        {
            match.group(0)
            for paper in papers
            for match in re.finditer(r"(19|20)\d{2}", paper["published"])
        }
    )
    year_text = "、".join(year_values) if year_values else "本次结果未提供稳定年份"
    refs = _mock_reference_span(papers)
    paper_refs = [f"[{paper['index']}]" for paper in papers]

    def ref_at(position: int) -> str:
        if not paper_refs:
            return "[1]"
        return paper_refs[min(position, len(paper_refs) - 1)]

    def title_at(position: int) -> str:
        if not papers:
            return "当前论文"
        paper = papers[min(position, len(papers) - 1)]
        return f"[{paper['index']}] {paper['title']}"

    recommendation_cards = []
    for rank, paper in enumerate(papers[:5], start=1):
        label = f"必读 {rank}" if rank <= 2 else f"重点 {rank}"
        recommendation_cards.append(
            f"### {label}：[{paper['index']}] {paper['title']}\n"
            f"- **年份 / 来源**：{paper['published']}，当前检索来源\n"
            "- **所属路线**：Agentic RAG / 相关 RAG 方法\n"
            f"- **为什么推荐**：这篇论文和“{query}”相关，可帮助建立方向入口、方法路线或评测问题意识。\n"
            f"- **适合先读吗**：{'适合，建议第一轮先读。' if rank <= 2 else '适合第二轮重点读。'}\n"
            "- **可产出**：整理一条论文贡献笔记，并标注它对应的系统模块或评测指标。"
        )

    background_lines = "\n".join(
        f"- [{paper['index']}] {paper['title']}：适合作为背景阅读，补充主题相关证据。"
        for paper in papers[5:8]
    )
    if background_lines:
        recommendation_cards.append(f"### 背景阅读\n{background_lines}")

    return f"""# PaperRadar：{query}

## 1. 方向概览
本次检索围绕“{query}”返回 {len(papers)} 篇候选论文。下面的分析只基于这些论文标题和摘要做保守整理，适合作为第一轮论文雷达，而不是最终综述结论 {refs}。

## 2. 方法路线分类
### 路线 A：Survey / Taxonomy / SoK
- **核心问题**：如何定义 Agentic RAG，并划分检索、规划、记忆、工具调用和评测边界。
- **代表论文**：{title_at(0)}。
- **主要思路**：先建立术语和系统模块地图，再判断后续论文分别解决哪个子问题。
- **优点**：适合建立全局概念框架，适合作为第一批阅读材料。
- **局限**：偏综述和框架整理，具体可复现系统仍需继续阅读方法论文。

### 路线 B：Planning / Reasoning / Iterative Retrieval
- **核心问题**：如何让 RAG 根据中间证据动态改写 query、规划检索策略并迭代修正答案。
- **代表论文**：{title_at(1)}。
- **主要思路**：引入 query rewrite、retrieval strategy、evidence integration 和 response verification。
- **优点**：更贴近 LangGraph 的检索-评分-重写-再检索工作流。
- **局限**：会增加调用成本、延迟和失败传播风险。

### 路线 C：Multi-Agent / Hierarchical RAG
- **核心问题**：如何用 planner、researcher、critic、writer 等角色协作完成复杂研究问题。
- **代表论文**：{title_at(2)}。
- **主要思路**：把文献研究拆成多个角色节点，并通过共享状态组织证据。
- **优点**：适合做可演示的科研 Agent 项目。
- **局限**：需要额外评估多 Agent 是否真的优于单 Agent。

### 路线 D：Evaluation / Benchmark
- **核心问题**：如何评估 citation accuracy、groundedness、answer faithfulness 和 retrieval precision。
- **代表论文**：{title_at(3)}。
- **主要思路**：构造问题集、证据集和自动评分流程，比较不同 RAG 流程。
- **优点**：容易形成工程闭环和简历项目亮点。
- **局限**：指标设计容易和真实用户价值脱节。

## 3. 代表论文推荐
{chr(10).join(recommendation_cards)}

## 4. 近年趋势
本次结果覆盖年份：{year_text}。建议优先比较较新论文的问题设定、数据集、指标和是否提供开源实现，再回看早期论文的基础方法 {refs}。

## 5. 研究空白
### Gap 1：Agentic RAG 评测指标不统一
- 现状：当前论文通常分别讨论检索质量、生成质量或系统架构 {ref_at(0)}。
- 缺口：缺少同时覆盖检索决策、引用准确率和答案忠实度的统一指标。
- 可验证方式：构造同一批问题，对比不同 RAG 流程的 citation accuracy 和 groundedness。
- 可做项目：做一个 LangGraph RAG 评测小面板。

### Gap 2：多 Agent 协作的成本边界不清楚
- 现状：多角色协作能提升流程可解释性，但会增加调用次数 {ref_at(1)}。
- 缺口：什么时候多 Agent 真正优于单 Agent RAG 还缺少清晰边界。
- 可验证方式：比较单 Agent、双 Agent、三 Agent 的耗时、token 成本和答案质量。
- 可做项目：实现 planner-researcher-critic 对照实验。

### Gap 3：长期记忆污染与遗忘机制
- 现状：长期记忆能保存用户偏好和已读论文，但错误记忆也可能被反复使用 {ref_at(2)}。
- 缺口：缺少 memory write、memory decay、memory correction 的可解释策略。
- 可验证方式：注入正确/错误记忆，观察后续回答是否被污染。
- 可做项目：做一个 memory audit 节点。

### Gap 4：Query Rewrite 的收益边界
- 现状：迭代检索和 query rewrite 常被用于提升召回 {ref_at(3)}。
- 缺口：缺少判断何时继续改写、何时停止检索的实用准则。
- 可验证方式：比较 0/1/3 次 rewrite 的召回率和噪声比例。
- 可做项目：做一个 Query Rewrite A/B 对比工具。

### Gap 5：Citation grounding 自动检查
- 现状：报告型 RAG 需要把每个 claim 绑定到可追溯证据 {ref_at(4)}。
- 缺口：很多系统只给引用编号，不检查引用是否真的支撑句子。
- 可验证方式：抽取 answer claim，逐条匹配摘要或原文片段。
- 可做项目：实现 Citation Grounding Checker。

## 6. 两周阅读路线
### Day 1-2：建立方向全景
- **阅读目标**：理解 Agentic RAG 和普通 RAG 的区别，掌握 planner、retriever、grader、rewriter、memory、citation checker 等核心概念。
- **推荐论文**：{ref_at(0)}、{ref_at(1)}
- **产出**：画一张 Agentic RAG 系统模块图，整理 10 个核心术语。

### Day 3-4：理解规划式检索和 Query Rewrite
- **阅读目标**：理解模型如何根据检索结果动态改写 query、重新检索和整合证据。
- **推荐论文**：{ref_at(1)}、{ref_at(2)}
- **产出**：写出一个 LangGraph 检索-评分-重写-再检索流程图。

### Day 5-7：理解多 Agent / 分层 RAG
- **阅读目标**：理解 planner、researcher、critic、writer 等角色如何协同完成复杂问题。
- **推荐论文**：{ref_at(2)}、{ref_at(3)}
- **产出**：总结多 Agent RAG 和单 Agent RAG 的优势、成本和失败点。

### Day 8-10：关注评测与幻觉检测
- **阅读目标**：理解 citation accuracy、groundedness、answer faithfulness、retrieval precision 等指标。
- **推荐论文**：{ref_at(3)}、{ref_at(4)}
- **产出**：设计一个小型 Agentic RAG 评测表。

### Day 11-12：寻找研究空白和可落地场景
- **阅读目标**：从评测、长期记忆、多模态、垂直领域中选择一个小切口。
- **推荐论文**：{ref_at(0)}、{ref_at(4)}
- **产出**：形成 2-3 个小项目候选。

### Day 13-14：形成小项目方案
- **阅读目标**：把前面读到的论文思想落到一个两周可完成的 demo。
- **推荐方向**：LangGraph Agentic RAG Evaluator / Citation Grounding Checker / Long-term Memory RAG Agent
- **产出**：README、架构图、Demo 问题、简历项目描述。

## 7. 可做小项目建议
### 项目 1：LangGraph Agentic RAG 评测器
- **目标**：比较普通 RAG 和 Agentic RAG 在 citation accuracy、groundedness、retrieval precision 上的差异。
- **核心功能**：Query rewrite、retrieval grading、answer grounding check、引用准确率统计。
- **技术栈**：LangGraph、ChromaDB、Qwen/DeepSeek、FastAPI/React。
- **用到哪些论文思想**：Agentic RAG 框架、信息检索评测和 citation grounding。
- **两周 MVP**：准备 20 个开放问题，分别跑普通 RAG 和 Agentic RAG，输出对比报告。
- **简历亮点**：体现 Agentic RAG、评测、幻觉检测和工程闭环。

### 项目 2：长期记忆污染检测 RAG Agent
- **目标**：验证长期记忆中的错误论文结论是否会污染后续回答。
- **核心功能**：记忆写入、记忆冲突检测、遗忘/修正策略、回答对照。
- **技术栈**：LangGraph、向量库、JSON/SQLite memory、FastAPI。
- **用到哪些论文思想**：Agent memory、RAG grounding 和可靠性评估。
- **两周 MVP**：构造正确/错误记忆对照组，展示污染和修正过程。
- **简历亮点**：把长短期记忆、RAG 和可靠性评估结合成一个可演示系统。

### 项目 3：Citation Grounding Checker
- **目标**：检查答案中的 claim 是否真的被对应引用支撑。
- **核心功能**：claim 抽取、证据匹配、引用编号检查、风险标注。
- **技术栈**：LangGraph、Embedding 检索、LLM judge、React 报告页面。
- **用到哪些论文思想**：citation accuracy、groundedness 和 answer faithfulness。
- **两周 MVP**：输入一段带引用回答，输出 claim-证据-风险清单。
- **简历亮点**：直接体现 RAG 幻觉检测和引用可靠性工程。

## 8. 参考来源
参考来源由系统根据引用自动生成。"""


def _mock_general(prompt: str) -> str:
    query = _extract_query(prompt)
    if "Respond with ONLY a decimal number" in prompt:
        return "0.0"
    if "Relevance verdicts" in prompt:
        return _mock_grader(prompt)
    if "Classification:" in prompt and "query classifier" in prompt:
        return _mock_router(prompt)
    if "rewriter" in prompt.lower() or "Rewrite" in prompt:
        return query or "LLM Agent long-term memory retrieval"
    if "Using ONLY the provided paper abstracts" in prompt:
        return _mock_paper_search(prompt)
    if "PaperRadar" in prompt or "论文雷达" in prompt:
        return _mock_paper_radar(prompt)
    return "你好，我是 PaperRadar-Agent。你可以输入研究方向，我会生成中文论文雷达报告。"


def _invoke_mock(messages: list[BaseMessage]) -> BaseMessage:
    return AIMessage(content=_mock_general(_full_prompt(messages)))


def _should_fallback_to_mock(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in [
            "http 402",
            "insufficient balance",
            "insufficient_quota",
            "resource_exhausted",
            "quota",
            "billing",
            "access denied",
            "arrearage",
            "overdue-payment",
            "overdue payment",
            "account is in good standing",
            "timed out",
            "timeout",
            "api key",
            "not set",
        ]
    )


def _fallback_to_mock(messages: list[BaseMessage], exc: Exception) -> BaseMessage:
    logger.warning(
        "Configured LLM provider is unavailable (%s); falling back to mock response",
        exc,
    )
    return _invoke_mock(messages)


def _availability_key(cfg: ProviderConfig) -> str:
    return f"{cfg.provider}:{cfg.base_url}:{cfg.model}"


def _mark_unavailable(cfg: ProviderConfig, exc: Exception) -> None:
    if settings.llm_unavailable_cache_seconds <= 0:
        return
    key = _availability_key(cfg)
    _UNAVAILABLE_UNTIL[key] = time.monotonic() + settings.llm_unavailable_cache_seconds
    logger.warning(
        "Temporarily marking provider unavailable: provider=%s model=%s key_present=%s error=%s",
        cfg.provider,
        cfg.model,
        bool(cfg.api_key),
        _redact_secrets(str(exc))[:300],
    )


def _cached_unavailable_error(cfg: ProviderConfig) -> RuntimeError | None:
    until = _UNAVAILABLE_UNTIL.get(_availability_key(cfg))
    if until is None:
        return None
    if until <= time.monotonic():
        _UNAVAILABLE_UNTIL.pop(_availability_key(cfg), None)
        return None
    return RuntimeError(
        f"provider={cfg.provider} model={cfg.model} key_present={bool(cfg.api_key)} "
        "is temporarily unavailable"
    )


def _invoke_gemini_with_retry(
    messages: list[BaseMessage],
    model_override: str | None = None,
) -> BaseMessage:
    primary = model_override or resolve_provider_config("gemini").model
    model_sequence = [
        primary,
        settings.fallback_model,
        settings.tertiary_model,
        settings.tertiary_model,
    ]

    for attempt, model in enumerate(model_sequence):
        llm = get_llm(model)
        try:
            return llm.invoke(messages)
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait_time = 5
                if "retry in" in err_str.lower():
                    match = re.search(r"retry in (\d+\.?\d*)", err_str.lower())
                    if match:
                        wait_time = min(float(match.group(1)) + 1, 15)
                next_model = (
                    model_sequence[attempt + 1]
                    if attempt + 1 < len(model_sequence)
                    else "none"
                )
                logger.warning(
                    "Rate limited on %s (attempt %d/%d), waiting %.0fs; next: %s",
                    model,
                    attempt + 1,
                    len(model_sequence),
                    wait_time,
                    next_model,
                )
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError("All Gemini models rate limited or unavailable.")


def invoke_with_retry(
    messages: list[BaseMessage],
    model_override: str | None = None,
) -> BaseMessage:
    """Invoke the configured provider."""
    cfg = resolve_provider_config(model_override=model_override)

    if cfg.provider == "mock":
        return _invoke_mock(messages)

    if cfg.provider in {"deepseek", "qwen"}:
        if not cfg.api_key:
            return _fallback_to_mock(
                messages,
                RuntimeError(f"{cfg.key_env} is not set for provider={cfg.provider}."),
            )
        cached_error = _cached_unavailable_error(cfg)
        if cached_error is not None:
            return _fallback_to_mock(messages, cached_error)
        models = _model_sequence_for_provider(cfg.provider, cfg.model)
        try:
            return _invoke_openai_compatible_with_model_fallback(
                messages=messages,
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                models=models,
            )
        except RuntimeError as exc:
            if _should_fallback_to_mock(exc):
                _mark_unavailable(cfg, exc)
                return _fallback_to_mock(messages, exc)
            raise

    if cfg.provider == "gemini":
        if not cfg.api_key:
            return _fallback_to_mock(
                messages,
                RuntimeError(f"{cfg.key_env} is not set for provider=gemini."),
            )
        try:
            return _invoke_gemini_with_retry(messages, model_override=cfg.model)
        except RuntimeError as exc:
            if _should_fallback_to_mock(exc):
                return _fallback_to_mock(messages, exc)
            raise

    raise RuntimeError(f"Unsupported LLM_PROVIDER={cfg.provider!r}")
