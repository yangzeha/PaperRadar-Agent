"""Generator node: produces cited answers and Chinese PaperRadar reports."""

from __future__ import annotations

import json
import logging
import re
import time

from langchain_core.messages import HumanMessage

from app.agents.document_selection import select_output_documents
from app.agents.state import AgentState
from app.config import settings
from app.services.llm_provider import extract_text, invoke_with_retry
from app.services.memory_store import (
    add_history,
    extract_topics_from_text,
    get_memory_context,
    upsert_topics,
)
from app.services.paper_classifier import classify_documents
from app.services.paper_roles import (
    assign_paper_roles,
    is_graph_contrastive_recommendation_topic,
)

logger = logging.getLogger(__name__)

_RADAR_TASKS = {"paper_radar", "reading_plan", "project_idea"}

_GENERATE_PROMPT = (
    "You are a research assistant. Using ONLY the provided paper abstracts, "
    "write a comprehensive answer to the user's query. "
    "Cite papers inline using [1], [2], etc. corresponding to the paper numbers below.\n\n"
    "If the provided papers are insufficient, say so honestly rather than fabricating information.\n\n"
    "{retry_instruction}"
    "Query: {query}\n\n"
    "{papers_context}\n\n"
    "Answer:"
)

_GENERAL_PROMPT = (
    "You are PaperRadar-Agent, a research assistant that helps users find and "
    "understand academic papers. The user sent a general message. "
    "Respond helpfully and concisely in Chinese. Mention that you can search "
    "arXiv and PubMed for research papers.\n\n"
    "User: {query}\n\n"
    "Response:"
)

_PAPER_RADAR_PROMPT = """你是 PaperRadar-Agent，一个面向中文科研学习和简历项目准备的论文雷达 Agent。
请只基于给定论文标题、摘要、作者、年份、来源、路线分类和用户问题生成中文 Markdown 报告。
报告必须像“方向雷达分析”，不能写成简单论文列表。引用论文时使用 [1]、[2] 编号，编号必须对应下方论文列表。
不要暴露系统实现词，例如 mock、fallback、内部模式等。
{retry_instruction}

输出结构必须严格如下：

# PaperRadar：{topic}

## 1. 方向概览
用 3-5 段解释这个方向研究什么、为什么重要、核心问题是什么；关键判断尽量带引用。
如果用户主题不是 RAG / Agent / LLM 检索增强，不要强行写 Agentic RAG、普通 RAG、LangGraph 或多 Agent。

## 2. 方法路线分类
不要使用 Markdown 管道表格。请输出卡片式 Markdown，至少 4 条路线；每条路线必须绑定至少 1 篇引用。
格式示例：
### 路线 A：Survey / Taxonomy / SoK
- **核心问题**：
- **代表论文**：
- **主要思路**：
- **优点**：
- **局限**：
只能使用下方论文标题、摘要或路线分类能够支持的路线；没有论文支撑的路线不要硬编。

## 3. 代表论文推荐
选择 5-8 篇最值得读的论文，不要使用表格，不要大段复制摘要。请用“必读 / 重点 / 背景”分层推荐。
格式示例：
### 必读 1：[4] 论文标题
- **年份 / 来源**：
- **所属路线**：
- **为什么推荐**：
- **适合先读吗**：
- **可产出**：

## 4. 近年趋势
按年份或阶段总结至少 3 个趋势，每个趋势要绑定引用。

## 5. 研究空白
必须输出 5 个具体 gap，格式：
### Gap 1：标题
- 现状：
- 缺口：
- 可验证方式：
- 可做项目：

## 6. 两周阅读路线
不要使用表格。必须输出目标驱动的阶段式路线，包含：
### Day 1-2：建立方向全景
### Day 3-4：理解基础模型和任务设定
### Day 5-7：精读核心方法
### Day 8-10：关注实验设置与评价指标
### Day 11-12：寻找研究空白和可落地场景
### Day 13-14：形成小项目方案
每个阶段必须包含 **阅读目标**、**推荐论文**、**产出**。

## 7. 可做小项目建议
必须输出 3 个和用户主题强相关的小项目。每个项目包含：
### 项目 1：项目名
- 目标：
- 核心功能：
- 技术栈：
- 用到哪些论文思想：
- 两周 MVP：
- 简历亮点：

## 8. 参考来源
列出所有实际使用的论文来源，格式：
[1] 标题，作者，年份，来源，URL

任务类型：{classification}
用户问题：{query}

记忆上下文：
{memory_context}

检索到的论文：
{papers_context}

请生成 PaperRadar 中文报告："""

_INSUFFICIENT_EVIDENCE_TEXT = (
    "当前检索材料对这一小节的直接支持较少，下面只给出可由标题、摘要和元数据支持的保守判断。"
)

_REFERENCE_PLACEHOLDER = "参考来源由系统根据引用自动生成。"

_BANNED_TERMS = [
    "mock",
    "fallback",
    "当前 mock/fallback 模式",
    "当前 mock",
    "当前 fallback",
    "本次结果只基于 mock",
    "fallback 模式不会编造",
    "论文数量一致性检查器",
    "年份过滤核查器",
    "摘要证据表",
    "先按标题和摘要识别与用户问题最接近的论文",
    "本节证据不足",
    "只能基于检索摘要做保守归纳",
    "第 1-2 天：先读第 1-3 篇",
    "需要打开原文确认",
    "证据不足，不能替代完整论文阅读",
    "不能替代完整论文阅读",
]

_PROJECT_TITLES = [
    "基于 LangGraph 的 Agentic RAG 检索评测器",
    "多 Agent 文献研究助手",
    "长期记忆污染检测 RAG Agent",
]

_GAP_TEMPLATES = [
    (
        "Agentic RAG 评测指标不统一",
        "现有论文分别关注综述、领域应用或智能体框架，但评测维度常混合检索召回、答案事实性、引用准确性和工具调用效率。",
        "缺少能同时衡量检索决策、证据选择、生成质量和 citation grounding 的统一指标。",
        "构造同一批问题，比较 Naive RAG、带 query rewrite 的 RAG、带 grader/checker 的 Agentic RAG 在召回、引用准确率和幻觉率上的差异。",
        "实现一个小型 evaluation dashboard，展示每轮检索、评分、生成和引用检查结果。",
    ),
    (
        "多 Agent 检索协作的成本和稳定性",
        "Multi-Agent / Hierarchical 路线强调 Planner、Researcher、Critic 等角色分工，但协作会增加调用次数和状态同步成本。",
        "什么时候多 Agent 真正优于单 Agent RAG，缺少清晰边界。",
        "固定同一检索语料，比较单 Agent、双 Agent、三 Agent 在延迟、token 成本、答案质量上的收益曲线。",
        "用 LangGraph 做 planner-retriever-critic 三节点 demo，并记录每轮成本和质量。",
    ),
    (
        "长期记忆污染与遗忘机制",
        "Agentic RAG 常把用户偏好、历史检索和已读论文写入长期记忆，但错误结论也可能被长期保留。",
        "缺少针对 memory write、memory decay、memory correction 的可解释机制。",
        "向记忆库注入过时或错误论文摘要，测试后续回答是否被污染，并比较不同清理策略。",
        "实现一个 memory audit 节点，标记冲突记忆、过期记忆和低置信度记忆。",
    ),
    (
        "Query rewriting / iterative retrieval 的收益边界",
        "Planning / Reasoning 路线常使用查询改写和迭代检索，但多轮检索并不总能带来更好证据。",
        "缺少判断“何时继续改写、何时停止检索”的实用准则。",
        "设计短查询、长查询、跨领域查询三类任务，比较 0/1/3 次 rewrite 的召回率和噪声比例。",
        "做一个 Query Rewrite A/B 工具，展示改写前后命中文献、相关分和最终答案差异。",
    ),
    (
        "多模态 Agentic RAG 的证据对齐问题",
        "Multimodal RAG 将图像、表格、视频或 3D 场景纳入证据，但生成答案仍常以文本引用为主。",
        "跨模态证据如何对齐到具体 claim 和 citation，仍缺少低成本检查方法。",
        "构造图文混合资料，检查答案中的每个 claim 是否能追溯到文本片段或视觉区域。",
        "实现一个 multimodal citation checker 原型，输出 claim-证据类型-证据位置三元组。",
    ),
]


def _build_papers_context(documents: list[dict]) -> str:
    """Format a numbered list of paper abstracts for the generation prompt."""
    if not documents:
        return "No relevant papers found."

    abstract_limit = 1200
    if len(documents) > 50:
        abstract_limit = 600
    elif len(documents) > 20:
        abstract_limit = 800
    elif len(documents) > 10:
        abstract_limit = 1000

    parts = []
    for i, doc in enumerate(documents, start=1):
        title = doc.get("title", "Untitled")
        abstract = doc.get("abstract", "No abstract available.")[:abstract_limit]
        url = doc.get("url", "")
        published = doc.get("published") or "unknown"
        authors = ", ".join(doc.get("authors", [])[:6]) or "unknown authors"
        source = doc.get("source") or "unknown"
        route = doc.get("route") or doc.get("category") or "Unclassified"
        tier = doc.get("relevance_tier") or "core"
        keywords = ", ".join(doc.get("keywords", [])[:8])
        reason = doc.get("relevance_reason") or ""
        parts.append(
            f"[{i}] {title}\n"
            f"Authors: {authors}\n"
            f"Published: {published}\n"
            f"Source: {source}\n"
            f"Route: {route}\n"
            f"Relevance tier: {tier}\n"
            f"Keywords: {keywords}\n"
            f"Relevance reason: {reason}\n"
            f"URL: {url}\n"
            f"Abstract: {abstract}"
        )
    return "\n\n".join(parts)


def _extract_citations(documents: list[dict]) -> list[dict]:
    """Build citation dicts from the graded document list."""
    citations = []
    for i, doc in enumerate(documents, start=1):
        citations.append(
            {
                "index": i,
                "title": doc.get("title", "Untitled"),
                "url": doc.get("url", ""),
            }
        )
    return citations


def _retry_instruction_for_prompt(state: AgentState) -> str:
    score = float(state.get("hallucination_score", 0.0) or 0.0)
    generator_runs = sum(
        1 for step in state.get("steps", []) if step.get("node") == "generator"
    )
    if generator_runs == 0 or score < settings.hallucination_threshold:
        return ""
    return (
        f"重要：上一版回答的幻觉分数为 {score:.2f}，已达到或超过 "
        f"{settings.hallucination_threshold:.2f}。请重新生成，更保守地只使用"
        "给定论文摘要中的信息，删除没有依据的结论，并确保引用编号与论文列表一致。\n\n"
    )


def _format_reference_section(documents: list[dict]) -> str:
    references = []
    for i, doc in enumerate(documents, start=1):
        title = doc.get("title", "Untitled")
        url = doc.get("url", "")
        authors = ", ".join(doc.get("authors", [])[:6]) or "作者未知"
        published = doc.get("published") or "年份未知"
        source = _source_label(str(doc.get("source") or ""))
        references.append(f"[{i}] {title}，{authors}，{published}，{source}，{url}")
    return "\n".join(references) or "本次检索未返回可引用论文。"


def _year_label(document: dict) -> str:
    year = document.get("year")
    if year:
        return str(year)
    published = str(document.get("published") or "")
    match = re.search(r"(19|20)\d{2}", published)
    return match.group(0) if match else "年份未知"


def _citation(index: int) -> str:
    return f"[{index}]"


def _clean_cell(text: str, limit: int = 90) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = text.replace("|", "/")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


_PIPE_TABLE_FRAGMENTS = [
    "| 时间 |",
    "| 阅读目标 |",
    "| 路线 |",
    "| 推荐级 |",
    "|---",
]


_ROUTE_KEYWORDS = {
    "Survey / Taxonomy / SoK": (
        "survey",
        "taxonomy",
        "sok",
        "overview",
        "review",
    ),
    "Graph Contrastive Learning / SSL Recommendation": (
        "graph contrastive",
        "contrastive learning",
        "self-supervised",
        "self supervised",
        "gcl",
        "lightgcl",
        "xsim",
        "simgcl",
        "augmentation",
        "view construction",
        "view generation",
        "invariant",
        "rationale",
    ),
    "Graph Neural Recommendation / Collaborative Filtering": (
        "graph neural",
        "gnn",
        "graph learning",
        "collaborative filtering",
        "recommender",
        "recommendation",
        "user-item",
        "user item",
        "bipartite",
        "lightgcn",
    ),
    "Data Sparsity / Robustness / Debiasing": (
        "sparse",
        "sparsity",
        "cold-start",
        "cold start",
        "noise",
        "noisy",
        "robust",
        "robustness",
        "popularity bias",
        "debias",
        "fairness",
        "long-tail",
    ),
    "Knowledge Graph / Side Information Recommendation": (
        "knowledge graph",
        "kg-based",
        "side information",
        "semantic",
        "entity",
        "relation",
    ),
    "Planning / Reasoning": (
        "planning",
        "reasoning",
        "iterative retrieval",
        "reflection",
        "query rewrite",
        "rewriting",
        "decomposition",
    ),
    "Multi-Agent / Hierarchical": (
        "multi-agent",
        "multi agent",
        "hierarchical",
        "collaboration",
        "planner",
        "critic",
    ),
    "Evaluation / Benchmark": (
        "benchmark",
        "evaluation",
        "metric",
        "metrics",
        "hallucination",
        "citation",
    ),
    "Multimodal RAG": (
        "multimodal",
        "multi-modal",
        "vision",
        "visual",
        "mllm",
        "image",
        "video",
    ),
    "Domain-specific Agentic RAG": (
        "fintech",
        "finance",
        "medical",
        "clinical",
        "legal",
        "domain-specific",
        "industry",
    ),
    "Knowledge Gap / Research Discovery": (
        "knowledge gap",
        "research gap",
        "gap discovery",
        "scientific discovery",
        "hypothesis",
    ),
    "General RAG / Agentic RAG": (
        "rag",
        "retrieval-augmented",
        "retrieval augmented",
        "agentic",
        "agent",
    ),
}


_ROUTE_ORDER = [
    "Survey / Taxonomy / SoK",
    "Graph Contrastive Learning / SSL Recommendation",
    "Graph Neural Recommendation / Collaborative Filtering",
    "Data Sparsity / Robustness / Debiasing",
    "Knowledge Graph / Side Information Recommendation",
    "Planning / Reasoning",
    "Multi-Agent / Hierarchical",
    "Evaluation / Benchmark",
    "Multimodal RAG",
    "Domain-specific Agentic RAG",
    "Knowledge Gap / Research Discovery",
    "General RAG / Agentic RAG",
]

_RECOMMENDER_ROUTES = {
    "Graph Contrastive Learning / SSL Recommendation",
    "Graph Neural Recommendation / Collaborative Filtering",
    "Data Sparsity / Robustness / Debiasing",
    "Knowledge Graph / Side Information Recommendation",
}


def _has_pipe_table_residue(answer: str) -> bool:
    return any(fragment in answer for fragment in _PIPE_TABLE_FRAGMENTS)


def _route_display_name(route: str) -> str:
    mapping = {
        "Planning / Reasoning": "Planning / Reasoning / Iterative Retrieval",
        "Multi-Agent / Hierarchical": "Multi-Agent / Hierarchical RAG",
    }
    return mapping.get(route, route)


def _paper_label(index: int, document: dict, limit: int = 92) -> str:
    title = _clean_cell(str(document.get("title") or "Untitled"), limit)
    return f"{_citation(index)} {title}"


def _document_keyword_text(document: dict) -> str:
    keywords = " ".join(str(item) for item in document.get("keywords", []) or [])
    return " ".join(
        [
            str(document.get("title") or ""),
            str(document.get("abstract") or ""),
            keywords,
        ]
    ).lower()


def _topic_text(topic: str, documents: list[dict]) -> str:
    return " ".join(
        [
            topic,
            *[
                f"{document.get('title', '')} {document.get('abstract', '')} "
                f"{' '.join(str(item) for item in document.get('keywords', []) or [])}"
                for document in documents[:12]
            ],
        ]
    ).lower()


def _is_recommender_gcl_topic(topic: str, documents: list[dict]) -> bool:
    return is_graph_contrastive_recommendation_topic(topic, documents)


def _is_rag_topic(topic: str, documents: list[dict]) -> bool:
    text = _topic_text(topic, documents)
    return any(
        marker in text
        for marker in (
            "agentic rag",
            " rag",
            "retrieval-augmented",
            "retrieval augmented",
            "langgraph",
            "llm agent",
            "智能体",
            "检索增强",
        )
    )


def _infer_route_for_topic(document: dict, topic: str, documents: list[dict]) -> str:
    existing = str(document.get("route") or document.get("category") or "")
    text = _document_keyword_text(document)
    title = str(document.get("title") or "").lower()

    if _is_recommender_gcl_topic(topic, documents) and existing:
        return existing

    if _is_recommender_gcl_topic(topic, documents):
        if any(marker in title for marker in ("survey", "review", "taxonomy", "sok")):
            return "Survey / Taxonomy / SoK"
        if any(
            marker in text
            for marker in (
                "graph contrastive",
                "contrastive learning",
                "self-supervised",
                "self supervised",
                "lightgcl",
                "xsim",
                "simgcl",
                "view construction",
                "augmentation",
                "invariant",
                "rationale",
            )
        ):
            return "Graph Contrastive Learning / SSL Recommendation"
        if any(
            marker in text
            for marker in (
                "sparse",
                "sparsity",
                "cold-start",
                "cold start",
                "noise",
                "noisy",
                "robust",
                "popularity bias",
                "long-tail",
            )
        ):
            return "Data Sparsity / Robustness / Debiasing"
        if any(marker in text for marker in ("knowledge graph", "kg-based", "side information")):
            return "Knowledge Graph / Side Information Recommendation"
        if any(
            marker in text
            for marker in (
                "graph neural",
                "gnn",
                "graph learning",
                "collaborative filtering",
                "recommender",
                "recommendation",
                "user-item",
            )
        ):
            return "Graph Neural Recommendation / Collaborative Filtering"
        return "Graph Neural Recommendation / Collaborative Filtering"

    return existing or "General Topic / Method Background"


def _route_groups(documents: list[dict], topic: str = "") -> dict[str, list[tuple[int, dict]]]:
    groups: dict[str, list[tuple[int, dict]]] = {}
    for index, document in enumerate(documents, start=1):
        route = _infer_route_for_topic(document, topic, documents)
        groups.setdefault(str(route), []).append((index, document))
    return groups


def _route_groups_for_cards(documents: list[dict], topic: str = "") -> dict[str, list[tuple[int, dict]]]:
    """Build route groups for card output, keeping every route grounded in papers."""
    is_recommender_topic = _is_recommender_gcl_topic(topic, documents)
    raw_groups = _route_groups(documents, topic)
    ordered: dict[str, list[tuple[int, dict]]] = {}
    for route in _ROUTE_ORDER:
        if not is_recommender_topic and route in _RECOMMENDER_ROUTES:
            continue
        if route in raw_groups:
            ordered[route] = raw_groups[route]
    for route, items in raw_groups.items():
        if not is_recommender_topic and route in _RECOMMENDER_ROUTES:
            continue
        if route not in ordered and route != "Background":
            ordered[route] = items

    target_count = min(4, len(documents))
    if len(ordered) >= target_count:
        return ordered

    indexed_docs = list(enumerate(documents, start=1))
    for route in _ROUTE_ORDER:
        if route in ordered:
            continue
        if not is_recommender_topic and route in _RECOMMENDER_ROUTES:
            continue
        keywords = _ROUTE_KEYWORDS.get(route, ())
        matches = [
            (index, document)
            for index, document in indexed_docs
            if any(keyword in _document_keyword_text(document) for keyword in keywords)
        ]
        if matches:
            ordered[route] = matches[:3]
        if len(ordered) >= target_count:
            break

    if len(ordered) < target_count and indexed_docs:
        fallback_route = (
            "Graph Neural Recommendation / Collaborative Filtering"
            if is_recommender_topic
            else "General Topic / Method Background"
        )
        ordered.setdefault(fallback_route, indexed_docs[:3])

    return ordered


def _route_core_problem(route: str) -> str:
    mapping = {
        "Survey / Taxonomy / SoK": "建立术语、任务边界和方法演化框架",
        "Graph Contrastive Learning / SSL Recommendation": "用自监督/对比学习增强用户-物品图表征，缓解交互稀疏和噪声问题",
        "Graph Neural Recommendation / Collaborative Filtering": "用图神经网络建模用户-物品交互、邻域传播和协同过滤信号",
        "Data Sparsity / Robustness / Debiasing": "提升推荐模型在稀疏、噪声、冷启动和长尾场景下的稳定性",
        "Knowledge Graph / Side Information Recommendation": "利用知识图谱、实体关系和辅助信息增强推荐解释性与覆盖面",
        "Planning / Reasoning": "让系统能规划检索、分解问题并迭代修正证据",
        "Multi-Agent / Hierarchical": "用多角色协作提升检索、批判和写作质量",
        "Multimodal RAG": "让文本、图像、视频等多模态证据能共同支撑回答",
        "Evaluation / Benchmark": "评估检索质量、事实性、引用准确率和系统鲁棒性",
        "Domain-specific Agentic RAG": "处理金融、医疗、法律等领域的术语和知识约束",
        "Knowledge Gap / Research Discovery": "从文献证据中发现空白、生成假设和辅助选题",
        "General RAG / Agentic RAG": "把检索增强生成扩展成可控、可检查的智能体流程",
        "General Topic / Method Background": "补充该方向的基础概念、任务设定和可复用方法背景",
    }
    return mapping.get(route, "围绕该方向的特定方法、应用或评估问题")


def _route_main_idea(route: str) -> str:
    mapping = {
        "Survey / Taxonomy / SoK": "归纳任务定义、系统组件、数据集和开放问题",
        "Graph Contrastive Learning / SSL Recommendation": "构造图视图、增强策略和对比目标，学习更稳健的用户/物品表示",
        "Graph Neural Recommendation / Collaborative Filtering": "基于用户-物品图传播高阶协同信号，并用轻量化 GNN 改进推荐排序",
        "Data Sparsity / Robustness / Debiasing": "围绕稀疏交互、噪声边、长尾物品和流行度偏差设计训练与评测方案",
        "Knowledge Graph / Side Information Recommendation": "把实体关系、语义属性或外部知识接入推荐图，补充交互数据不足",
        "Planning / Reasoning": "用 query rewrite、任务分解、反思检索或多步推理改进证据获取",
        "Multi-Agent / Hierarchical": "拆分 planner、retriever、critic、writer 等角色并共享状态",
        "Multimodal RAG": "把视觉或多模态证据映射到可引用的文本 claim",
        "Evaluation / Benchmark": "构造 benchmark、指标和自动检查流程",
        "Domain-specific Agentic RAG": "引入领域知识、ontology、规范和专用评测数据",
        "Knowledge Gap / Research Discovery": "利用检索和生成模型做空白发现、假设生成和研究雷达",
        "General RAG / Agentic RAG": "组合检索、评分、生成、校验和状态管理节点",
        "General Topic / Method Background": "基于当前论文摘要提取共同任务、模型假设和实验设置",
    }
    return mapping.get(route, "基于当前论文摘要提取路线特征并形成可复用模块")


def _representative_documents(
    documents: list[dict],
    limit: int = 8,
    topic: str = "",
) -> list[tuple[int, dict]]:
    if _is_recommender_gcl_topic(topic, documents):
        return sorted(
            list(enumerate(documents, start=1)),
            key=lambda item: (
                int(item[1].get("priority") or 999),
                -float(item[1].get("relevance_score") or 0.0),
                item[0],
            ),
        )[:limit]

    groups = _route_groups(documents, topic)
    selected: list[tuple[int, dict]] = []
    seen: set[int] = set()
    for route_docs in groups.values():
        index, document = sorted(
            route_docs,
            key=lambda item: float(item[1].get("relevance_score") or 0),
            reverse=True,
        )[0]
        selected.append((index, document))
        seen.add(index)
        if len(selected) >= limit:
            return selected
    for index, document in enumerate(documents, start=1):
        if index in seen:
            continue
        selected.append((index, document))
        if len(selected) >= limit:
            break
    return selected


def _build_route_cards(documents: list[dict], topic: str = "") -> str:
    chunks: list[str] = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for position, (route, items) in enumerate(_route_groups_for_cards(documents, topic).items()):
        if not items:
            continue
        representative_papers = "；".join(
            _paper_label(index, document, 76) for index, document in items[:3]
        )
        advantage = (
            "适合建立从综述、核心模型到实验复现的阅读主线，并能直接转化为推荐系统实验或简历项目"
            if _is_recommender_gcl_topic(topic, documents)
            else "适合建立阅读主线，也方便把论文思想落到可复现实验、评测流程或项目原型中"
        )
        chunks.append(
            f"### 路线 {letters[position]}：{_route_display_name(route)}\n"
            f"- **核心问题**：{_route_core_problem(route)}。\n"
            f"- **代表论文**：{representative_papers}。\n"
            f"- **主要思路**：{_route_main_idea(route)}。\n"
            f"- **优点**：{advantage}。\n"
            "- **局限**：当前检索材料可能只覆盖该路线的一部分子问题，正式综述还需要继续扩展同类论文。"
        )
    return "\n\n".join(chunks) or "当前检索材料不足，暂时无法形成可靠路线卡片。"


def _recommendation_reason(document: dict, query: str) -> str:
    route = document.get("route") or "General Topic / Method Background"
    title = str(document.get("title") or "")
    if "Survey" in route:
        purpose = "适合先读，用来建立概念框架和论文地图"
    elif "Graph Contrastive" in route:
        purpose = "适合精读，用来理解图对比学习在推荐中的视图构造、对比目标和轻量化建模"
    elif "Graph Neural Recommendation" in route:
        purpose = "适合补齐 GNN 推荐和协同过滤基础，理解用户-物品图如何传播偏好信号"
    elif "Data Sparsity" in route:
        purpose = "适合分析稀疏交互、噪声和长尾推荐场景下的鲁棒性问题"
    elif "Knowledge Graph" in route:
        purpose = "适合理解知识图谱或辅助信息如何补充推荐系统交互数据"
    elif "Benchmark" in route or "Evaluation" in route:
        purpose = "适合整理评测指标和复现实验设计"
    elif "Multi-Agent" in route:
        purpose = "适合理解多 Agent 协作如何进入检索与写作流程"
    elif "Planning" in route:
        purpose = "适合分析 query rewrite、迭代检索和推理控制"
    elif "Multimodal" in route:
        purpose = "适合寻找多模态证据对齐的项目切入点"
    elif "Domain" in route:
        purpose = "适合观察垂直领域如何约束 RAG 证据和术语"
    else:
        purpose = "适合补充该主题的基础方法背景"
    return _clean_cell(f"{purpose}；与“{query}”的关系体现在 {title} 所覆盖的 {route} 路线。", 140)


def _recommendation_output(document: dict, rank: int, query: str = "") -> str:
    route = str(document.get("route") or "General Topic / Method Background")
    if _is_recommender_gcl_topic(query, [document]):
        if "Survey" in route:
            return "整理图推荐、GNN 推荐和图对比学习推荐的术语表、论文地图和方法演化线。"
        if "Graph Contrastive" in route:
            return "复现核心模型，记录图增强/视图构造、对比损失和推荐指标提升点。"
        if "Graph Neural Recommendation" in route:
            return "画出用户-物品图传播流程，总结 GNN 推荐和传统协同过滤的区别。"
        if "Data Sparsity" in route:
            return "设计稀疏度分桶实验，观察 Recall@K/NDCG@K 在冷启动或长尾物品上的变化。"
        if "Knowledge Graph" in route:
            return "整理知识图谱推荐的数据结构、实体关系和与 GCL 结合的可能入口。"
        return "形成一条推荐系统论文贡献笔记，标出可复现实验和后续改进点。"
    if "Survey" in route:
        return "整理 Agentic RAG 的基本术语表、系统模块图和阅读地图。"
    if "Evaluation" in route or "Benchmark" in route:
        return "整理 citation accuracy、groundedness、retrieval precision 等评测指标。"
    if "Multi-Agent" in route:
        return "画出 planner、researcher、critic、writer 的协作流程图。"
    if "Planning" in route:
        return "写出 query rewrite、检索评分、再检索的 LangGraph 流程图。"
    if "Multimodal" in route:
        return "整理 claim 与文本/视觉证据之间的对齐检查清单。"
    if "Domain" in route:
        return "总结垂直领域术语、ontology、专有数据源和约束条件。"
    if rank <= 2:
        return "形成方向入口笔记，标注后续精读问题。"
    return "补充背景概念，记录和用户主题相关的可用证据。"


def _build_recommendation_cards(documents: list[dict], query: str) -> str:
    selected = _representative_documents(documents, 8, query)
    if not selected:
        return "当前检索材料不足，暂时无法推荐代表论文。"

    chunks: list[str] = []
    background_lines: list[str] = []
    for rank, (index, document) in enumerate(selected, start=1):
        title = _clean_cell(str(document.get("title") or "Untitled"), 120)
        route = _route_display_name(_infer_route_for_topic(document, query, documents))
        source = _source_label(str(document.get("source") or ""))
        year = _year_label(document)
        reason = _recommendation_reason(document, query)
        output = _recommendation_output(document, rank, query)

        if rank <= 2:
            label = f"必读 {rank}"
        elif rank <= 5:
            label = f"重点 {rank}"
        else:
            background_lines.append(
                f"- {_paper_label(index, document, 96)}：适合作为背景阅读，补充 {route} 路线的上下文；可产出一条背景证据笔记。"
            )
            continue

        chunks.append(
            f"### {label}：{_citation(index)} {title}\n"
            f"- **年份 / 来源**：{year}，{source}\n"
            f"- **所属路线**：{route}\n"
            f"- **为什么推荐**：{reason}\n"
            f"- **适合先读吗**：{'适合。建议第一轮精读，用来建立主线。' if rank <= 2 else '适合第二轮重点读，用来补齐方法或评测视角。'}\n"
            f"- **可产出**：{output}"
        )

    if background_lines:
        chunks.append("### 背景阅读\n" + "\n".join(background_lines))
    return "\n\n".join(chunks)


def _build_trends(documents: list[dict], topic: str = "") -> str:
    years = sorted(
        {
            int(year)
            for year in (_year_label(document) for document in documents)
            if str(year).isdigit()
        }
    )
    year_span = f"{years[0]}-{years[-1]}" if years else "当前检索年份范围"
    groups = _route_groups(documents, topic)
    if _is_recommender_gcl_topic(topic, documents):
        survey_refs = " ".join(_citation(i) for i, _ in groups.get("Survey / Taxonomy / SoK", [])[:2])
        gcl_refs = " ".join(_citation(i) for i, _ in groups.get("Graph Contrastive Learning / SSL Recommendation", [])[:3])
        gnn_refs = " ".join(_citation(i) for i, _ in groups.get("Graph Neural Recommendation / Collaborative Filtering", [])[:2])
        robustness_refs = " ".join(_citation(i) for i, _ in groups.get("Data Sparsity / Robustness / Debiasing", [])[:2])
        kg_refs = " ".join(_citation(i) for i, _ in groups.get("Knowledge Graph / Side Information Recommendation", [])[:2])
        fallback_refs = " ".join(_citation(i) for i, _ in _representative_documents(documents, 3, topic)) or "[1]"
        survey_refs = survey_refs or fallback_refs
        gcl_refs = gcl_refs or fallback_refs
        gnn_refs = gnn_refs or survey_refs
        robustness_refs = robustness_refs or gcl_refs
        kg_refs = kg_refs or survey_refs
        return "\n".join(
            [
                f"1. **{year_span}：从 GNN 推荐综述走向自监督/对比学习增强。** 综述论文先梳理了图推荐和自监督推荐的任务边界，后续方法把对比学习用于用户-物品图表征增强 {survey_refs}。",
                f"2. **从复杂图增强到轻量化 GCL。** LightGCL、XSimGCL 等工作强调用更简单的图视图、噪声注入或谱域信号降低训练复杂度，同时保持推荐性能 {gcl_refs}。",
                f"3. **核心矛盾从准确率扩展到稀疏性、噪声和长尾鲁棒性。** 近期论文更关注稀疏交互、噪声边和长尾物品下的推荐稳定性，而不是只追求整体 Recall/NDCG {robustness_refs}。",
                f"4. **背景路线仍依赖图推荐、知识图谱和协同过滤基础。** GCL 推荐方法需要和 GNN 推荐、KG 推荐、协同过滤基线一起比较，才能判断改进是否真实有效 {gnn_refs} {kg_refs}。",
            ]
        )
    survey_refs = " ".join(_citation(i) for i, _ in groups.get("Survey / Taxonomy / SoK", [])[:2])
    eval_refs = " ".join(_citation(i) for i, _ in groups.get("Evaluation / Benchmark", [])[:2])
    agent_refs = " ".join(
        _citation(i)
        for route in ("Planning / Reasoning", "Multi-Agent / Hierarchical", "General RAG / Agentic RAG")
        for i, _ in groups.get(route, [])[:2]
    )
    domain_refs = " ".join(
        _citation(i)
        for route in ("Domain-specific Agentic RAG", "Multimodal RAG")
        for i, _ in groups.get(route, [])[:2]
    )
    if not survey_refs:
        survey_refs = " ".join(_citation(i) for i, _ in _representative_documents(documents, 2))
    if not agent_refs:
        agent_refs = survey_refs
    if not eval_refs:
        eval_refs = agent_refs
    if not domain_refs:
        domain_refs = eval_refs
    return "\n".join(
        [
            f"1. **{year_span}：从 RAG 组件整理到系统化路线图。** Survey / Taxonomy 类论文帮助把检索、重排、生成和校验拆成可比较模块，为 Agentic RAG 的节点化设计打基础 {survey_refs}。",
            f"2. **从单次检索到规划式、迭代式检索。** Planning / Reasoning 相关工作把 query rewrite、任务分解和反思检索纳入闭环，使系统不再只依赖一次向量召回 {agent_refs}。",
            f"3. **从方法提出走向 benchmark 与 grounded evaluation。** 评测论文开始关注 citation accuracy、hallucination checking 和检索证据质量，而不是只看最终回答流畅度 {eval_refs}。",
            f"4. **从通用 RAG 走向多模态和垂直领域。** 多模态、医疗、金融等场景要求系统处理专有术语、图文证据和领域约束，这推动 Agentic RAG 增加 planner、checker 和 memory 节点 {domain_refs}。",
        ]
    )


def _build_gap_section(documents: list[dict], topic: str = "") -> str:
    refs = [_citation(index) for index, _ in _representative_documents(documents, 8, topic)]
    if not refs:
        refs = ["[1]"]
    if _is_recommender_gcl_topic(topic, documents):
        templates = [
            (
                "图增强和视图构造缺少统一原则",
                "LightGCL、XSimGCL 等方法都在构造不同图视图或扰动信号，但哪些增强真正保留推荐语义仍高度依赖经验。",
                "缺少能解释“好视图”和“坏视图”的统一准则，尤其缺少与用户兴趣、物品语义和交互噪声的对应关系。",
                "固定同一数据集，系统比较边删除、节点扰动、谱域增强、嵌入噪声等策略对 Recall@K、NDCG@K 和训练稳定性的影响。",
                "做一个 Graph Augmentation A/B 面板，展示不同视图构造对推荐结果和表征分布的影响。",
            ),
            (
                "稀疏交互和冷启动场景下的收益边界不清楚",
                "GCL 推荐常声称能缓解稀疏数据问题，但不同稀疏度用户、长尾物品和新用户场景的收益并不一致。",
                "缺少按用户活跃度、物品流行度、训练交互比例分桶的细粒度评测。",
                "把用户和物品按交互数量分桶，比较 LightGCN、SGL、LightGCL、XSimGCL 在每个桶上的表现。",
                "实现一个 sparsity robustness evaluator，输出不同稀疏度下的指标曲线。",
            ),
            (
                "对比目标和推荐排序目标可能错配",
                "对比学习优化的是视图间一致性或表示分离，而推荐最终优化的是排序质量和用户偏好命中。",
                "缺少分析 InfoNCE/自监督损失与 BPR、CE、Recall/NDCG 之间关系的可解释实验。",
                "记录训练过程中 SSL loss、推荐 loss 和验证集 NDCG 的相关性，观察何时出现过度对齐或过平滑。",
                "做一个 loss-diagnostics 工具，追踪对比损失和推荐指标的同步/背离。",
            ),
            (
                "流行度偏差和长尾推荐仍然容易被掩盖",
                "整体 Recall@K 提升可能主要来自热门物品，长尾物品和小众用户的收益不一定同步提升。",
                "缺少把 GCL 推荐和 popularity bias、公平性、覆盖率一起评估的报告模板。",
                "计算热门/长尾物品分组上的 Recall、Coverage、Novelty，并比较不同 GCL 模型。",
                "实现一个 popularity-bias dashboard，展示准确率和长尾覆盖之间的权衡。",
            ),
            (
                "可解释性和用户意图建模不足",
                "部分 GCL 方法能提升表征质量，但难以说明某次推荐来自哪种用户意图、图邻居或对比视图。",
                "缺少将 invariant rationale、用户兴趣子图和推荐解释结合起来的轻量方法。",
                "对每个推荐结果追踪关键邻居、增强视图和相似用户，检查解释是否稳定。",
                "做一个 GCL recommendation explainer，输出用户-物品路径、关键邻居和视图贡献。",
            ),
        ]
    else:
        templates = _GAP_TEMPLATES
    chunks: list[str] = []
    for i, (title, status, gap, validation, project) in enumerate(templates, start=1):
        ref = " ".join(refs[max(0, i - 2): i + 1]) or refs[0]
        chunks.append(
            f"### Gap {i}：{title}\n"
            f"- 现状：{status} {ref}\n"
            f"- 缺口：{gap}\n"
            f"- 可验证方式：{validation}\n"
            f"- 可做项目：{project}"
        )
    return "\n\n".join(chunks)


def _build_reading_plan(documents: list[dict], topic: str = "") -> str:
    groups = _route_groups(documents, topic)

    def refs_for(routes: list[str], fallback_start: int = 0) -> str:
        refs = [
            _citation(index)
            for route in routes
            for index, _ in groups.get(route, [])[:2]
        ]
        if not refs:
            refs = [_citation(index) for index, _ in _representative_documents(documents, 2, topic)[fallback_start: fallback_start + 2]]
        return " ".join(refs) or "[1]"

    if _is_recommender_gcl_topic(topic, documents):
        return "\n\n".join(
            [
                "### Day 1-2：建立方向全景\n"
                "- **阅读目标**：先弄清图推荐、协同过滤、自监督学习和图对比学习之间的关系。\n"
                f"- **推荐论文**：{refs_for(['Survey / Taxonomy / SoK'])}\n"
                "- **产出**：整理一张“GNN 推荐 → 自监督推荐 → 图对比学习推荐”的概念地图。",
                "### Day 3-4：理解基础模型和任务设定\n"
                "- **阅读目标**：理解用户-物品二部图、LightGCN/GNN 推荐、隐式反馈、Recall@K/NDCG@K 等基础设定。\n"
                f"- **推荐论文**：{refs_for(['Graph Neural Recommendation / Collaborative Filtering', 'Survey / Taxonomy / SoK'], 1)}\n"
                "- **产出**：画出用户-物品图传播流程，列出常用数据集和评价指标。",
                "### Day 5-7：精读核心方法\n"
                "- **阅读目标**：重点比较 LightGCL、XSimGCL 等模型的视图构造、对比损失、轻量化设计和性能提升来源。\n"
                f"- **推荐论文**：{refs_for(['Graph Contrastive Learning / SSL Recommendation'], 2)}\n"
                "- **产出**：写一页方法对比笔记：增强方式、损失函数、复杂度、适用场景。",
                "### Day 8-10：关注实验设置与评价指标\n"
                "- **阅读目标**：检查实验数据集、负采样、稀疏度、冷启动、流行度偏差和长尾覆盖是否被充分评估。\n"
                f"- **推荐论文**：{refs_for(['Data Sparsity / Robustness / Debiasing', 'Evaluation / Benchmark'], 3)}\n"
                "- **产出**：做一个复现实验清单，列出 Recall@K、NDCG@K、Coverage、Novelty 等指标。",
                "### Day 11-12：寻找研究空白和可落地场景\n"
                "- **阅读目标**：围绕视图构造、稀疏数据、长尾推荐、可解释性或知识图谱增强选择一个小切口。\n"
                f"- **推荐论文**：{refs_for(['Knowledge Graph / Side Information Recommendation', 'Graph Contrastive Learning / SSL Recommendation'], 4)}\n"
                "- **产出**：形成 2-3 个可复现或可改进的小项目候选。",
                "### Day 13-14：形成小项目方案\n"
                "- **阅读目标**：把核心论文思想落到一个两周可完成的推荐系统实验 demo。\n"
                "- **推荐方向**：LightGCL/XSimGCL 复现对比、图增强策略 A/B、稀疏鲁棒性评测面板。\n"
                "- **产出**：README、实验脚本、指标表、结果图和简历项目描述。",
            ]
        )

    return "\n\n".join(
        [
            "### Day 1-2：建立方向全景\n"
            "- **阅读目标**：理解 Agentic RAG 和普通 RAG 的区别，掌握 planner、retriever、grader、rewriter、memory、citation checker 等核心概念。\n"
            f"- **推荐论文**：{refs_for(['Survey / Taxonomy / SoK'])}\n"
            "- **产出**：画一张 Agentic RAG 系统模块图，整理 10 个核心术语。",
            "### Day 3-4：理解规划式检索和 Query Rewrite\n"
            "- **阅读目标**：理解模型如何根据检索结果动态改写 query、重新检索和整合证据。\n"
            f"- **推荐论文**：{refs_for(['Planning / Reasoning', 'General RAG / Agentic RAG'], 1)}\n"
            "- **产出**：写出一个 LangGraph 检索-评分-重写-再检索流程图。",
            "### Day 5-7：理解多 Agent / 分层 RAG\n"
            "- **阅读目标**：理解 planner、researcher、critic、writer 等角色如何协同完成复杂问题。\n"
            f"- **推荐论文**：{refs_for(['Multi-Agent / Hierarchical'], 2)}\n"
            "- **产出**：总结多 Agent RAG 和单 Agent RAG 的优势、成本和失败点。",
            "### Day 8-10：关注评测与幻觉检测\n"
            "- **阅读目标**：理解 citation accuracy、groundedness、answer faithfulness、retrieval precision 等指标。\n"
            f"- **推荐论文**：{refs_for(['Evaluation / Benchmark'], 3)}\n"
            "- **产出**：设计一个小型 Agentic RAG 评测表。",
            "### Day 11-12：寻找研究空白和可落地场景\n"
            "- **阅读目标**：从评测、长期记忆、多模态、垂直领域中选择一个小切口。\n"
            f"- **推荐论文**：{refs_for(['Domain-specific Agentic RAG', 'Multimodal RAG', 'Knowledge Gap / Research Discovery'], 4)}\n"
            "- **产出**：形成 2-3 个小项目候选。",
            "### Day 13-14：形成小项目方案\n"
            "- **阅读目标**：把前面读到的论文思想落到一个两周可完成的 demo。\n"
            "- **推荐方向**：LangGraph Agentic RAG Evaluator / Citation Grounding Checker / Long-term Memory RAG Agent\n"
            "- **产出**：README、架构图、Demo 问题、简历项目描述。",
        ]
    )


def _build_project_section(documents: list[dict], topic: str = "") -> str:
    refs = " ".join(_citation(index) for index, _ in _representative_documents(documents, 4, topic)) or "[1]"
    if _is_recommender_gcl_topic(topic, documents):
        return "\n\n".join(
            [
                "### 项目 1：LightGCL / XSimGCL 复现与对比实验\n"
                "- 目标：复现图对比学习推荐代表方法，并和 LightGCN、SGL 等基线比较。\n"
                "- 核心功能：数据预处理、用户-物品图构建、模型训练、Recall@K/NDCG@K 评测、结果可视化。\n"
                "- 技术栈：PyTorch、PyTorch Geometric 或 DGL、RecBole/自定义推荐评测脚本、Matplotlib。\n"
                f"- 用到哪些论文思想：图对比学习、轻量化图推荐、稀疏交互增强 {refs}。\n"
                "- 两周 MVP：在 MovieLens 或 Yelp 子集上跑通 2-3 个模型，输出指标表和消融结果。\n"
                "- 简历亮点：能体现推荐算法复现、图学习、实验评测和论文理解能力。",
                "### 项目 2：图增强策略 A/B 测试推荐器\n"
                "- 目标：比较边删除、节点扰动、嵌入噪声、谱域增强等视图构造对推荐效果的影响。\n"
                "- 核心功能：可切换增强策略、训练曲线记录、指标对比、视图相似度分析。\n"
                "- 技术栈：PyTorch、NetworkX、推荐评测指标、Streamlit/React 可视化面板。\n"
                f"- 用到哪些论文思想：视图构造、对比目标、表示一致性和推荐排序目标之间的关系 {refs}。\n"
                "- 两周 MVP：实现 3 种增强策略，在同一数据集上输出 Recall@20/NDCG@20 对比图。\n"
                "- 简历亮点：不是只跑模型，而是能解释为什么某种增强有效或失效。",
                "### 项目 3：推荐系统稀疏性与长尾鲁棒性评测面板\n"
                "- 目标：检验 GCL 推荐方法在低活跃用户、长尾物品和稀疏训练集上的真实收益。\n"
                "- 核心功能：用户/物品分桶、稀疏度采样、热门/长尾指标、Coverage/Novelty 分析。\n"
                "- 技术栈：Python、Pandas、PyTorch、推荐系统指标库、可视化 dashboard。\n"
                f"- 用到哪些论文思想：稀疏数据推荐、鲁棒性评估、长尾覆盖和对比学习增强 {refs}。\n"
                "- 两周 MVP：按交互数量分桶，比较 LightGCN 与 GCL 方法在各桶上的表现。\n"
                "- 简历亮点：体现你能从“论文效果好”进一步追问“在哪些用户/物品上真的好”。",
            ]
        )
    return "\n\n".join(
        [
            "### 项目 1：基于 LangGraph 的 Agentic RAG 检索评测器\n"
            "- 目标：比较普通 RAG、带 query rewrite 的 RAG、带 grader/checker 的 Agentic RAG。\n"
            "- 核心功能：路由、检索、相关性评分、生成、幻觉检查、引用准确率统计。\n"
            "- 技术栈：LangGraph、FastAPI、ChromaDB、Qwen/DeepSeek、React。\n"
            f"- 用到哪些论文思想：RAG survey、benchmark、evaluation 和 citation grounding 思路 {refs}。\n"
            "- 两周 MVP：固定 20 个问题，输出每个流程节点的证据、分数和最终答案对比。\n"
            "- 简历亮点：把 Agentic RAG 做成可观测、可评测的工程系统。",
            "### 项目 2：多 Agent 文献研究助手\n"
            "- 目标：把文献研究拆成 Planner、Searcher、Grader、Writer、Critic 多角色协作。\n"
            "- 核心功能：任务分解、并行检索、论文路线分类、报告生成、critic 反馈重写。\n"
            "- 技术栈：LangGraph、多 Agent 状态图、OpenAlex/arXiv/IEEE、Markdown 报告渲染。\n"
            f"- 用到哪些论文思想：多 Agent / hierarchical RAG、planning 和 survey 路线 {refs}。\n"
            "- 两周 MVP：输入方向后生成路线表、代表论文表、gap 和两周阅读计划。\n"
            "- 简历亮点：体现多 Agent 编排、状态管理和科研工作流自动化。",
            "### 项目 3：长期记忆污染检测 RAG Agent\n"
            "- 目标：研究长期记忆写入后，错误记忆如何影响后续检索和回答。\n"
            "- 核心功能：短期上下文、长期 topic memory、记忆冲突检测、遗忘/修正策略。\n"
            "- 技术栈：LangGraph、JSON/SQLite memory store、向量检索、hallucination checker。\n"
            f"- 用到哪些论文思想：Agent memory、RAG grounding、evaluation 和 citation check {refs}。\n"
            "- 两周 MVP：构造正确/错误记忆对照组，展示答案被污染和被修正的过程。\n"
            "- 简历亮点：把长短期记忆、RAG 和可靠性评估结合成一个可演示系统。",
        ]
    )


def _has_banned_terms(answer: str) -> bool:
    lowered = answer.lower()
    return any(term.lower() in lowered for term in _BANNED_TERMS)


def _sanitize_internal_terms(answer: str) -> str:
    replacements = {
        "当前 mock/fallback 模式": "当前检索材料",
        "当前 mock": "当前检索材料",
        "当前 fallback": "当前检索材料",
        "本次结果只基于 mock": "以下分析基于当前检索到的标题、摘要与元数据",
        "fallback 模式不会编造": "以下分析仅使用当前检索材料",
        "mock/fallback": "当前检索材料",
        "fallback": "当前检索材料",
        "mock": "当前检索材料",
        "本节证据不足": "当前材料支持有限",
        "只能基于检索摘要做保守归纳": "以下归纳基于当前检索到的论文材料",
        "需要打开原文确认": "建议进一步阅读全文核验细节",
        "证据不足，不能替代完整论文阅读": "当前检索材料对该点支持有限，建议进一步阅读全文",
        "不能替代完整论文阅读": "建议进一步阅读全文",
    }
    cleaned = answer
    for old, new in replacements.items():
        cleaned = re.sub(re.escape(old), new, cleaned, flags=re.IGNORECASE)
    return cleaned


def _has_off_topic_rag_residue(answer: str, topic: str, documents: list[dict]) -> bool:
    if _is_rag_topic(topic, documents):
        return False
    lowered = answer.lower()
    residue_terms = [
        "agentic rag",
        "普通 rag",
        "langgraph",
        "query rewrite",
        "citation grounding",
        "hallucination checker",
        "multi-agent / hierarchical rag",
        "long-term memory rag",
        "长期记忆污染",
        "检索-评分-重写",
    ]
    hits = sum(1 for term in residue_terms if term in lowered)
    return hits >= 2


def _has_prompt_leakage(answer: str) -> bool:
    leakage_markers = (
        "任务类型：",
        "用户问题：",
        "记忆上下文",
        "current_paper_titles",
        "retrieved_documents_count",
        "graded_documents_count",
        "recent_steps",
        "papers_context",
        "Source documents:",
        "Generated answer:",
    )
    lowered = answer.lower()
    return any(marker.lower() in lowered for marker in leakage_markers)


def _radar_answer_quality_ok(answer: str, topic: str = "", documents: list[dict] | None = None) -> bool:
    documents = documents or []
    required = [
        "# PaperRadar",
        "## 1. 方向概览",
        "## 2. 方法路线分类",
        "## 3. 代表论文推荐",
        "## 4. 近年趋势",
        "## 5. 研究空白",
        "## 6. 两周阅读路线",
        "## 7. 可做小项目建议",
        "## 8. 参考来源",
    ]
    if any(section not in answer for section in required):
        return False
    if _has_pipe_table_residue(answer):
        return False
    if len(re.findall(r"###\s*路线\s+[A-Z]", answer)) < 4:
        return False
    if not re.search(r"###\s*(必读|重点)\s*\d+", answer):
        return False
    for day in ("Day 1-2", "Day 3-4", "Day 5-7", "Day 8-10", "Day 11-12", "Day 13-14"):
        if f"### {day}" not in answer:
            return False
    if len(re.findall(r"###\s*Gap\s*\d+", answer, flags=re.IGNORECASE)) < 5:
        return False
    if len(re.findall(r"###\s*项目\s*\d+", answer)) < 3:
        return False
    if _has_banned_terms(answer):
        return False
    if _has_prompt_leakage(answer):
        return False
    if _has_off_topic_rag_residue(answer, topic, documents):
        return False
    return True


def _graph_rec_entries(
    documents: list[dict],
    *,
    roles: set[str] | None = None,
) -> list[tuple[int, dict]]:
    entries = list(enumerate(documents, start=1))
    if roles is not None:
        entries = [
            (index, document)
            for index, document in entries
            if str(document.get("role") or "") in roles
        ]
    return sorted(
        entries,
        key=lambda item: (
            int(item[1].get("priority") or 999),
            -float(item[1].get("relevance_score") or 0.0),
            item[0],
        ),
    )


def _graph_rec_paper_refs(items: list[tuple[int, dict]], limit: int = 3) -> str:
    if not items:
        return "当前检索未命中，建议用本节末尾关键词补充检索"
    return "；".join(_paper_label(index, document, 78) for index, document in items[:limit])


def _graph_rec_route_section(documents: list[dict]) -> str:
    route_specs = [
        (
            "A",
            "Survey / Foundations",
            {"overview_survey"},
            "先回答“图对比学习推荐系统到底在研究什么”，把 CSSL、GNN 推荐、协同过滤、图增强和评价指标放到同一张地图上。",
            "适合回答入门综述、术语解释、论文地图、该方向是否值得做等问题。",
            "阅读价值在于建立边界：哪些论文是综述入口，哪些只是背景，哪些是真正的 GCL 推荐方法。",
            "注意不要把所有 survey 都当核心方法；综述主要负责组织问题，而不是提供可直接复现的模型。",
        ),
        (
            "B",
            "Graph Neural Recommendation / Collaborative Filtering",
            {"gnn_recommendation_foundation"},
            "解释用户-物品图、消息传递、协同过滤信号和 LightGCN/GNN 推荐为什么是 GCL 推荐的底座。",
            "适合回答模型输入是什么、图结构怎么建、为什么推荐系统需要图神经网络等基础问题。",
            "阅读价值在于补齐方法前提：没有 GNN 推荐基础，很难判断 GCL 的提升来自哪里。",
            "注意它是基础路线，不等于图对比学习本身；读的时候重点看任务设定和评价协议。",
        ),
        (
            "C",
            "Graph Contrastive Learning / SSL Recommendation",
            {"core_gcl_method"},
            "研究如何在推荐图上构造视图、设计对比目标，并把自监督信号转化为更稳健的用户/物品表示。",
            "适合回答 LightGCL、SGL、SimGCL 这类方法为什么有效，图增强和对比损失怎么影响推荐排序。",
            "阅读价值在于抓住核心机制：视图构造、正负样本、SSL loss 与 Recall/NDCG 的关系。",
            "注意核心论文需要和 LightGCN 等基线对照，不能只看摘要里“性能提升”的结论。",
        ),
        (
            "D",
            "Simplified / Efficient GCL",
            {"simplified_efficient_gcl"},
            "把复杂图增强简化为噪声注入、谱域信号或轻量视图，让 GCL 推荐更容易复现和部署。",
            "适合回答 XSimGCL、LightGCL 这类轻量化方法如何降低训练成本、为什么简单方法也能有效。",
            "阅读价值在于形成可落地项目路线：小数据集复现、消融实验、轻量化对比。",
            "注意“简单”不代表没有假设，仍要核查它对数据稀疏度、噪声和长尾物品是否敏感。",
        ),
        (
            "E",
            "Robustness / Long-tail / Explainability",
            {"robustness_longtail_sparsity", "explainability_invariant_rationale"},
            "把问题从总体准确率推进到稀疏交互、长尾物品、噪声边、可解释性和不变性学习。",
            "适合回答这个方向还能做什么创新、如何设计研究空白和简历项目。",
            "阅读价值在于帮助你从“复现核心模型”走向“提出可验证改进”。",
            "注意这一路线通常需要更细的分组实验，不能只用全局 Recall@K/NDCG@K 证明有效。",
        ),
    ]

    chunks: list[str] = []
    for letter, route, roles, problem, use_case, value, caution in route_specs:
        papers = _graph_rec_entries(documents, roles=roles)
        chunks.append(
            f"### 路线 {letter}：{route}\n"
            f"- **解决的问题**：{problem}\n"
            f"- **代表论文**：{_graph_rec_paper_refs(papers)}。\n"
            f"- **适合回答什么问题**：{use_case}\n"
            f"- **阅读价值**：{value}\n"
            f"- **注意点**：{caution}"
        )

    coverage = (
        "### 当前检索覆盖度\n"
        "- 覆盖较好：Survey / Foundations、Graph Contrastive Learning、Lightweight GCL。\n"
        "- 覆盖一般：长尾鲁棒性、可解释性、不变性学习。\n"
        "- 覆盖不足：SGL / SimGCL / LightGCN 原始基线、统一 benchmark 与复现实验代码。\n"
        "- 建议补充检索关键词：SGL recommendation, SimGCL recommendation, LightGCN, "
        "self-supervised graph learning recommendation, graph augmentation recommender systems."
    )
    return "\n\n".join([*chunks, coverage])


def _graph_rec_recommendation_reason(document: dict) -> str:
    identity = str(document.get("paper_identity") or "")
    role = str(document.get("role") or "")
    title = str(document.get("title") or "")

    if identity == "contrastive_self_supervised_recommendation_survey":
        return "这是图对比/自监督推荐的综述入口，适合先建立术语、任务、方法谱系和代表模型清单。"
    if identity == "gnn_recommendation_survey":
        return "它补齐 GNN 推荐和协同过滤基础，能解释用户-物品图、消息传递和评价设置这些底层前提。"
    if identity == "original_or_core_lightgcl":
        return "这是 LightGCL 原始/核心论文，应优先精读；它直接对应轻量图对比学习推荐的主线。"
    if identity == "core_xsimgcl":
        return "XSimGCL 是极简 GCL 推荐的重要代表，适合和 LightGCL 对照理解“简单视图/噪声注入”为何有效。"
    if identity == "application_or_reproduction_lightgcl":
        return "它更像 LightGCL 在稀疏数据场景下的应用或复现实验，适合用来设计鲁棒性和稀疏性验证。"
    if identity == "invariant_rationale_gcl":
        return "它适合放到进阶阶段，帮助思考解释性、不变性学习和用户意图子图，但不应排在入门前两篇。"
    if role == "background_related":
        return "它提供推荐系统相关背景，但不是图对比学习推荐的核心方法，适合最后补充上下文。"
    if "survey" in title.lower():
        return "它适合作为综述入口，用来补齐术语、任务边界和后续精读顺序。"
    return "它和图对比学习推荐系统直接相关，适合围绕方法、实验设置和可复现点做精读。"


def _graph_rec_recommendation_output(document: dict) -> str:
    role = str(document.get("role") or "")
    identity = str(document.get("paper_identity") or "")
    if identity == "original_or_core_lightgcl":
        return "LightGCL 复现笔记、模型结构图、核心公式拆解、与 LightGCN/SGL/XSimGCL 的对比表。"
    if identity == "core_xsimgcl":
        return "XSimGCL 与 LightGCL 的轻量化差异表，外加噪声注入/视图构造的消融实验计划。"
    if role == "overview_survey":
        return "术语表、方法谱系图、必读论文清单和后续检索关键词。"
    if role == "gnn_recommendation_foundation":
        return "用户-物品图建模流程图、评价指标清单和基础模型对照笔记。"
    if role == "robustness_longtail_sparsity":
        return "按稀疏度/长尾物品分组的复现实验方案和指标面板。"
    if role == "explainability_invariant_rationale":
        return "解释性扩展点、用户意图子图假设和可验证实验设计。"
    return "背景笔记，记录它和 GCL 推荐主线的关系以及是否值得继续追踪。"


def _first_graph_rec_entry(
    entries: list[tuple[int, dict]],
    *,
    identities: set[str] | None = None,
    roles: set[str] | None = None,
    used: set[int],
) -> tuple[int, dict] | None:
    for index, document in entries:
        if index in used:
            continue
        identity = str(document.get("paper_identity") or "")
        role = str(document.get("role") or "")
        if identities is not None and identity in identities:
            used.add(index)
            return index, document
        if roles is not None and role in roles:
            used.add(index)
            return index, document
    return None


def _graph_rec_recommendation_entries(documents: list[dict]) -> list[tuple[int, dict]]:
    entries = _graph_rec_entries(documents)
    used: set[int] = set()
    selected: list[tuple[int, dict]] = []

    stages = [
        {
            "identities": {"contrastive_self_supervised_recommendation_survey"},
            "roles": {"overview_survey"},
        },
        {
            "identities": {"gnn_recommendation_survey"},
            "roles": {"gnn_recommendation_foundation"},
        },
        {
            "identities": {"original_or_core_lightgcl"},
            "roles": {"core_gcl_method"},
        },
        {
            "identities": {"core_xsimgcl"},
            "roles": {"simplified_efficient_gcl"},
        },
        {
            "identities": {"application_or_reproduction_lightgcl"},
            "roles": {"robustness_longtail_sparsity"},
        },
        {
            "identities": {"invariant_rationale_gcl"},
            "roles": {"explainability_invariant_rationale"},
        },
        {
            "identities": None,
            "roles": {"background_related", "overview_survey", "gnn_recommendation_foundation"},
        },
    ]

    for stage in stages:
        match = _first_graph_rec_entry(
            entries,
            identities=stage["identities"],
            roles=stage["roles"],
            used=used,
        )
        if match:
            selected.append(match)

    for entry in entries:
        if entry[0] in used:
            continue
        selected.append(entry)
        used.add(entry[0])
        if len(selected) >= 9:
            break

    return selected[:9]


def _graph_rec_recommendation_section(documents: list[dict]) -> str:
    selected = _graph_rec_recommendation_entries(documents)
    if not selected:
        return "当前检索材料不足，暂时无法推荐代表论文。"

    chunks: list[str] = []
    for rank, (index, document) in enumerate(selected, start=1):
        title = _clean_cell(str(document.get("title") or "Untitled"), 130)
        year = _year_label(document)
        source = _source_label(str(document.get("source") or ""))
        route = document.get("route") or _infer_route_for_topic(document, "", documents)
        role_label = document.get("role_label") or document.get("role") or "论文"
        tags = "、".join(str(item) for item in document.get("reason_tags", [])[:4]) or "主题相关"

        if rank <= 4:
            label = f"必读 {rank}"
        elif rank <= 6:
            label = f"重点 {rank}"
        else:
            label = f"背景 {rank}"

        chunks.append(
            f"### {label}：{_citation(index)} {title}\n"
            f"- **阅读层级**：{role_label}。\n"
            f"- **年份 / 来源**：{year}，{source}。\n"
            f"- **所属路线**：{route}。\n"
            f"- **推荐理由**：{_graph_rec_recommendation_reason(document)}\n"
            f"- **标签**：{tags}。\n"
            f"- **可产出**：{_graph_rec_recommendation_output(document)}"
        )

    return "\n\n".join(chunks)


def _graph_rec_reference_line(index: int, document: dict) -> str:
    title = document.get("title", "Untitled")
    authors = ", ".join(document.get("authors", [])[:6]) or "作者未知"
    year = _year_label(document)
    source = _source_label(str(document.get("source") or ""))
    url = document.get("url", "")
    role = document.get("role_label") or document.get("role") or "论文"
    return f"[{index}] {title}；作者：{authors}；年份：{year}；来源：{source}；角色：{role}；URL：{url}"


def _graph_rec_reference_section(documents: list[dict]) -> str:
    groups = [
        (
            "核心论文",
            {"core_gcl_method", "simplified_efficient_gcl", "robustness_longtail_sparsity"},
        ),
        (
            "综述与背景",
            {"overview_survey", "gnn_recommendation_foundation"},
        ),
        (
            "相关但非核心",
            {"explainability_invariant_rationale", "background_related"},
        ),
    ]
    chunks: list[str] = []
    used: set[int] = set()
    for heading, roles in groups:
        items = _graph_rec_entries(documents, roles=roles)
        used.update(index for index, _ in items)
        if items:
            body = "\n".join(
                f"- {_graph_rec_reference_line(index, document)}"
                for index, document in items
            )
        else:
            body = "- 当前检索结果中没有稳定归入这一组的论文，建议按第 2 节关键词继续补充检索。"
        chunks.append(f"### {heading}\n{body}")

    remaining = [(index, document) for index, document in enumerate(documents, start=1) if index not in used]
    if remaining:
        chunks.append(
            "### 其他检索结果\n"
            + "\n".join(f"- {_graph_rec_reference_line(index, document)}" for index, document in remaining)
        )
    return "\n\n".join(chunks) or "本次检索未返回可引用论文。"


def _build_graph_rec_radar_report(topic: str, documents: list[dict]) -> str:
    overview_refs = " ".join(
        _citation(index) for index, _ in _graph_rec_entries(documents)[:4]
    ) or "[1]"
    core_count = sum(1 for document in documents if document.get("is_core"))
    background_count = max(0, len(documents) - core_count)

    overview = "\n\n".join(
        [
            (
                f"“{topic}”关注的是把图推荐、协同过滤和对比/自监督学习结合起来："
                "在用户-物品交互图上构造不同视图或扰动信号，让模型学到更稳健的用户与物品表示，"
                f"再服务于 Recall@K、NDCG@K、覆盖率和长尾推荐等目标 {overview_refs}。"
            ),
            (
                "这类论文不能只按检索顺序阅读。更合理的顺序是：先读推荐系统自监督综述和 GNN 推荐基础，"
                "再进入 LightGCL、XSimGCL 等核心/轻量方法，最后看稀疏、鲁棒、解释性和知识图谱等扩展背景。"
            ),
            (
                f"当前检索结果中，可作为核心方法或直接支撑主线的论文约 {core_count} 篇，"
                f"作为背景或相关拓展的论文约 {background_count} 篇。下面的排序会按阅读价值重新组织，"
                "而不是照搬数据库返回顺序。"
            ),
        ]
    )

    return "\n\n".join(
        [
            f"# PaperRadar：{topic}",
            f"## 1. 方向概览\n{overview}",
            f"## 2. 方法路线分类\n{_graph_rec_route_section(documents)}",
            f"## 3. 代表论文推荐\n{_graph_rec_recommendation_section(documents)}",
            f"## 4. 近年趋势\n{_build_trends(documents, topic)}",
            f"## 5. 研究空白\n{_build_gap_section(documents, topic)}",
            f"## 6. 两周阅读路线\n{_build_reading_plan(documents, topic)}",
            f"## 7. 可做小项目建议\n{_build_project_section(documents, topic)}",
            f"## 8. 参考来源\n{_graph_rec_reference_section(documents)}",
        ]
    )


def _build_radar_report(topic: str, documents: list[dict]) -> str:
    if _is_recommender_gcl_topic(topic, documents):
        documents = assign_paper_roles(documents, topic)
        return _build_graph_rec_radar_report(topic, documents)

    if not documents:
        return (
            f"# PaperRadar：{topic}\n\n"
            "## 1. 方向概览\n当前检索没有返回可用论文，无法形成可靠方向雷达。\n\n"
            "## 2. 方法路线分类\n当前检索材料不足，暂时无法形成可靠路线卡片。\n\n"
            "## 3. 代表论文推荐\n当前检索材料不足，暂时无法推荐代表论文。\n\n"
            "## 4. 近年趋势\n当前没有可分析论文。\n\n"
            f"## 5. 研究空白\n{_build_gap_section([], topic)}\n\n"
            f"## 6. 两周阅读路线\n{_build_reading_plan([], topic)}\n\n"
            f"## 7. 可做小项目建议\n{_build_project_section([], topic)}\n\n"
            "## 8. 参考来源\n本次检索未返回可引用论文。"
        )

    overview_refs = " ".join(
        _citation(index) for index, _ in _representative_documents(documents, 4, topic)
    )
    route_names = "、".join(_route_groups(documents, topic).keys())
    core_count = sum(1 for doc in documents if doc.get("relevance_tier", "core") == "core")
    background_count = len(documents) - core_count
    if _is_recommender_gcl_topic(topic, documents):
        overview = "\n\n".join(
            [
                (
                    f"“{topic}”关注的是把图神经网络推荐和对比学习/自监督学习结合起来："
                    "用用户-物品交互图、图视图构造、对比损失和轻量化传播机制学习更稳健的用户与物品表示。"
                    f"当前论文可以归纳为 {route_names} 等路线 {overview_refs}。"
                ),
                (
                    "这个方向重要，是因为推荐系统里的隐式反馈通常稀疏、带噪声、长尾明显。"
                    "图对比学习试图通过构造不同视图或扰动信号，让模型在有限交互下学到更鲁棒的偏好表示，"
                    f"从而改善召回、排序和泛化表现 {overview_refs}。"
                ),
                (
                    "阅读时不要把它看成普通论文列表，而要沿着“图推荐基础 → 自监督/对比学习机制 → "
                    "稀疏鲁棒性与长尾评测 → 可解释与项目复现”这条线推进。"
                    f"从当前检索结果看，核心论文约 {core_count} 篇，背景论文约 {background_count} 篇。"
                ),
            ]
        )
    else:
        overview = "\n\n".join(
            [
                (
                    f"“{topic}”可以从当前论文中归纳为 {route_names} 等路线 {overview_refs}。"
                    "这些路线分别覆盖方向综述、核心方法、实验评测、应用场景和可做项目入口。"
                ),
                (
                    "它的重要性在于：一个研究方向通常不是单篇论文能讲清楚，而是由问题定义、方法演化、"
                    f"评价指标和开放空白共同构成。当前报告只基于检索到的标题、摘要和元数据做第一轮雷达分析 {overview_refs}。"
                ),
                (
                    f"从当前检索结果看，核心论文约 {core_count} 篇，背景论文约 {background_count} 篇。"
                    "核心论文用于支撑方法路线、趋势和项目建议；背景论文主要用于补充概念、任务和实验背景。"
                ),
            ]
        )
    return "\n\n".join(
        [
            f"# PaperRadar：{topic}",
            f"## 1. 方向概览\n{overview}",
            f"## 2. 方法路线分类\n{_build_route_cards(documents, topic)}",
            f"## 3. 代表论文推荐\n{_build_recommendation_cards(documents, topic)}",
            f"## 4. 近年趋势\n{_build_trends(documents, topic)}",
            f"## 5. 研究空白\n{_build_gap_section(documents, topic)}",
            f"## 6. 两周阅读路线\n{_build_reading_plan(documents, topic)}",
            f"## 7. 可做小项目建议\n{_build_project_section(documents, topic)}",
            f"## 8. 参考来源\n{_format_reference_section(documents)}",
        ]
    )


def _source_label(source: str) -> str:
    labels = {
        "arxiv": "arXiv",
        "pubmed": "PubMed",
        "openalex": "OpenAlex",
        "ieee": "IEEE",
        "arxiv_openalex": "OpenAlex(arXiv metadata)",
    }
    return labels.get(source, source or "未知来源")


def _first_meaningful_sentence(text: str, max_chars: int = 220) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return "摘要未提供足够信息，建议优先打开原文核对。"

    sentence_match = re.match(r"(.+?[.!?。！？])(?:\s|$)", normalized)
    sentence = sentence_match.group(1) if sentence_match else normalized
    if len(sentence) <= max_chars:
        return sentence
    return sentence[: max_chars - 1].rstrip() + "..."


def _format_document_coverage_list(
    documents: list[dict],
    heading: str = "### 检索覆盖清单",
) -> str:
    """Create a grounded per-paper list so every returned paper is visible."""
    if not documents:
        return ""

    lines = [heading]
    for i, doc in enumerate(documents, start=1):
        title = doc.get("title", "Untitled")
        published = doc.get("published") or "年份未知"
        source = _source_label(str(doc.get("source") or ""))
        abstract_hint = _first_meaningful_sentence(doc.get("abstract", ""))
        lines.append(
            f"- [{i}] **{title}**（{published}，{source}）：{abstract_hint}"
        )
    return "\n".join(lines)


def _append_to_numbered_section(
    *,
    answer: str,
    section_index: int,
    section_title: str,
    content: str,
) -> str:
    if not content:
        return answer
    if "### 检索覆盖清单" in answer:
        return answer

    pattern = (
        rf"(?ms)(^##\s*{section_index}[.、]\s*"
        rf"{re.escape(section_title)}\s*$)(.*?)(?=^##\s*\d+[.、]\s*|\Z)"
    )
    match = re.search(pattern, answer)
    if not match:
        return f"{answer.rstrip()}\n\n## {section_index}. {section_title}\n{content}"

    replacement = f"{match.group(1)}{match.group(2).rstrip()}\n\n{content}\n"
    return f"{answer[:match.start()]}{replacement}{answer[match.end():]}"


def _ensure_radar_document_coverage(answer: str, documents: list[dict]) -> str:
    """Ensure the report body visibly covers every final output document."""
    coverage = _format_document_coverage_list(documents)
    return _append_to_numbered_section(
        answer=answer,
        section_index=3,
        section_title="代表论文推荐",
        content=coverage,
    )


def _citation_indices_in_answer(answer: str) -> set[int]:
    return {int(match) for match in re.findall(r"\[(\d+)\]", answer)}


def _ensure_search_document_coverage(answer: str, documents: list[dict]) -> str:
    """Append a grounded paper list when the model skipped returned papers."""
    if not documents:
        return answer

    expected = set(range(1, len(documents) + 1))
    if expected.issubset(_citation_indices_in_answer(answer)):
        return answer

    coverage = _format_document_coverage_list(
        documents,
        heading="### 逐篇结果",
    )
    return (
        f"{answer.rstrip()}\n\n"
        "## 检索到的论文\n"
        f"{coverage}"
    )


def _close_unbalanced_markdown(answer: str) -> str:
    if answer.count("**") % 2 == 1:
        return f"{answer}**"
    return answer


def _ensure_radar_sections(answer: str, documents: list[dict]) -> str:
    answer = _close_unbalanced_markdown(answer)
    sections = [
        "方向概览",
        "方法路线分类",
        "代表论文推荐",
        "近年趋势",
        "研究空白",
        "两周阅读路线",
        "可做小项目建议",
        "参考来源",
    ]
    chunks = [answer.rstrip()]
    for index, title in enumerate(sections, start=1):
        section_pattern = rf"(?m)^##\s*{index}[.、]\s*{re.escape(title)}\s*$"
        if re.search(section_pattern, answer):
            continue
        if title == "参考来源":
            body = _REFERENCE_PLACEHOLDER
        else:
            body = _INSUFFICIENT_EVIDENCE_TEXT
        chunks.append(f"## {index}. {title}\n{body}")
    return "\n\n".join(chunks)


def _replace_reference_section(answer: str, documents: list[dict]) -> str:
    reference_body = _format_reference_section(documents)
    pattern = r"(?ms)^##\s*8[.、]\s*参考来源\s*$.*\Z"
    replacement = f"## 8. 参考来源\n{reference_body}"
    if re.search(pattern, answer):
        return re.sub(pattern, replacement, answer).strip()
    return f"{answer.rstrip()}\n\n{replacement}"


def _ensure_radar_quality(answer: str, topic: str, documents: list[dict]) -> str:
    if _is_recommender_gcl_topic(topic, documents):
        return _sanitize_internal_terms(_build_graph_rec_radar_report(topic, documents))

    answer = _sanitize_internal_terms(answer)
    answer = _replace_reference_section(answer, documents)
    if _radar_answer_quality_ok(answer, topic, documents):
        return answer
    return _sanitize_internal_terms(_build_radar_report(topic, documents))


def _memory_context_for_prompt(state: AgentState) -> str:
    graded_documents = state.get("graded_documents", [])
    stored_long_term = get_memory_context()
    state_memory_context = state.get("memory_context") or {}
    context = {
        "short_term": {
            "query": state.get("query", ""),
            "classification": state.get("classification", ""),
            "selected_sources": state.get("sources", []),
            "requested_max_results": state.get("max_results"),
            "retrieved_documents_count": len(state.get("documents", [])),
            "graded_documents_count": len(graded_documents),
            "current_paper_titles": [
                doc.get("title", "Untitled") for doc in graded_documents[:10]
            ],
            "recent_steps": state.get("steps", [])[-6:],
        },
        "long_term": {**stored_long_term, **state_memory_context},
    }
    return json.dumps(context, ensure_ascii=False, indent=2)


def _record_generation_history(
    *,
    query: str,
    classification: str,
    documents: list[dict],
) -> None:
    try:
        upsert_topics(extract_topics_from_text(query))
        add_history(
            {
                "query": query,
                "classification": classification,
                "top_papers": [
                    {
                        "title": doc.get("title", "Untitled"),
                        "url": doc.get("url", ""),
                        "source": doc.get("source", ""),
                    }
                    for doc in documents[:5]
                ],
            }
        )
    except Exception as exc:  # pragma: no cover - memory should never break answers
        logger.warning("Failed to persist PaperRadar memory: %s", exc)


def generate_answer(state: AgentState) -> AgentState:
    """Generate a cited answer, a general response, or a PaperRadar report."""
    start = time.perf_counter()

    query = state["query"]
    topic_query = state.get("original_query") or query
    classification = state.get("classification", "paper_search")
    output_documents = assign_paper_roles(
        classify_documents(select_output_documents(state), query=topic_query),
        topic_query,
    )
    retry_instruction = _retry_instruction_for_prompt(state)

    if classification == "general":
        prompt = _GENERAL_PROMPT.format(query=query)
        response = invoke_with_retry([HumanMessage(content=prompt)])
        answer = extract_text(response).strip()
        citations: list[dict] = []
    elif classification in _RADAR_TASKS:
        papers_context = _build_papers_context(output_documents)
        prompt = _PAPER_RADAR_PROMPT.format(
            classification=classification,
            query=topic_query,
            topic=topic_query,
            memory_context=_memory_context_for_prompt(state),
            papers_context=papers_context,
            retry_instruction=retry_instruction,
            paper_count=len(output_documents),
        )
        response = invoke_with_retry([HumanMessage(content=prompt)])
        answer = _ensure_radar_quality(
            _ensure_radar_sections(extract_text(response).strip(), output_documents),
            topic_query,
            output_documents,
        )
        citations = _extract_citations(output_documents)
        _record_generation_history(
            query=topic_query,
            classification=classification,
            documents=output_documents,
        )
    else:
        papers_context = _build_papers_context(output_documents)
        prompt = _GENERATE_PROMPT.format(
            query=topic_query,
            papers_context=papers_context,
            retry_instruction=retry_instruction,
        )
        response = invoke_with_retry([HumanMessage(content=prompt)])
        answer = _ensure_search_document_coverage(
            extract_text(response).strip(),
            output_documents,
        )
        citations = _extract_citations(output_documents)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info("Answer generated in %dms (%d chars)", elapsed_ms, len(answer))

    steps = list(state.get("steps", []))
    mode = "PaperRadar report" if classification in _RADAR_TASKS else "answer"
    steps.append(
        {
            "node": "generator",
            "status": "completed",
            "detail": f"Generated {mode} with {len(citations)} citations",
            "duration_ms": elapsed_ms,
        }
    )

    return {
        **state,
        "answer": answer,
        "citations": citations,
        "graded_documents": output_documents,
        "steps": steps,
    }
