"""Rule-based helpers for PaperRadar paper route classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RouteRule:
    name: str
    keywords: tuple[str, ...]
    reason: str


ROUTE_RULES: tuple[RouteRule, ...] = (
    RouteRule(
        "Survey / Taxonomy / SoK",
        ("survey", "taxonomy", "sok", "overview", "review", "comprehensive"),
        "适合建立方向全景、术语体系和方法演化脉络。",
    ),
    RouteRule(
        "Graph Contrastive Learning / SSL Recommendation",
        (
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
        "聚焦推荐系统中的图对比学习、自监督信号、视图构造和表征鲁棒性。",
    ),
    RouteRule(
        "Graph Neural Recommendation / Collaborative Filtering",
        (
            "graph neural",
            "gnn",
            "graph learning",
            "collaborative filtering",
            "recommender",
            "recommendation",
            "recommendation system",
            "user-item",
            "user item",
            "bipartite",
            "lightgcn",
        ),
        "聚焦用户-物品图建模、GNN 推荐和协同过滤表征学习。",
    ),
    RouteRule(
        "Data Sparsity / Robustness / Debiasing",
        (
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
        "关注推荐数据稀疏、噪声、冷启动、流行度偏差和长尾鲁棒性。",
    ),
    RouteRule(
        "Knowledge Graph / Side Information Recommendation",
        (
            "knowledge graph",
            "knowledge graph-based",
            "kg-based",
            "side information",
            "semantic",
            "entity",
            "relation",
        ),
        "关注知识图谱、实体关系和辅助信息如何增强推荐。",
    ),
    RouteRule(
        "Planning / Reasoning",
        (
            "planning",
            "reasoning",
            "iterative retrieval",
            "reflection",
            "self-reflection",
            "system 1",
            "system 2",
            "query rewrite",
            "rewriting",
            "decomposition",
        ),
        "聚焦 Agentic RAG 中的规划、迭代检索、反思和查询改写循环。",
    ),
    RouteRule(
        "Multi-Agent / Hierarchical",
        (
            "multi-agent",
            "multi agent",
            "hierarchical",
            "collaboration",
            "cooperation",
            "coordinator",
            "planner",
            "critic",
        ),
        "体现多 Agent 分工、层级协作和检索-生成角色拆分。",
    ),
    RouteRule(
        "Multimodal RAG",
        (
            "multimodal",
            "multi-modal",
            "vision",
            "visual",
            "mllm",
            "mrag",
            "image",
            "video",
            "3d",
        ),
        "扩展到图像、视频或多模态证据，强调跨模态证据对齐。",
    ),
    RouteRule(
        "Evaluation / Benchmark",
        (
            "benchmark",
            "evaluation",
            "evaluate",
            "metric",
            "metrics",
            "assessment",
            "leaderboard",
            "infodeepseek",
            "hallucination",
            "citation",
        ),
        "用于比较系统效果、事实性、引用准确性和检索质量。",
    ),
    RouteRule(
        "Domain-specific Agentic RAG",
        (
            "fintech",
            "finance",
            "financial",
            "medical",
            "medicine",
            "healthcare",
            "clinical",
            "legal",
            "law",
            "industry",
            "domain-specific",
            "education",
        ),
        "说明 Agentic RAG 在垂直领域中的术语、知识源和评测约束。",
    ),
    RouteRule(
        "Knowledge Gap / Research Discovery",
        (
            "knowledge gap",
            "research gap",
            "gap discovery",
            "scientific discovery",
            "research discovery",
            "hypothesis",
        ),
        "关注从文献中发现研究空白、生成假设和辅助选题。",
    ),
)

_GENERAL_RAG_TERMS = (
    "rag",
    "retrieval-augmented",
    "retrieval augmented",
    "retrieval",
    "large language model",
    "llm",
    "agent",
    "agentic",
)

_STOPWORDS = {
    "this",
    "that",
    "with",
    "from",
    "into",
    "about",
    "paper",
    "papers",
    "study",
    "using",
    "based",
    "large",
    "language",
    "model",
    "models",
    "systems",
    "recommender",
    "recommendation",
}


def extract_year(value: Any) -> int | None:
    """Extract a year from a publication date or free-form text."""
    match = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _find_keywords(text: str, candidates: tuple[str, ...]) -> list[str]:
    return [keyword for keyword in candidates if keyword in text]


def extract_keywords(title: str, abstract: str, limit: int = 8) -> list[str]:
    """Extract simple route-oriented keywords from title and abstract."""
    text = _normalize(f"{title} {abstract}")
    keywords: list[str] = []
    for rule in ROUTE_RULES:
        keywords.extend(_find_keywords(text, rule.keywords))
    keywords.extend(_find_keywords(text, _GENERAL_RAG_TERMS))

    if len(keywords) < limit:
        for token in re.findall(r"[a-z][a-z0-9-]{3,}", text):
            if token in _STOPWORDS or token in keywords:
                continue
            keywords.append(token)
            if len(keywords) >= limit:
                break

    deduped: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        if keyword in seen:
            continue
        seen.add(keyword)
        deduped.append(keyword)
        if len(deduped) >= limit:
            break
    return deduped


def classify_paper(
    title: str,
    abstract: str,
    published: Any = None,
    query: str = "",
) -> dict[str, Any]:
    """Classify a paper into a PaperRadar route using title/abstract rules."""
    combined = _normalize(f"{title} {abstract}")
    normalized_title = _normalize(title)
    query_text = _normalize(query)

    if any(marker in normalized_title for marker in ("survey", "review", "taxonomy", "sok")):
        return {
            "route": "Survey / Taxonomy / SoK",
            "category": "Survey / Taxonomy / SoK",
            "year": extract_year(published),
            "keywords": extract_keywords(title, abstract),
            "matched_keywords": [
                marker
                for marker in ("survey", "review", "taxonomy", "sok")
                if marker in normalized_title
            ],
            "relevance_reason": "标题明确表明这是综述/分类论文，适合建立方向全景、术语体系和方法演化脉络。",
        }

    best_rule: RouteRule | None = None
    best_hits: list[str] = []
    for rule in ROUTE_RULES:
        hits = _find_keywords(combined, rule.keywords)
        if len(hits) > len(best_hits):
            best_rule = rule
            best_hits = hits

    if best_rule is None:
        general_hits = _find_keywords(combined, _GENERAL_RAG_TERMS)
        if general_hits:
            route = "General RAG / Agentic RAG"
            reason = "与 RAG、LLM 或 Agent 基础概念相关，可作为背景或方法基线。"
            best_hits = general_hits
        else:
            route = "Background"
            reason = "标题和摘要缺少与 Agentic RAG 直接相关的路线关键词。"
    else:
        route = best_rule.name
        reason = best_rule.reason

    query_terms = [
        term
        for term in re.findall(r"[a-z][a-z0-9-]{2,}", query_text)
        if term not in _STOPWORDS
    ]
    overlap = [term for term in query_terms if term in combined]
    if overlap:
        reason = f"{reason} 与用户主题关键词 {', '.join(overlap[:4])} 有直接重合。"

    return {
        "route": route,
        "category": route,
        "year": extract_year(published),
        "keywords": extract_keywords(title, abstract),
        "matched_keywords": best_hits,
        "relevance_reason": reason,
    }


def classify_document(document: dict[str, Any], query: str = "") -> dict[str, Any]:
    """Return a copy of a paper dict enriched with route metadata."""
    classification = classify_paper(
        title=str(document.get("title") or ""),
        abstract=str(document.get("abstract") or ""),
        published=document.get("published"),
        query=query,
    )
    existing_reason = str(document.get("relevance_reason") or "").strip()
    rule_reason = classification["relevance_reason"]
    relevance_reason = (
        f"{existing_reason}；{rule_reason}" if existing_reason else rule_reason
    )
    return {
        **document,
        "route": classification["route"],
        "category": classification["category"],
        "year": classification["year"],
        "keywords": classification["keywords"],
        "relevance_reason": relevance_reason,
    }


def classify_documents(
    documents: list[dict[str, Any]],
    query: str = "",
) -> list[dict[str, Any]]:
    return [classify_document(document, query=query) for document in documents]
