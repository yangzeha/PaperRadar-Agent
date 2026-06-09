"""Paper role classification used by topic-specific PaperRadar reports."""

from __future__ import annotations

import re
from typing import Any


GRAPH_REC_TOPIC_MARKERS = (
    "图对比学习",
    "图对比",
    "graph contrastive",
    "contrastive learning",
    "self-supervised",
    "self supervised",
    "gcl",
    "lightgcl",
    "xsimgcl",
    "simgcl",
    "sgl",
)

RECOMMENDER_TOPIC_MARKERS = (
    "推荐",
    "推荐系统",
    "推荐算法",
    "recommender",
    "recommendation",
    "collaborative filtering",
    "user-item",
    "user item",
)

ROLE_LABELS = {
    "overview_survey": "综述",
    "gnn_recommendation_foundation": "GNN 推荐基础",
    "core_gcl_method": "核心方法",
    "simplified_efficient_gcl": "轻量高效 GCL",
    "robustness_longtail_sparsity": "稀疏/鲁棒",
    "explainability_invariant_rationale": "解释/不变性",
    "background_related": "相关但非核心",
}

ROLE_ROUTES = {
    "overview_survey": "Survey / Foundations",
    "gnn_recommendation_foundation": "Graph Neural Recommendation / Collaborative Filtering",
    "core_gcl_method": "Graph Contrastive Learning / SSL Recommendation",
    "simplified_efficient_gcl": "Simplified / Efficient GCL",
    "robustness_longtail_sparsity": "Robustness / Long-tail / Explainability",
    "explainability_invariant_rationale": "Robustness / Long-tail / Explainability",
    "background_related": "Background Related",
}


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def document_text(document: dict[str, Any]) -> str:
    keywords = " ".join(str(item) for item in document.get("keywords", []) or [])
    return normalize_text(
        " ".join(
            [
                str(document.get("title") or ""),
                str(document.get("abstract") or ""),
                keywords,
            ]
        )
    )


def is_graph_contrastive_recommendation_topic(
    topic: str,
    documents: list[dict[str, Any]] | None = None,
) -> bool:
    text = normalize_text(topic)
    if documents:
        text = " ".join([text, *[document_text(document) for document in documents[:12]]])
    has_graph_contrastive = any(marker in text for marker in GRAPH_REC_TOPIC_MARKERS)
    has_recommender = any(marker in text for marker in RECOMMENDER_TOPIC_MARKERS)
    return has_graph_contrastive and has_recommender


def _tags(*items: str) -> list[str]:
    return [item for item in items if item]


def classify_graph_rec_paper(document: dict[str, Any]) -> dict[str, Any]:
    title = normalize_text(document.get("title"))
    text = document_text(document)

    role = "background_related"
    priority = 90
    identity = ""
    reason_tags = _tags("相关背景")
    is_core = False
    is_background = True

    if "contrastive self-supervised learning in recommender systems" in title and "survey" in title:
        role = "overview_survey"
        priority = 10
        identity = "contrastive_self_supervised_recommendation_survey"
        reason_tags = _tags("CSSL 综述", "推荐系统自监督入口", "方法地图")
        is_core = True
        is_background = False
    elif (
        "graph neural networks in recommender systems" in title
        or ("graph neural network" in title and "recommender" in title and "survey" in title)
        or ("graph neural networks" in title and "recommend" in title and "survey" in title)
    ):
        role = "gnn_recommendation_foundation"
        priority = 20
        identity = "gnn_recommendation_survey"
        reason_tags = _tags("GNN 推荐基础", "用户-物品图", "协同过滤背景")
        is_core = True
        is_background = False
    elif "lightgcl: simple yet effective graph contrastive learning for recommendation" in title:
        role = "core_gcl_method"
        priority = 30
        identity = "original_or_core_lightgcl"
        reason_tags = _tags("LightGCL 原始/核心论文", "轻量图对比学习", "可复现主线")
        is_core = True
        is_background = False
    elif "xsimgcl" in title:
        role = "simplified_efficient_gcl"
        priority = 40
        identity = "core_xsimgcl"
        reason_tags = _tags("XSimGCL", "极简图对比学习", "轻量化路线")
        is_core = True
        is_background = False
    elif "with lightgcl" in title or "optimizing sparse data" in title:
        role = "robustness_longtail_sparsity"
        priority = 50
        identity = "application_or_reproduction_lightgcl"
        reason_tags = _tags("LightGCL 应用/复现", "稀疏数据", "鲁棒性验证")
        is_core = True
        is_background = False
    elif "invariant rationale" in title:
        role = "explainability_invariant_rationale"
        priority = 60
        identity = "invariant_rationale_gcl"
        reason_tags = _tags("不变性学习", "解释性", "进阶扩展")
        is_core = False
        is_background = False
    elif any(marker in title for marker in ("knowledge graph", "kg-based", "multi-objective", "multi objective")):
        role = "background_related"
        priority = 95
        identity = "background_related_survey"
        reason_tags = _tags("背景综述", "非核心 GCL", "可补充上下文")
    elif "graph learning" in title and ("recommender" in title or "recommendation" in title):
        role = "background_related"
        priority = 85
        identity = "graph_learning_recommender_background"
        reason_tags = _tags("图学习推荐背景", "非核心 GCL", "基础补充")
    elif "survey" in title or ("review" in title and "review-aware" not in title):
        role = "overview_survey"
        priority = 25
        identity = "topic_survey"
        reason_tags = _tags("综述", "术语地图", "阅读入口")
        is_core = True
        is_background = False
    elif any(marker in text for marker in ("lightgcl", "simgcl", "sgl", "graph contrastive", "contrastive learning", "self-supervised")):
        role = "core_gcl_method"
        priority = 45
        identity = "gcl_recommendation_method"
        reason_tags = _tags("GCL 推荐方法", "核心路线")
        is_core = True
        is_background = False
    elif any(marker in text for marker in ("sparse", "sparsity", "cold-start", "cold start", "long-tail", "robust", "noise")):
        role = "robustness_longtail_sparsity"
        priority = 55
        identity = "robustness_sparsity_method"
        reason_tags = _tags("稀疏/长尾", "鲁棒性")
        is_core = True
        is_background = False
    elif any(marker in text for marker in ("graph neural", "gnn", "collaborative filtering", "user-item", "user item", "lightgcn")):
        role = "gnn_recommendation_foundation"
        priority = 35
        identity = "gnn_recommendation_foundation"
        reason_tags = _tags("GNN 推荐基础", "协同过滤")
        is_core = True
        is_background = False

    return {
        "role": role,
        "role_label": ROLE_LABELS[role],
        "route": ROLE_ROUTES[role],
        "priority": priority,
        "reason_tags": reason_tags,
        "is_core": is_core,
        "is_background": is_background,
        "paper_identity": identity,
    }


def assign_paper_roles(
    documents: list[dict[str, Any]],
    topic: str,
) -> list[dict[str, Any]]:
    if not documents:
        return []

    if not is_graph_contrastive_recommendation_topic(topic, documents):
        assigned = []
        for index, document in enumerate(documents, start=1):
            route = document.get("route") or document.get("category") or "General Topic / Method Background"
            tier = document.get("relevance_tier") or "core"
            assigned.append(
                {
                    **document,
                    "route": route,
                    "role": document.get("role") or ("core" if tier == "core" else "background"),
                    "role_label": document.get("role_label") or ("核心论文" if tier == "core" else "背景论文"),
                    "priority": document.get("priority") or index * 10,
                    "reason_tags": document.get("reason_tags") or [],
                    "is_core": document.get("is_core", tier == "core"),
                    "is_background": document.get("is_background", tier != "core"),
                    "paper_identity": document.get("paper_identity") or "",
                }
            )
        return assigned

    return [
        {
            **document,
            **classify_graph_rec_paper(document),
        }
        for document in documents
    ]


def role_label(role: str | None) -> str:
    return ROLE_LABELS.get(str(role or ""), "论文")
