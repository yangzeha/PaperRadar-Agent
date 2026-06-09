"""Retriever node — fetches papers from external APIs and indexes them in ChromaDB."""

import asyncio
import logging
import re
import time

from app.agents.state import AgentState
from app.config import settings
from app.services.paper_roles import (
    assign_paper_roles,
    is_graph_contrastive_recommendation_topic,
)

logger = logging.getLogger(__name__)

_QUERY_EXPANSIONS = {
    "agentic rag": (
        "agentic retrieval augmented generation autonomous agents planning "
        "verification grounded generation"
    ),
    "rag": "retrieval augmented generation",
    "llm agent": "large language model agents autonomous agents",
    "图对比学习推荐算法": "graph contrastive learning recommender systems",
    "图对比学习推荐": "graph contrastive learning recommender systems",
    "图对比学习": "graph contrastive learning",
    "对比学习": "contrastive learning",
    "推荐算法": "recommendation algorithm recommender systems",
    "推荐系统": "recommender systems recommendation",
    "推荐": "recommender systems recommendation",
    "协同过滤": "collaborative filtering",
    "知识图谱": "knowledge graph",
    "图神经网络": "graph neural network GNN",
    "序列推荐": "sequential recommendation",
    "多模态推荐": "multimodal recommendation",
    "长期记忆": "long-term memory",
    "论文雷达": "survey representative papers",
    "研究空白": "research gaps",
    "趋势": "research trends",
}

_STOP_WORDS = {
    "and",
    "are",
    "for",
    "from",
    "have",
    "paper",
    "papers",
    "the",
    "with",
}

_GRAPH_REC_EXPANDED_QUERIES = [
    "graph contrastive learning recommendation",
    "contrastive self-supervised learning recommender systems",
    "LightGCL recommendation",
    "XSimGCL recommendation",
    "SGL self-supervised graph learning recommendation",
    "SimGCL recommendation",
    "LightGCN graph collaborative filtering",
    "graph neural networks recommender systems survey",
    "self-supervised recommendation survey",
]


def _unique_terms(text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for term in re.findall(r"[A-Za-z][A-Za-z0-9+#./-]*|\d{4}", text):
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def _query_keywords(search_query: str) -> list[str]:
    return [
        term.lower()
        for term in _unique_terms(search_query)
        if len(term) > 2
        and term.lower() not in _STOP_WORDS
        and not re.fullmatch(r"\d{4}", term)
    ]


def _paper_relevance_score(paper: dict, search_query: str) -> int:
    title = str(paper.get("title") or "").lower()
    abstract = str(paper.get("abstract") or "").lower()
    combined = f"{title}\n{abstract}"
    query = search_query.lower()
    keywords = _query_keywords(search_query)

    score = 0
    important_phrases = [
        "graph contrastive learning",
        "recommender systems",
        "recommendation",
        "collaborative filtering",
        "long-term memory",
        "retrieval augmented generation",
    ]
    for phrase in important_phrases:
        if phrase not in query:
            continue
        if phrase in title:
            score += 10
        elif phrase in combined:
            score += 5

    for keyword in keywords:
        if keyword in title:
            score += 3
        elif keyword in abstract:
            score += 1

    return score


def _rank_papers_for_query(papers: list[dict], search_query: str) -> list[dict]:
    scored_papers = [
        {
            **paper,
            "relevance_score": float(_paper_relevance_score(paper, search_query)),
        }
        for paper in papers
    ]
    return sorted(
        scored_papers,
        key=lambda paper: float(paper.get("relevance_score") or 0.0),
        reverse=True,
    )


def _matched_expansions(query: str) -> list[str]:
    expansions: list[str] = []
    occupied_spans: list[tuple[int, int]] = []
    for chinese, english in sorted(
        _QUERY_EXPANSIONS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        for match in re.finditer(re.escape(chinese), query, flags=re.IGNORECASE):
            start, end = match.span()
            overlaps_existing = any(
                max(start, occupied_start) < min(end, occupied_end)
                for occupied_start, occupied_end in occupied_spans
            )
            if overlaps_existing:
                continue
            occupied_spans.append((start, end))
            expansions.append(english)
            break
    return expansions


def _external_search_query(query: str) -> str:
    additions = _matched_expansions(query)
    if not additions:
        return query
    expanded_terms = _unique_terms(" ".join(additions))
    original_english_terms = [
        term
        for term in _unique_terms(query)
        if not re.fullmatch(r"\d{4}", term)
    ]
    search_terms = _unique_terms(" ".join([*original_english_terms, *expanded_terms]))
    if not search_terms:
        return query
    return " ".join(search_terms)


def _expanded_search_queries(query: str) -> list[str]:
    primary_query = _external_search_query(query)
    if not is_graph_contrastive_recommendation_topic(query):
        return [primary_query]

    queries = [primary_query, *_GRAPH_REC_EXPANDED_QUERIES]
    unique: list[str] = []
    seen: set[str] = set()
    for item in queries:
        normalized = re.sub(r"\s+", " ", item).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


async def _search_expanded_queries(
    fetcher,
    *,
    queries: list[str],
    sources: list[str],
    max_results: int,
    year_range: tuple[int, int] | None,
):
    if len(queries) == 1:
        return await fetcher.search(
            query=queries[0],
            sources=sources,
            max_results=max_results,
            year_range=year_range,
        )

    per_query_limit = max(6, min(12, max_results))
    results = await asyncio.gather(
        *[
            fetcher.search(
                query=query,
                sources=sources,
                max_results=per_query_limit,
                year_range=year_range,
            )
            for query in queries
        ],
        return_exceptions=True,
    )
    papers = []
    for result in results:
        if isinstance(result, BaseException):
            logger.warning("Expanded paper search failed: %s", result)
            continue
        papers.extend(result)
    return papers


def _extract_year_range(query: str) -> tuple[int, int] | None:
    """Extract a publication year constraint from user text when present."""
    patterns = [
        r"(20\d{2})\s*[-~—–至到]\s*(20\d{2})\s*年?",
        r"(20\d{2})\s*年\s*[-~—–至到]\s*(20\d{2})\s*年",
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            return (min(start, end), max(start, end))
    return None


def _paper_year(paper: dict) -> int | None:
    published = str(paper.get("published") or "")
    match = re.search(r"(19|20)\d{2}", published)
    return int(match.group(0)) if match else None


def _filter_by_year_range(
    papers: list[dict],
    year_range: tuple[int, int] | None,
) -> list[dict]:
    if year_range is None:
        return papers
    start, end = year_range
    return [
        paper
        for paper in papers
        if (year := _paper_year(paper)) is not None and start <= year <= end
    ]


def _dedupe_papers(papers: list[dict]) -> list[dict]:
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[dict] = []
    for paper in papers:
        url_key = str(paper.get("url") or "").strip().lower()
        title_key = re.sub(
            r"\s+",
            " ",
            str(paper.get("title") or "").strip().lower(),
        )
        if not url_key and not title_key:
            continue
        if url_key and url_key in seen_urls:
            continue
        if title_key and title_key in seen_titles:
            continue
        if url_key:
            seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        unique.append(paper)
    return unique


def _prioritize_graph_rec_documents(
    *,
    query: str,
    ranked_pool: list[dict],
    vector_ranked: list[dict],
    top_k: int,
) -> list[dict]:
    if not is_graph_contrastive_recommendation_topic(query, ranked_pool):
        return vector_ranked[:top_k]

    role_ranked = sorted(
        assign_paper_roles(ranked_pool, query),
        key=lambda paper: (
            int(paper.get("priority") or 999),
            -float(paper.get("relevance_score") or 0.0),
        ),
    )

    desired_identities = {
        "contrastive_self_supervised_recommendation_survey",
        "gnn_recommendation_survey",
        "original_or_core_lightgcl",
        "core_xsimgcl",
        "application_or_reproduction_lightgcl",
        "invariant_rationale_gcl",
    }
    selected = [
        paper
        for paper in role_ranked
        if str(paper.get("paper_identity") or "") in desired_identities
    ]

    merged = _dedupe_papers([*selected, *role_ranked, *vector_ranked])
    return merged[:top_k]


def _mock_documents(query: str, max_results: int) -> list[dict]:
    """Return deterministic sample papers for local mock-mode demos."""
    docs = [
        {
            "title": "Agentic RAG: A Survey and Taxonomy",
            "authors": ["Example Author"],
            "abstract": (
                "This survey reviews agentic retrieval augmented generation systems, "
                "including taxonomy, retrieval planning, evidence grading, citation "
                "grounding, and open research gaps."
            ),
            "url": "https://arxiv.org/abs/2401.00001",
            "source": "arxiv",
            "published": "2024-01-01",
            "relevance_score": 0.91,
        },
        {
            "title": "Planning and Reasoning for Iterative Retrieval-Augmented Generation",
            "authors": ["Example Author"],
            "abstract": (
                "This work studies planning, reasoning, query rewriting, and iterative "
                "retrieval loops for grounded generation in LLM agents."
            ),
            "url": "https://arxiv.org/abs/2402.00002",
            "source": "arxiv",
            "published": "2024-02-01",
            "relevance_score": 0.88,
        },
        {
            "title": "Hierarchical Multi-Agent RAG for Scientific Discovery",
            "authors": ["Example Author"],
            "abstract": (
                "This study explores planner, researcher, critic, and writer agents "
                "for collaborative retrieval, grading, and scientific literature analysis."
            ),
            "url": "https://arxiv.org/abs/2403.00003",
            "source": "arxiv",
            "published": "2024-03-01",
            "relevance_score": 0.84,
        },
        {
            "title": "Benchmarking Citation Accuracy and Hallucination in RAG Agents",
            "authors": ["Example Author"],
            "abstract": (
                "This benchmark evaluates citation accuracy, hallucination checking, "
                "retrieval recall, and answer faithfulness for agentic RAG pipelines."
            ),
            "url": "https://arxiv.org/abs/2404.00004",
            "source": "arxiv",
            "published": "2024-04-01",
            "relevance_score": 0.81,
        },
        {
            "title": "Multimodal Agentic RAG with Vision-Language Evidence Alignment",
            "authors": ["Example Author"],
            "abstract": (
                "This paper studies multimodal RAG with image evidence, MLLM reasoning, "
                "and claim-level alignment between visual regions and citations."
            ),
            "url": "https://arxiv.org/abs/2405.00005",
            "source": "arxiv",
            "published": "2024-05-01",
            "relevance_score": 0.78,
        },
        {
            "title": "Domain-Specific Agentic RAG for Medical Question Answering",
            "authors": ["Example Author"],
            "abstract": (
                "This work adapts agentic RAG to medical question answering with domain "
                "ontology, terminology normalization, evidence verification, and clinical citations."
            ),
            "url": "https://arxiv.org/abs/2406.00006",
            "source": "arxiv",
            "published": "2024-06-01",
            "relevance_score": 0.76,
        },
    ]
    return docs[: max(1, min(max_results, len(docs)))]


def _run_async(coro):
    """Run an async coroutine from sync context, handling nested event loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # If there's already a running loop (e.g., called from async FastAPI context),
    # use a new thread to avoid "cannot run nested event loop" error
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


def retrieve_papers(state: AgentState) -> AgentState:
    """Search arXiv/PubMed, embed results in ChromaDB, and retrieve top-k."""
    start = time.perf_counter()

    query = state["query"]
    search_queries = _expanded_search_queries(query)
    search_query = " ".join(_unique_terms(" ".join(search_queries))) or query
    sources = state.get("sources", ["arxiv", "pubmed", "openalex"])
    max_results = state.get("max_results", settings.max_papers)
    year_range = _extract_year_range(query)
    fetch_limit = min(max_results * 4, 99)

    if settings.llm_provider.strip().lower() == "mock":
        documents = _mock_documents(query, max_results)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        steps = list(state.get("steps", []))
        steps.append({
            "node": "retriever",
            "status": "completed",
            "detail": f"Mock retrieval returned {len(documents)} papers for: {query[:60]}",
            "duration_ms": elapsed_ms,
        })
        return {**state, "documents": documents, "steps": steps}

    # --- 1. Fetch papers from external APIs ---
    from langchain_core.documents import Document

    from app.services.paper_fetcher import PaperFetcher
    from app.services.vector_store import VectorStoreService

    fetcher = PaperFetcher()
    raw_papers = _run_async(
        _search_expanded_queries(
            fetcher,
            queries=search_queries,
            sources=sources,
            max_results=fetch_limit,
            year_range=year_range,
        )
    )
    logger.info("PaperFetcher returned %d papers", len(raw_papers))

    if not raw_papers:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        steps = list(state.get("steps", []))
        steps.append({
            "node": "retriever",
            "status": "completed",
            "detail": "No papers found from external APIs",
            "duration_ms": elapsed_ms,
        })
        return {**state, "documents": [], "steps": steps}

    # --- 2. Convert PaperResult objects to dicts for downstream nodes ---
    paper_dicts = _dedupe_papers([p.model_dump() for p in raw_papers])
    paper_dicts = _filter_by_year_range(paper_dicts, year_range)
    paper_dicts = _rank_papers_for_query(paper_dicts, search_query)

    if year_range and not paper_dicts:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        steps = list(state.get("steps", []))
        steps.append({
            "node": "retriever",
            "status": "completed",
            "detail": (
                f"No papers found within {year_range[0]}-{year_range[1]} "
                "after filtering external results"
            ),
            "duration_ms": elapsed_ms,
        })
        return {**state, "documents": [], "steps": steps}

    # --- 3. Convert to LangChain Documents and index in ChromaDB ---
    vector_store = VectorStoreService()
    vector_store.clear()

    lc_docs = [
        Document(
            page_content=paper["abstract"],
            metadata={
                "title": paper["title"],
                "authors": ", ".join(paper["authors"]),
                "url": paper["url"],
                "source": paper["source"],
                "published": paper.get("published") or "",
                "relevance_score": paper.get("relevance_score") or 0.0,
            },
        )
        for paper in paper_dicts
        if paper.get("abstract")
    ]

    if not lc_docs:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        steps = list(state.get("steps", []))
        steps.append({
            "node": "retriever",
            "status": "completed",
            "detail": "No papers found from external APIs",
            "duration_ms": elapsed_ms,
        })
        return {**state, "documents": [], "steps": steps}

    vector_store.add_documents(lc_docs)

    # --- 4. Retrieve top-k most relevant via similarity search + lexical rerank ---
    top_k = max(1, min(max_results, 99))
    candidate_k = min(len(lc_docs), max(top_k, min(top_k * 3, 99)))
    retrieved = vector_store.search(search_query, k=candidate_k)

    documents = []
    for doc in retrieved:
        meta = doc.metadata
        documents.append({
            "title": meta.get("title", ""),
            "authors": [a.strip() for a in meta.get("authors", "").split(",") if a.strip()],
            "abstract": doc.page_content,
            "url": meta.get("url", ""),
            "source": meta.get("source", ""),
            "published": meta.get("published"),
            "relevance_score": meta.get("relevance_score", 0.0),
        })
    vector_ranked_documents = _rank_papers_for_query(documents, search_query)
    documents = _prioritize_graph_rec_documents(
        query=query,
        ranked_pool=paper_dicts,
        vector_ranked=vector_ranked_documents,
        top_k=top_k,
    )

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info("Retrieved %d papers from vector store in %dms", len(documents), elapsed_ms)

    steps = list(state.get("steps", []))
    steps.append({
        "node": "retriever",
        "status": "completed",
        "detail": (
            f"扩展为 {len(search_queries)} 个检索查询，抓取 {len(raw_papers)} 篇论文，"
            f"筛选后保留 {len(paper_dicts)} 篇，最终呈现 {len(documents)} 篇"
        ),
        "duration_ms": elapsed_ms,
    })

    return {**state, "documents": documents, "steps": steps}
