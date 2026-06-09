"""Async paper fetcher for arXiv and PubMed."""

from __future__ import annotations

import asyncio
import logging
import re
from xml.etree import ElementTree

import httpx

from app.config import settings
from app.models.schemas import PaperResult

logger = logging.getLogger(__name__)

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
IEEE_SEARCH_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
arxiv = None


def _is_arxiv_rate_limited(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "http 429" in message
        or "http 503" in message
        or "rate exceeded" in message
        or "page request resulted" in message
        or "export.arxiv.org" in message
    )


def _openalex_abstract(inverted_index: dict[str, list[int]]) -> str:
    if not inverted_index:
        return ""
    positioned_words: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        positioned_words.extend((position, word) for position in positions)
    return " ".join(word for _, word in sorted(positioned_words))


def _safe_error_message(error: BaseException) -> str:
    return re.sub(r"apikey=[^&\\s']+", "apikey=<redacted>", str(error), flags=re.IGNORECASE)


class PaperFetcher:
    """Fetches academic papers from arXiv and PubMed."""

    # ------------------------------------------------------------------
    # arXiv
    # ------------------------------------------------------------------

    async def search_arxiv(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[PaperResult]:
        """Search arXiv using the ``arxiv`` library.

        The library is synchronous, so we delegate to a thread to keep
        the event loop free.
        """
        max_results = max_results or settings.max_papers

        global arxiv
        if arxiv is None:
            try:
                import arxiv as arxiv_module
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "arxiv package is required for real arXiv retrieval. "
                    "Install with: python -m pip install -e \".[rag]\""
                ) from exc
            arxiv = arxiv_module

        def _sync_search() -> list[PaperResult]:
            client = arxiv.Client(page_size=max_results, delay_seconds=1.0, num_retries=0)
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            papers: list[PaperResult] = []
            for result in client.results(search):
                papers.append(
                    PaperResult(
                        title=result.title,
                        authors=[a.name for a in result.authors],
                        abstract=result.summary,
                        url=result.entry_id,
                        source="arxiv",
                        published=(
                            result.published.strftime("%Y-%m-%d")
                            if result.published
                            else None
                        ),
                    )
                )
            return papers

        try:
            papers = await asyncio.to_thread(_sync_search)
        except Exception as exc:
            if _is_arxiv_rate_limited(exc):
                logger.warning("arXiv rate-limited; using OpenAlex fallback: %s", exc)
                return await self.search_openalex_fallback(query, max_results)
            raise
        logger.info("arXiv returned %d papers for '%s'", len(papers), query)
        return papers

    async def search_openalex_fallback(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[PaperResult]:
        """Search OpenAlex when arXiv's public endpoint is rate-limited."""
        return await self.search_openalex(query, max_results, source="arxiv_openalex")

    async def search_openalex(
        self,
        query: str,
        max_results: int | None = None,
        source: str = "openalex",
        year_range: tuple[int, int] | None = None,
    ) -> list[PaperResult]:
        """Search OpenAlex's open scholarly works index."""
        max_results = max_results or settings.max_papers
        filters = ["has_abstract:true"]
        if year_range:
            filters.extend(
                [
                    f"from_publication_date:{year_range[0]}-01-01",
                    f"to_publication_date:{year_range[1]}-12-31",
                ]
            )
        params = {
            "search": query,
            "per-page": max_results,
            "filter": ",".join(filters),
            "select": (
                "title,display_name,authorships,abstract_inverted_index,"
                "publication_date,primary_location,ids,relevance_score"
            ),
        }
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            response = await client.get(OPENALEX_WORKS_URL, params=params)
            response.raise_for_status()
        results = response.json().get("results", [])

        papers: list[PaperResult] = []
        for item in results:
            abstract = _openalex_abstract(item.get("abstract_inverted_index") or {})
            if not abstract:
                continue
            authors = [
                authorship.get("author", {}).get("display_name", "")
                for authorship in item.get("authorships", [])
            ]
            authors = [author for author in authors if author]
            primary_location = item.get("primary_location") or {}
            ids = item.get("ids") or {}
            url = (
                primary_location.get("landing_page_url")
                or ids.get("doi")
                or ids.get("openalex")
                or ""
            )
            papers.append(
                PaperResult(
                    title=item.get("display_name") or item.get("title") or "",
                    authors=authors,
                    abstract=abstract,
                    url=url,
                    source=source,
                    published=item.get("publication_date"),
                    relevance_score=item.get("relevance_score"),
                )
            )

        logger.info("OpenAlex returned %d papers for '%s'", len(papers), query)
        return papers

    async def search_ieee(
        self,
        query: str,
        max_results: int | None = None,
        year_range: tuple[int, int] | None = None,
    ) -> list[PaperResult]:
        """Search IEEE Xplore when an IEEE API key is configured."""
        if not settings.ieee_api_key:
            logger.warning("IEEE source selected but IEEE_API_KEY is not configured")
            return []

        max_results = max_results or settings.max_papers
        params = {
            "apikey": settings.ieee_api_key,
            "querytext": query,
            "max_records": max_results,
            "sort_field": "publication_year",
            "sort_order": "desc",
        }
        if year_range:
            params["start_year"] = year_range[0]
            params["end_year"] = year_range[1]
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            response = await client.get(IEEE_SEARCH_URL, params=params)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"IEEE API request failed with status {exc.response.status_code}"
                ) from exc

        articles = response.json().get("articles", [])
        papers: list[PaperResult] = []
        for article in articles:
            authors_raw = article.get("authors") or {}
            authors = [
                author.get("full_name", "")
                for author in authors_raw.get("authors", [])
                if isinstance(author, dict)
            ]
            authors = [author for author in authors if author]
            abstract = (
                article.get("abstract")
                or article.get("abstract_url")
                or article.get("title", "")
            )
            url = article.get("html_url") or article.get("pdf_url") or article.get("doi") or ""
            published = str(article.get("publication_year") or "") or None
            papers.append(
                PaperResult(
                    title=article.get("title", ""),
                    authors=authors,
                    abstract=abstract,
                    url=url,
                    source="ieee",
                    published=published,
                )
            )

        logger.info("IEEE returned %d papers for '%s'", len(papers), query)
        return papers

    # ------------------------------------------------------------------
    # PubMed
    # ------------------------------------------------------------------

    async def search_pubmed(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[PaperResult]:
        """Search PubMed via the NCBI E-utilities REST API.

        Workflow: esearch (get IDs) -> efetch (get full records as XML).
        """
        max_results = max_results or settings.max_papers

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1 — esearch: get matching PubMed IDs
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
            }
            search_resp = await client.get(
                f"{PUBMED_BASE_URL}esearch.fcgi", params=search_params
            )
            search_resp.raise_for_status()
            id_list: list[str] = (
                search_resp.json()
                .get("esearchresult", {})
                .get("idlist", [])
            )

            if not id_list:
                logger.info("PubMed returned 0 results for '%s'", query)
                return []

            # Step 2 — efetch: retrieve article metadata as XML
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "rettype": "xml",
                "retmode": "xml",
            }
            fetch_resp = await client.get(
                f"{PUBMED_BASE_URL}efetch.fcgi", params=fetch_params
            )
            fetch_resp.raise_for_status()

        papers = self._parse_pubmed_xml(fetch_resp.text)
        logger.info("PubMed returned %d papers for '%s'", len(papers), query)
        return papers

    @staticmethod
    def _parse_pubmed_xml(xml_text: str) -> list[PaperResult]:
        """Parse PubMed efetch XML into PaperResult objects."""
        root = ElementTree.fromstring(xml_text)
        papers: list[PaperResult] = []

        for article_el in root.findall(".//PubmedArticle"):
            medline = article_el.find("MedlineCitation")
            if medline is None:
                continue

            article = medline.find("Article")
            if article is None:
                continue

            # Title
            title_el = article.find("ArticleTitle")
            title = title_el.text if title_el is not None and title_el.text else ""

            # Abstract
            abstract_parts: list[str] = []
            abstract_el = article.find("Abstract")
            if abstract_el is not None:
                for text_el in abstract_el.findall("AbstractText"):
                    if text_el.text:
                        abstract_parts.append(text_el.text)
            abstract = " ".join(abstract_parts)

            # Authors
            authors: list[str] = []
            author_list = article.find("AuthorList")
            if author_list is not None:
                for author_el in author_list.findall("Author"):
                    last = author_el.findtext("LastName", "")
                    fore = author_el.findtext("ForeName", "")
                    name = f"{fore} {last}".strip()
                    if name:
                        authors.append(name)

            # Published date
            pub_date_el = article.find(".//PubDate")
            published: str | None = None
            if pub_date_el is not None:
                year = pub_date_el.findtext("Year", "")
                month = pub_date_el.findtext("Month", "")
                day = pub_date_el.findtext("Day", "")
                date_parts = [p for p in (year, month, day) if p]
                if date_parts:
                    published = "-".join(date_parts)

            # PubMed ID -> URL
            pmid_el = medline.find("PMID")
            pmid = pmid_el.text if pmid_el is not None and pmid_el.text else ""
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

            papers.append(
                PaperResult(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=url,
                    source="pubmed",
                    published=published,
                )
            )

        return papers

    # ------------------------------------------------------------------
    # Unified search dispatcher
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        sources: list[str] | None = None,
        max_results: int | None = None,
        year_range: tuple[int, int] | None = None,
    ) -> list[PaperResult]:
        """Search one or more sources in parallel and merge results.

        Args:
            query: The search query string.
            sources: List of source names (``"arxiv"``, ``"pubmed"``,
                     ``"openalex"``, ``"ieee"``). Defaults to arXiv,
                     PubMed, and OpenAlex.
            max_results: Per-source cap. Falls back to ``settings.max_papers``.
            year_range: Optional publication-year range used by sources that
                        support server-side filtering.
        """
        sources = sources or ["arxiv", "pubmed", "openalex"]
        max_results = max_results or settings.max_papers

        tasks: list[asyncio.Task[list[PaperResult]]] = []

        if "arxiv" in sources:
            tasks.append(
                asyncio.create_task(self.search_arxiv(query, max_results))
            )
        if "pubmed" in sources:
            tasks.append(
                asyncio.create_task(self.search_pubmed(query, max_results))
            )
        if "openalex" in sources:
            tasks.append(
                asyncio.create_task(self.search_openalex(query, max_results, year_range=year_range))
            )
        if "ieee" in sources:
            tasks.append(
                asyncio.create_task(self.search_ieee(query, max_results, year_range=year_range))
            )

        if not tasks:
            logger.warning("No valid sources in %s — returning empty", sources)
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        papers: list[PaperResult] = []
        for result in results:
            if isinstance(result, BaseException):
                message = _safe_error_message(result) or result.__class__.__name__
                logger.error("Paper fetch failed: %s", message)
                continue
            papers.extend(result)

        logger.info(
            "Combined search returned %d papers from %s", len(papers), sources
        )
        return papers
