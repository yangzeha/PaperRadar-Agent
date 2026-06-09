from typing import Any, Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """User search request."""

    query: str = Field(..., min_length=1, max_length=500)
    sources: list[str] = Field(default=["arxiv", "pubmed", "openalex"])
    max_results: int = Field(default=10, ge=1, le=99)
    session_id: str | None = None


class PaperResult(BaseModel):
    """A single paper from arXiv or PubMed."""

    title: str
    authors: list[str]
    abstract: str
    url: str
    source: str  # "arxiv", "pubmed", "openalex", "ieee", or fallback
    published: str | None = None
    relevance_score: float | None = None
    route: str | None = None
    role: str | None = None
    role_label: str | None = None
    priority: int | None = None
    reason_tags: list[str] = Field(default_factory=list)
    is_core: bool | None = None
    is_background: bool | None = None
    paper_identity: str | None = None


class AgentStep(BaseModel):
    """A single step in the agent execution trace."""

    node: str
    status: str  # "running", "completed", "skipped"
    detail: str = ""
    duration_ms: int | None = None


class Citation(BaseModel):
    """Inline citation linking answer text to a source paper."""

    index: int
    title: str
    url: str


class SearchResponse(BaseModel):
    """Full response from the agent pipeline."""

    query: str
    answer: str
    citations: list[Citation] = []
    papers: list[PaperResult] = []
    steps: list[AgentStep] = []
    rewrite_count: int = 0
    classification: str | None = None
    session_id: str | None = None
    report_template_version: str | None = None


class TranslationRequest(BaseModel):
    """Translate generated answer text while preserving Markdown structure."""

    text: str = Field(..., min_length=1)
    target_language: Literal["zh", "en"] = "en"


class TranslationResponse(BaseModel):
    """Translated answer text."""

    text: str
    target_language: Literal["zh", "en"]


class TopicMemoryRequest(BaseModel):
    """Topic memory update request."""

    text: str | None = Field(default=None, max_length=500)
    topics: list[str] = Field(default_factory=list)


class SavedPaperRequest(BaseModel):
    """Saved paper request for the to-read list."""

    title: str = Field(..., min_length=1, max_length=500)
    url: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    source: str = "manual"
    topic: str = ""
    status: str = "to_read"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatSessionCreate(BaseModel):
    """Create a new chat session."""

    title: str | None = Field(default=None, max_length=80)
