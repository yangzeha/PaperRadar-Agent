import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage

from app.config import settings
from app.models.schemas import (
    ChatSessionCreate,
    SavedPaperRequest,
    SearchRequest,
    SearchResponse,
    TopicMemoryRequest,
    TranslationRequest,
    TranslationResponse,
)
from app.services.memory_store import (
    append_chat_turn,
    create_chat_session,
    delete_chat_session,
    extract_topics_from_text,
    get_chat_session,
    list_history,
    list_chat_sessions,
    list_saved_papers,
    list_topics,
    save_paper,
    upsert_topics,
)
from app.services.llm_provider import extract_text, invoke_with_retry, provider_status

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ScholarAgent API starting up")
    yield
    logger.info("ScholarAgent API shutting down")


app = FastAPI(
    title="PaperRadar-Agent API",
    description="Chinese paper radar and topic-tracking agent with LangGraph",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "paper-radar-agent"}


@app.get("/api/provider")
async def get_provider_status():
    """Return current LLM provider metadata for the frontend."""
    return provider_status()


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Run the agent graph and return the full search response."""
    from app.agents.graph import run_search

    start = time.time()
    try:
        response = await run_search(
            query=request.query,
            sources=request.sources,
            max_results=request.max_results,
            session_id=request.session_id,
        )
        session = append_chat_turn(
            session_id=request.session_id,
            query=request.query,
            response=response.model_dump(),
        )
        response.session_id = session.get("id")
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            logger.warning("Rate limited during search: %s", err_str[:200])
            return JSONResponse(
                status_code=429,
                content={"detail": "Gemini API rate limit reached. Please wait a moment and try again."},
            )
        logger.error("Search failed: %s", err_str[:300])
        return JSONResponse(
            status_code=500,
            content={"detail": f"Search failed: {err_str[:200]}"},
        )
    elapsed_ms = int((time.time() - start) * 1000)
    logger.info(f"Search completed in {elapsed_ms}ms for query: {request.query[:80]}")
    return response


@app.post("/api/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """Translate generated report text while preserving Markdown and citations."""
    target = "English" if request.target_language == "en" else "Chinese"
    prompt = (
        f"Translate the following Markdown report into {target}. "
        "Preserve all Markdown headings, lists, links, and citation markers like [1]. "
        "Do not add new facts, remove facts, or change citation numbers.\n\n"
        f"Report:\n{request.text}\n\nTranslation:"
    )
    try:
        response = invoke_with_retry([HumanMessage(content=prompt)])
        return TranslationResponse(
            text=extract_text(response).strip(),
            target_language=request.target_language,
        )
    except Exception as e:
        err_str = str(e)
        logger.error("Translation failed: %s", err_str[:300])
        return JSONResponse(
            status_code=500,
            content={"detail": f"Translation failed: {err_str[:200]}"},
        )


@app.get("/api/memory/topics")
async def get_memory_topics():
    """Return long-term user topic memory."""
    return list_topics()


@app.post("/api/memory/topics")
async def post_memory_topics(request: TopicMemoryRequest):
    """Add topic memory from explicit topics or a free-form text query."""
    topics = list(request.topics)
    if request.text:
        topics.extend(extract_topics_from_text(request.text))
    return upsert_topics(topics)


@app.get("/api/memory/saved-papers")
async def get_saved_papers():
    """Return the saved paper/to-read list."""
    return list_saved_papers()


@app.post("/api/memory/saved-papers")
async def post_saved_paper(request: SavedPaperRequest):
    """Save a paper to the to-read list."""
    return save_paper(request.model_dump())


@app.get("/api/memory/history")
async def get_reading_history():
    """Return recent PaperRadar retrieval and reading history."""
    return list_history()


@app.get("/api/chat/sessions")
async def get_chat_sessions():
    """Return chat sessions for the left-side history list."""
    return list_chat_sessions()


@app.post("/api/chat/sessions")
async def post_chat_session(request: ChatSessionCreate):
    """Create a new chat session."""
    return create_chat_session(request.title)


@app.get("/api/chat/sessions/{session_id}")
async def get_chat_session_detail(session_id: str):
    """Return one chat session, including compressed memory and recent messages."""
    session = get_chat_session(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"detail": "Chat session not found"})
    return session


@app.delete("/api/chat/sessions/{session_id}")
async def delete_chat_session_endpoint(session_id: str):
    """Delete one chat session from long-term chat memory."""
    return delete_chat_session(session_id)


@app.websocket("/ws/search")
async def websocket_search(websocket: WebSocket):
    """Stream agent steps in real-time via WebSocket."""
    await websocket.accept()
    try:
        data = await websocket.receive_text()
        request = SearchRequest.model_validate_json(data)

        from app.agents.graph import build_graph
        from app.agents.state import AgentState

        graph = build_graph()
        initial_state: AgentState = {
            "query": request.query,
            "original_query": request.query,
            "documents": [],
            "graded_documents": [],
            "background_documents": [],
            "rewrite_count": 0,
            "answer": "",
            "hallucination_score": 0.0,
            "steps": [],
            "citations": [],
            "memory_context": {},
            "sources": request.sources,
            "max_results": request.max_results,
        }

        prev_steps_count = 0
        final_state = dict(initial_state)
        async for state_update in graph.astream(initial_state):
            for node_name, node_state in state_update.items():
                final_state.update(node_state)
                steps = node_state.get("steps", [])
                if len(steps) > prev_steps_count:
                    new_steps = steps[prev_steps_count:]
                    for step in new_steps:
                        await websocket.send_text(
                            json.dumps({"type": "step", "data": step})
                        )
                    prev_steps_count = len(steps)

        from app.agents.document_selection import select_output_documents
        from app.agents.graph import REPORT_TEMPLATE_VERSION
        from app.models.schemas import AgentStep, Citation, PaperResult

        response = SearchResponse(
            query=request.query,
            answer=final_state.get("answer", ""),
            citations=[Citation(**c) for c in final_state.get("citations", [])],
            papers=[
                PaperResult(**p)
                for p in select_output_documents(final_state)
            ],
            steps=[AgentStep(**s) for s in final_state.get("steps", [])],
            rewrite_count=final_state.get("rewrite_count", 0),
            classification=final_state.get("classification"),
            report_template_version=REPORT_TEMPLATE_VERSION,
        )
        await websocket.send_text(
            json.dumps({"type": "result", "data": response.model_dump()})
        )
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "data": str(e)})
            )
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
