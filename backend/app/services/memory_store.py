"""Lightweight JSON memory store for PaperRadar."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

MEMORY_DIR = Path(__file__).resolve().parents[2] / "data" / "memory"
TOPICS_FILE = MEMORY_DIR / "user_topics.json"
SAVED_PAPERS_FILE = MEMORY_DIR / "saved_papers.json"
HISTORY_FILE = MEMORY_DIR / "reading_history.json"
CHAT_SESSIONS_FILE = MEMORY_DIR / "chat_sessions.json"
CURRENT_CHAT_UI_VERSION = "paper-radar-card-v3"

RECENT_MESSAGE_LIMIT = 10
SUMMARY_CHAR_LIMIT = 3200
IMPORTANT_NOTE_LIMIT = 40

LEGACY_CHAT_MARKERS = [
    "mock/fallback 模式",
    "当前 mock",
    "当前 fallback",
    "论文数量一致性检查器",
    "年份过滤核查器",
    "摘要证据表",
    "先按标题和摘要识别与用户问题最接近的论文",
    "第 1-2 天：先读第 1-3 篇",
    "Agentic RAG 方向论文雷达",
    "Agentic RAG 研究的是",
    "LangGraph Agentic RAG",
    "Graph Contrastive Learning / Optimizing Sparse Data",
    "LLM Agent 长期记忆",
    "Multi-Agent Collaboration",
    "RAG Hallucination Evaluation",
    "????",
    "ï¼",
    "æ\u0096",
    "â\u0080",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid4().hex


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return _write_json(path, default.copy())
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return _write_json(path, default.copy())
    if isinstance(data, dict):
        return data
    return _write_json(path, default.copy())


def _write_json(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return data


def _unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            result.append(normalized)
            seen.add(key)
    return result


def _truncate(text: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 1].rstrip()}…"


def _session_title_from_query(query: str) -> str:
    lowered = query.lower()
    if (
        ("图对比" in query or "graph contrastive" in lowered or "lightgcl" in lowered)
        and ("推荐" in query or "recommender" in lowered or "recommendation" in lowered)
    ):
        return "图对比学习推荐系统"
    return _truncate(query, 32) or "新聊天"


def _safe_response(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": response.get("query", ""),
        "answer": response.get("answer", ""),
        "citations": response.get("citations", []),
        "papers": response.get("papers", []),
        "steps": response.get("steps", []),
        "rewrite_count": response.get("rewrite_count", 0),
        "classification": response.get("classification"),
        "session_id": response.get("session_id"),
        "report_template_version": response.get("report_template_version"),
    }


def _is_legacy_chat_session(session: dict[str, Any]) -> bool:
    message_text = "\n".join(
        str(message.get("content", ""))
        for message in list(session.get("messages", []))
    )
    response_text = "\n".join(
        str((message.get("response") or {}).get("answer", ""))
        for message in list(session.get("messages", []))
        if isinstance(message.get("response"), dict)
    )
    combined = "\n".join(
        [
            str(session.get("title", "")),
            str((session.get("memory") or {}).get("summary", "")),
            message_text,
            response_text,
        ]
    )
    return any(marker in combined for marker in LEGACY_CHAT_MARKERS)


def _extract_important_notes(text: str) -> list[str]:
    keywords = [
        "任务",
        "目标",
        "要求",
        "框架",
        "总分",
        "分支",
        "计划",
        "步骤",
        "结论",
        "记忆",
        "长期",
        "短期",
        "不要",
        "必须",
        "需要",
        "bug",
        "错误",
        "文件",
        "接口",
        "节点",
        "论文",
        "研究空白",
    ]
    notes: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip(" \t-*#>：:")
        if not line:
            continue
        is_structured = raw_line.lstrip().startswith(("#", "-", "*")) or re.match(
            r"^\d+[.、]", raw_line.strip()
        )
        has_keyword = any(keyword.lower() in line.lower() for keyword in keywords)
        if is_structured or has_keyword:
            notes.append(_truncate(line, 180))
    if not notes and text.strip():
        notes.append(_truncate(text, 180))
    return _unique_keep_order(notes)


def _summarize_messages(messages: list[dict[str, Any]]) -> tuple[str, list[str]]:
    lines: list[str] = []
    notes: list[str] = []
    for message in messages:
        role = "用户" if message.get("role") == "user" else "助手"
        created_at = str(message.get("created_at", ""))[:10]
        content = str(message.get("content", ""))
        lines.append(f"- {created_at} {role}: {_truncate(content, 220)}")
        notes.extend(_extract_important_notes(content))
    return "\n".join(lines), _unique_keep_order(notes)


def _compact_summary(summary: str) -> str:
    if len(summary) <= SUMMARY_CHAR_LIMIT:
        return summary

    lines = [line for line in summary.splitlines() if line.strip()]
    important = [
        line
        for line in lines
        if any(
            keyword in line
            for keyword in ["任务", "目标", "要求", "框架", "总分", "计划", "记忆", "bug", "错误"]
        )
    ]
    compacted_lines = _unique_keep_order([*important, *lines[-20:]])
    compacted = "\n".join(compacted_lines)
    if len(compacted) <= SUMMARY_CHAR_LIMIT:
        return compacted
    return f"{compacted[: SUMMARY_CHAR_LIMIT - 1].rstrip()}…"


def _compress_session_if_needed(session: dict[str, Any]) -> dict[str, Any]:
    messages = list(session.get("messages", []))
    if len(messages) <= RECENT_MESSAGE_LIMIT:
        return session

    older_messages = messages[:-RECENT_MESSAGE_LIMIT]
    recent_messages = messages[-RECENT_MESSAGE_LIMIT:]
    summary_chunk, important_notes = _summarize_messages(older_messages)

    memory = dict(session.get("memory") or {})
    previous_summary = str(memory.get("summary", "")).strip()
    combined_summary = "\n".join(
        part
        for part in [
            previous_summary,
            f"较早对话压缩摘要（{older_messages[0].get('created_at', '')[:10]} - {older_messages[-1].get('created_at', '')[:10]}）：",
            summary_chunk,
        ]
        if part
    )

    existing_notes = list(memory.get("important_notes", []))
    memory.update(
        {
            "summary": _compact_summary(combined_summary),
            "important_notes": _unique_keep_order(
                [*existing_notes, *important_notes]
            )[-IMPORTANT_NOTE_LIMIT:],
            "compressed_message_count": int(memory.get("compressed_message_count", 0))
            + len(older_messages),
            "last_compressed_at": _now_iso(),
        }
    )
    session["memory"] = memory
    session["messages"] = recent_messages
    return session


def extract_topics_from_text(text: str) -> list[str]:
    """Extract rough topic candidates from a user query."""
    cleaned = text.replace("，", ",").replace("、", ",").replace("；", ",")
    chunks = [chunk.strip(" .:：;；") for chunk in cleaned.split(",")]
    topics = [chunk for chunk in chunks if 2 <= len(chunk) <= 80]
    if not topics and text.strip():
        topics = [text.strip()[:80]]
    return _unique_keep_order(topics)


def list_topics() -> dict[str, Any]:
    return _read_json(TOPICS_FILE, {"topics": [], "updated_at": None})


def upsert_topics(topics: list[str]) -> dict[str, Any]:
    current = list_topics()
    merged = _unique_keep_order([*current.get("topics", []), *topics])
    return _write_json(TOPICS_FILE, {"topics": merged, "updated_at": _now_iso()})


def list_saved_papers() -> dict[str, Any]:
    return _read_json(SAVED_PAPERS_FILE, {"papers": []})


def save_paper(paper: dict[str, Any]) -> dict[str, Any]:
    current = list_saved_papers()
    papers = list(current.get("papers", []))
    url = str(paper.get("url", "")).strip()
    title = str(paper.get("title", "")).strip()
    exists = any(
        (url and item.get("url") == url) or (title and item.get("title") == title)
        for item in papers
    )
    record = {**paper, "saved_at": paper.get("saved_at") or _now_iso()}
    if not exists:
        papers.append(record)
    return _write_json(SAVED_PAPERS_FILE, {"papers": papers})


def list_history(limit: int = 50) -> dict[str, Any]:
    current = _read_json(HISTORY_FILE, {"history": []})
    history = list(current.get("history", []))[-limit:]
    return {"history": history}


def add_history(entry: dict[str, Any]) -> dict[str, Any]:
    current = _read_json(HISTORY_FILE, {"history": []})
    history = list(current.get("history", []))
    history.append({**entry, "created_at": entry.get("created_at") or _now_iso()})
    return _write_json(HISTORY_FILE, {"history": history[-200:]})


def get_memory_context(limit: int = 10) -> dict[str, Any]:
    return {
        "topics": list_topics().get("topics", []),
        "saved_papers": list_saved_papers().get("papers", [])[-limit:],
        "history": list_history(limit=limit).get("history", []),
        "chat_sessions": list_chat_sessions(limit=limit).get("sessions", []),
    }


def _read_chat_sessions() -> dict[str, Any]:
    return _read_json(CHAT_SESSIONS_FILE, {"sessions": []})


def _write_chat_sessions(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    sessions = sorted(
        sessions,
        key=lambda session: str(session.get("updated_at", "")),
        reverse=True,
    )
    return _write_json(CHAT_SESSIONS_FILE, {"sessions": sessions[:300]})


def _session_preview(session: dict[str, Any]) -> dict[str, Any]:
    messages = list(session.get("messages", []))
    last_message = messages[-1] if messages else {}
    return {
        "id": session.get("id"),
        "title": session.get("title") or "新聊天",
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
        "message_count": len(messages)
        + int((session.get("memory") or {}).get("compressed_message_count", 0)),
        "compressed_message_count": int(
            (session.get("memory") or {}).get("compressed_message_count", 0)
        ),
        "last_message_preview": _truncate(str(last_message.get("content", "")), 80),
        "memory_preview": _truncate(
            str((session.get("memory") or {}).get("summary", "")),
            120,
        ),
    }


def list_chat_sessions(limit: int = 100) -> dict[str, Any]:
    sessions = list(_read_chat_sessions().get("sessions", []))
    sessions = [session for session in sessions if not _is_legacy_chat_session(session)]
    sessions = sorted(
        sessions,
        key=lambda session: str(session.get("updated_at", "")),
        reverse=True,
    )
    return {"sessions": [_session_preview(session) for session in sessions[:limit]]}


def create_chat_session(title: str | None = None) -> dict[str, Any]:
    now = _now_iso()
    session = {
        "id": _new_id(),
        "title": title.strip() if title and title.strip() else "新聊天",
        "created_at": now,
        "updated_at": now,
        "messages": [],
        "memory": {
            "summary": "",
            "important_notes": [],
            "compressed_message_count": 0,
            "last_compressed_at": None,
        },
    }
    sessions = list(_read_chat_sessions().get("sessions", []))
    sessions.append(session)
    _write_chat_sessions(sessions)
    return session


def get_chat_session(session_id: str) -> dict[str, Any] | None:
    sessions = list(_read_chat_sessions().get("sessions", []))
    for session in sessions:
        if session.get("id") == session_id:
            if _is_legacy_chat_session(session):
                return None
            return session
    return None


def delete_chat_session(session_id: str) -> dict[str, Any]:
    sessions = list(_read_chat_sessions().get("sessions", []))
    remaining = [session for session in sessions if session.get("id") != session_id]
    _write_chat_sessions(remaining)
    return {"deleted": len(sessions) != len(remaining)}


def append_chat_turn(
    *,
    session_id: str | None,
    query: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    sessions = list(_read_chat_sessions().get("sessions", []))
    session = None
    for existing in sessions:
        if session_id and existing.get("id") == session_id:
            session = existing
            break

    if session is None:
        session = create_chat_session(_session_title_from_query(query))
        sessions = list(_read_chat_sessions().get("sessions", []))
        for existing in sessions:
            if existing.get("id") == session["id"]:
                session = existing
                break
    elif not session.get("title") or session.get("title") == "新聊天":
        session["title"] = _session_title_from_query(query)

    now = _now_iso()
    response = _safe_response({**response, "session_id": session["id"]})
    messages = list(session.get("messages", []))
    messages.extend(
        [
            {
                "id": _new_id(),
                "role": "user",
                "content": query,
                "created_at": now,
            },
            {
                "id": _new_id(),
                "role": "assistant",
                "content": response.get("answer", ""),
                "created_at": now,
                "response": response,
            },
        ]
    )
    session["messages"] = messages
    session["updated_at"] = now
    session = _compress_session_if_needed(session)

    for index, existing in enumerate(sessions):
        if existing.get("id") == session["id"]:
            sessions[index] = session
            break
    else:
        sessions.append(session)

    _write_chat_sessions(sessions)
    return session
