"""LangGraph Studio compatibility tests."""

from __future__ import annotations


def test_graph_accepts_studio_serialized_chat_message(monkeypatch):
    from app.config import settings
    from app.agents.graph import build_graph

    monkeypatch.setattr(settings, "llm_provider", "mock")

    serialized_human_message = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain_core", "messages", "HumanMessage"],
        "kwargs": {
            "content": [
                {
                    "type": "text",
                    "text": "hello",
                }
            ],
        },
    }

    final_state = build_graph().invoke({"messages": [serialized_human_message]})

    assert final_state["query"] == "hello"
    assert final_state["classification"] == "general"
    assert final_state["answer"]
    assert final_state["messages"][-1].content
