"""PaperRadar-Agent LangGraph pipeline package."""

from app.agents.state import AgentState

__all__ = ["AgentState", "build_graph", "run_search"]


def __getattr__(name: str):
    if name in {"build_graph", "run_search"}:
        from app.agents.graph import build_graph, run_search

        return {"build_graph": build_graph, "run_search": run_search}[name]
    raise AttributeError(name)
