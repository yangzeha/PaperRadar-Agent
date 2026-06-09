"""Agent graph nodes package."""

__all__ = [
    "generate_answer",
    "grade_documents",
    "check_hallucination",
    "retrieve_papers",
    "rewrite_query",
    "route_query",
    "synthesize_response",
]


def __getattr__(name: str):
    if name == "generate_answer":
        from app.agents.nodes.generator import generate_answer

        return generate_answer
    if name == "grade_documents":
        from app.agents.nodes.grader import grade_documents

        return grade_documents
    if name == "check_hallucination":
        from app.agents.nodes.hallucination_checker import check_hallucination

        return check_hallucination
    if name == "retrieve_papers":
        from app.agents.nodes.retriever import retrieve_papers

        return retrieve_papers
    if name == "rewrite_query":
        from app.agents.nodes.rewriter import rewrite_query

        return rewrite_query
    if name == "route_query":
        from app.agents.nodes.router import route_query

        return route_query
    if name == "synthesize_response":
        from app.agents.nodes.synthesizer import synthesize_response

        return synthesize_response
    raise AttributeError(name)
