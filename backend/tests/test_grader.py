"""Tests for the Grader node — relevance filtering."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage


class TestGradeDocuments:
    """Test that grade_documents filters documents by relevance."""

    @patch("app.agents.nodes.grader.invoke_with_retry")
    def test_all_relevant(self, mock_invoke, sample_state, sample_papers):
        """When the LLM says 'yes' for all docs, all should be kept."""
        from app.agents.nodes.grader import grade_documents

        mock_invoke.return_value = AIMessage(content="1: yes\n2: yes\n3: yes")

        state = {**sample_state, "documents": sample_papers}
        result = grade_documents(state)

        assert len(result["graded_documents"]) == len(sample_papers)
        assert [doc["title"] for doc in result["graded_documents"]] == [
            doc["title"] for doc in sample_papers
        ]
        assert all(doc["relevance_tier"] == "core" for doc in result["graded_documents"])
        assert all(doc["relevance_score"] >= 0.55 for doc in result["graded_documents"])

    @patch("app.agents.nodes.grader.invoke_with_retry")
    def test_all_irrelevant(self, mock_invoke, sample_state, sample_papers):
        """When the LLM says 'no' for all docs, none should be kept."""
        from app.agents.nodes.grader import grade_documents

        mock_invoke.return_value = AIMessage(content="1: no\n2: no\n3: no")

        state = {**sample_state, "documents": sample_papers}
        result = grade_documents(state)

        assert len(result["graded_documents"]) == 0
        assert len(result["background_documents"]) == 0

    @patch("app.agents.nodes.grader.invoke_with_retry")
    def test_mixed_relevance(self, mock_invoke, sample_state, sample_papers):
        """When the LLM alternates yes/no, only relevant docs are kept."""
        from app.agents.nodes.grader import grade_documents

        mock_invoke.return_value = AIMessage(content="1: yes\n2: no\n3: yes")

        state = {**sample_state, "documents": sample_papers}
        result = grade_documents(state)

        assert len(result["graded_documents"]) == 2
        assert result["graded_documents"][0]["title"] == sample_papers[0]["title"]
        assert result["graded_documents"][1]["title"] == sample_papers[2]["title"]

    @patch("app.agents.nodes.grader.invoke_with_retry")
    def test_fuzzy_yes_accepted(self, mock_invoke, sample_state, sample_papers):
        """Variations like '1: Yes, it is relevant' should still count as relevant."""
        from app.agents.nodes.grader import grade_documents

        mock_invoke.return_value = AIMessage(content="1: yes")

        state = {**sample_state, "documents": sample_papers[:1]}
        result = grade_documents(state)

        assert len(result["graded_documents"]) == 1

    @patch("app.agents.nodes.grader.invoke_with_retry")
    def test_score_tiers_core_background_exclude(self, mock_invoke, sample_state, sample_papers):
        """Numeric scores should split papers into core/background/exclude."""
        from app.agents.nodes.grader import grade_documents

        mock_invoke.return_value = AIMessage(
            content=(
                "1: score=0.8 tier=core reason=direct\n"
                "2: score=0.42 tier=background reason=nearby\n"
                "3: score=0.1 tier=exclude reason=unrelated"
            )
        )

        state = {**sample_state, "documents": sample_papers}
        result = grade_documents(state)

        assert [doc["title"] for doc in result["graded_documents"]] == [
            sample_papers[0]["title"]
        ]
        assert [doc["title"] for doc in result["background_documents"]] == [
            sample_papers[1]["title"]
        ]

    @patch("app.agents.nodes.grader.invoke_with_retry")
    def test_empty_documents(self, mock_invoke, sample_state):
        """Grading with no documents should return an empty list."""
        from app.agents.nodes.grader import grade_documents

        state = {**sample_state, "documents": []}
        result = grade_documents(state)

        assert result["graded_documents"] == []
        # LLM should not be called at all
        mock_invoke.assert_not_called()

    @patch("app.agents.nodes.grader.invoke_with_retry")
    def test_step_records_counts(self, mock_invoke, sample_state, sample_papers):
        """The grader step should record how many docs were relevant."""
        from app.agents.nodes.grader import grade_documents

        mock_invoke.return_value = AIMessage(content="1: yes\n2: no\n3: yes")

        state = {**sample_state, "documents": sample_papers}
        result = grade_documents(state)

        step = result["steps"][-1]
        assert step["node"] == "grader"
        assert step["status"] == "completed"
        assert "2/3" in step["detail"]

    @patch("app.agents.nodes.grader.invoke_with_retry")
    def test_preserves_existing_steps(self, mock_invoke, sample_state, sample_papers):
        """Grader should append to existing steps, not overwrite."""
        from app.agents.nodes.grader import grade_documents

        mock_invoke.return_value = AIMessage(content="1: yes\n2: yes\n3: yes")

        existing_step = {"node": "retriever", "status": "completed", "detail": "test"}
        state = {**sample_state, "documents": sample_papers, "steps": [existing_step]}
        result = grade_documents(state)

        assert len(result["steps"]) == 2
        assert result["steps"][0] == existing_step
        assert result["steps"][1]["node"] == "grader"
