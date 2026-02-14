import sys
import os
import pytest
from unittest.mock import MagicMock

# Add backend to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import SearchResults


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore with all required methods."""
    store = MagicMock()
    store.search = MagicMock(return_value=SearchResults(
        documents=[], metadata=[], distances=[]
    ))
    store.get_lesson_link = MagicMock(return_value=None)
    store.get_course_outline = MagicMock(return_value=None)
    return store


def sample_search_results(variant="success"):
    """Factory returning SearchResults in different states.

    Args:
        variant: "success", "empty", or "error"
    """
    if variant == "success":
        return SearchResults(
            documents=["Chunk about AI basics", "Chunk about neural networks"],
            metadata=[
                {"course_title": "Intro to AI", "lesson_number": 1, "chunk_index": 0},
                {"course_title": "Intro to AI", "lesson_number": 2, "chunk_index": 1},
            ],
            distances=[0.3, 0.5],
        )
    elif variant == "empty":
        return SearchResults(documents=[], metadata=[], distances=[])
    elif variant == "error":
        return SearchResults(
            documents=[], metadata=[], distances=[],
            error="Search error: n_results must be a positive integer"
        )
    raise ValueError(f"Unknown variant: {variant}")


def mock_tool_use_response(tool_name="search_course_content", tool_input=None, tool_use_id="tool_123"):
    """Factory for Anthropic API response with tool_use stop_reason."""
    if tool_input is None:
        tool_input = {"query": "AI basics"}

    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = tool_name
    tool_use_block.input = tool_input
    tool_use_block.id = tool_use_id

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [tool_use_block]
    return response


def mock_text_response(text="Here is the answer about AI."):
    """Factory for Anthropic API response with end_turn stop_reason."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [text_block]
    return response
