import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults
from conftest import sample_search_results


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute()"""

    def test_execute_successful_search(self, mock_vector_store):
        """Returns formatted results with [Course - Lesson N] headers, populates last_sources."""
        mock_vector_store.search.return_value = sample_search_results("success")
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="AI basics")

        assert "[Intro to AI - Lesson 1]" in result
        assert "Chunk about AI basics" in result
        assert "[Intro to AI - Lesson 2]" in result
        assert "Chunk about neural networks" in result
        assert len(tool.last_sources) == 2

    def test_execute_empty_results(self, mock_vector_store):
        """Returns 'No relevant content found' message."""
        mock_vector_store.search.return_value = sample_search_results("empty")
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_error_results(self, mock_vector_store):
        """Returns error string directly when SearchResults.error is set."""
        mock_vector_store.search.return_value = sample_search_results("error")
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="anything")

        assert "Search error:" in result

    def test_execute_with_course_filter(self, mock_vector_store):
        """Passes course_name to store.search()."""
        mock_vector_store.search.return_value = sample_search_results("empty")
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="AI", course_name="Intro to AI")

        mock_vector_store.search.assert_called_once_with(
            query="AI", course_name="Intro to AI", lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Passes lesson_number to store.search()."""
        mock_vector_store.search.return_value = sample_search_results("empty")
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="AI", lesson_number=3)

        mock_vector_store.search.assert_called_once_with(
            query="AI", course_name=None, lesson_number=3
        )

    def test_execute_with_both_filters(self, mock_vector_store):
        """Passes both filters to store.search()."""
        mock_vector_store.search.return_value = sample_search_results("empty")
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="AI", course_name="Intro to AI", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="AI", course_name="Intro to AI", lesson_number=2
        )

    def test_sources_include_lesson_links(self, mock_vector_store):
        """last_sources entries have text and link from get_lesson_link()."""
        mock_vector_store.search.return_value = sample_search_results("success")
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="AI basics")

        assert len(tool.last_sources) == 2
        for source in tool.last_sources:
            assert "text" in source
            assert "link" in source
            assert source["link"] == "https://example.com/lesson"

    def test_tool_definition_schema(self, mock_vector_store):
        """get_tool_definition() returns correct name/schema."""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "input_schema" in definition
        assert "query" in definition["input_schema"]["properties"]
        assert "query" in definition["input_schema"]["required"]


class TestToolManager:
    """Tests for ToolManager dispatch and source management."""

    def test_tool_manager_dispatches_correctly(self, mock_vector_store):
        """ToolManager.execute_tool('search_course_content', ...) calls CourseSearchTool.execute()."""
        mock_vector_store.search.return_value = sample_search_results("empty")
        tool = CourseSearchTool(mock_vector_store)
        manager = ToolManager()
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test query")

        mock_vector_store.search.assert_called_once()
        assert "No relevant content found" in result

    def test_tool_manager_unknown_tool(self):
        """Returns error string for unregistered tool name."""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result

    def test_tool_manager_source_reset(self, mock_vector_store):
        """reset_sources() clears last_sources."""
        mock_vector_store.search.return_value = sample_search_results("success")
        mock_vector_store.get_lesson_link.return_value = "https://example.com"
        tool = CourseSearchTool(mock_vector_store)
        manager = ToolManager()
        manager.register_tool(tool)

        manager.execute_tool("search_course_content", query="AI")
        assert len(manager.get_last_sources()) > 0

        manager.reset_sources()
        assert len(manager.get_last_sources()) == 0
