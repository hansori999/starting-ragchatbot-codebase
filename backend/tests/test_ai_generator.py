import sys
import os
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from ai_generator import AIGenerator
from conftest import mock_tool_use_response, mock_text_response


@pytest.fixture
def mock_anthropic_client():
    """Mock anthropic.Anthropic so client.messages.create() is controllable."""
    with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        yield mock_client


@pytest.fixture
def generator(mock_anthropic_client):
    """AIGenerator with mocked Anthropic client."""
    return AIGenerator(api_key="test-key", model="claude-test")


@pytest.fixture
def mock_tool_manager():
    """Mock ToolManager."""
    tm = MagicMock()
    tm.execute_tool.return_value = "Tool result: AI basics explained"
    return tm


class TestAIGeneratorDirectResponse:
    """Tests for direct (non-tool-use) responses."""

    def test_direct_response_no_tool_use(self, generator, mock_anthropic_client):
        """When stop_reason='end_turn', returns content[0].text without calling tools."""
        mock_anthropic_client.messages.create.return_value = mock_text_response("Direct answer")

        result = generator.generate_response(query="What is 2+2?")

        assert result == "Direct answer"
        mock_anthropic_client.messages.create.assert_called_once()


class TestAIGeneratorToolUse:
    """Tests for tool-calling flow."""

    def test_tool_use_triggers_execution(self, generator, mock_anthropic_client, mock_tool_manager):
        """When stop_reason='tool_use', calls tool_manager.execute_tool() with correct name/params."""
        tool_response = mock_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "AI basics"},
            tool_use_id="tool_abc"
        )
        final_response = mock_text_response("Here is info about AI basics.")
        mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]

        result = generator.generate_response(
            query="Tell me about AI",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="AI basics"
        )
        assert result == "Here is info about AI basics."

    def test_tool_result_sent_back_to_claude(self, generator, mock_anthropic_client, mock_tool_manager):
        """Follow-up API call includes tool_result message with correct tool_use_id and content."""
        tool_response = mock_tool_use_response(tool_use_id="tool_xyz")
        final_response = mock_text_response("Final answer")
        mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]

        generator.generate_response(
            query="Search query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Check the second call's messages argument
        second_call = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call.kwargs.get("messages") or second_call[1].get("messages")
        # Last message should contain tool_result
        tool_result_msg = messages[-1]
        assert tool_result_msg["role"] == "user"
        tool_result_content = tool_result_msg["content"][0]
        assert tool_result_content["type"] == "tool_result"
        assert tool_result_content["tool_use_id"] == "tool_xyz"
        assert tool_result_content["content"] == "Tool result: AI basics explained"

    def test_follow_up_call_excludes_tools(self, generator, mock_anthropic_client, mock_tool_manager):
        """Second messages.create() call does NOT include tools param (prevents infinite loop)."""
        tool_response = mock_tool_use_response()
        final_response = mock_text_response("Done")
        mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]

        generator.generate_response(
            query="Query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1].kwargs
        assert "tools" not in second_call_kwargs

    def test_tool_error_string_forwarded(self, generator, mock_anthropic_client, mock_tool_manager):
        """When tool returns error string, it's passed to Claude as tool_result content."""
        mock_tool_manager.execute_tool.return_value = "Search error: n_results must be positive"
        tool_response = mock_tool_use_response(tool_use_id="tool_err")
        final_response = mock_text_response("Sorry, search failed")
        mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]

        generator.generate_response(
            query="Query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        second_call = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call.kwargs.get("messages") or second_call[1].get("messages")
        tool_result_content = messages[-1]["content"][0]
        assert tool_result_content["content"] == "Search error: n_results must be positive"


class TestAIGeneratorSystemPrompt:
    """Tests for system prompt construction."""

    def test_conversation_history_in_system_prompt(self, generator, mock_anthropic_client):
        """When conversation_history provided, system prompt includes 'Previous conversation:' section."""
        mock_anthropic_client.messages.create.return_value = mock_text_response("Answer")

        generator.generate_response(
            query="Follow-up question",
            conversation_history="User: What is AI?\nAssistant: AI is..."
        )

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        system = call_kwargs.get("system") or mock_anthropic_client.messages.create.call_args[1].get("system")
        assert "Previous conversation:" in system
        assert "What is AI?" in system

    def test_no_history_system_prompt(self, generator, mock_anthropic_client):
        """When conversation_history=None, system prompt is just SYSTEM_PROMPT."""
        mock_anthropic_client.messages.create.return_value = mock_text_response("Answer")

        generator.generate_response(query="First question")

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        system = call_kwargs.get("system") or mock_anthropic_client.messages.create.call_args[1].get("system")
        assert system == AIGenerator.SYSTEM_PROMPT
        assert "Previous conversation:" not in system
