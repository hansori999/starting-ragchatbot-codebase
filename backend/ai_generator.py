import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
1. **search_course_content** — Search course materials for specific content or detailed educational information.
2. **get_course_outline** — Retrieve a course's full outline: title, course link, and all lessons (number + title). Use this for questions about what a course covers, its structure, syllabus, or lesson listings.

Tool Usage:
- **Up to two tool calls per query**
- Use a second tool call only when the first tool's results are insufficient or when you need information from a different tool to fully answer the question
- For questions about course structure, outlines, or lesson lists → use `get_course_outline`
- For questions about specific course content or topics → use `search_course_content`
- Synthesize tool results into accurate, fact-based responses
- When returning an outline, include the course title, course link, and every lesson's number and title
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course outline/structure questions**: Use `get_course_outline`, then present the full outline
- **Course content questions**: Use `search_course_content`, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the tool results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]

        # Initial API call
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        response = self.client.messages.create(**api_params)

        # Tool-use loop
        for round_num in range(self.MAX_TOOL_ROUNDS):
            if response.stop_reason != "tool_use" or not tool_manager:
                break

            # Append assistant's tool_use response
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tool_use blocks, collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = tool_manager.execute_tool(block.name, **block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})

            # Build follow-up params
            follow_up_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }

            # Include tools only if more rounds remain
            if round_num < self.MAX_TOOL_ROUNDS - 1 and tools:
                follow_up_params["tools"] = tools
                follow_up_params["tool_choice"] = {"type": "auto"}

            response = self.client.messages.create(**follow_up_params)

        return self._extract_text(response)

    def _extract_text(self, response):
        """Extract text from a response that may contain mixed content blocks."""
        for block in response.content:
            if block.type == "text":
                return block.text
        return response.content[0].text