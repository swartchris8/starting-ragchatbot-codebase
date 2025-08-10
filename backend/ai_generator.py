from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage Guidelines:
- **Search Tool**: Use for questions about specific course content or detailed educational materials
- **Outline Tool**: Use for course structure, lesson lists, or overview requests  
- **Sequential Tool Calls**: You can make up to 2 rounds of tool calls total
  - Round 1: Gather initial information, identify what else you might need
  - Round 2: Make additional searches based on Round 1 results to provide complete answers
- **Complex Queries**: For queries requiring multiple searches (e.g., "find courses that discuss the same topic as lesson X of course Y"), use the first round to identify the topic, then search for related content in the second round
- Synthesize all tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use search tool first, then answer
- **Course structure/outline questions**: Use outline tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the outline"

For outline queries, always include:
- Course title and instructor (if available)
- Course link (if available)  
- Complete lesson list with lesson numbers and titles
- Video links for lessons (if available)

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
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Uses sequential tool calling for up to 2 rounds when tools are available.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Use sequential tool calling when tools are available
        if tools and tool_manager:
            return self.generate_response_with_sequential_tools(
                query, conversation_history, tools, tool_manager
            )

        # Fallback: Original single-call behavior when no tools
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Return direct response - handle empty content gracefully
        if not response.content:
            return "I apologize, but I encountered an issue generating a response. Please try rephrasing your question."

        return response.content[0].text

    def generate_response_with_sequential_tools(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with up to 2 sequential rounds of tool calling.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        if not tools or not tool_manager:
            # No tools available, use direct generation
            return self.generate_response(
                query, conversation_history, tools, tool_manager
            )

        max_rounds = 2
        messages = [{"role": "user", "content": query}]

        # Build system content with conversation history
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Loop through up to max_rounds of tool calling
        for round_num in range(max_rounds):
            try:
                # Make API call with tools available
                response = self._make_api_call(messages, system_content, tools)

                # Termination condition: Claude doesn't use tools
                if response.stop_reason != "tool_use":
                    return self._extract_text_response(response)

                # Execute tools and add results to conversation
                tool_success = self._execute_and_append_tools(
                    response, messages, tool_manager
                )

                # Termination condition: Tool execution failed
                if not tool_success:
                    return "I encountered an error while searching. Please try rephrasing your question."

            except Exception as e:
                print(f"API call failed in round {round_num + 1}: {e}")
                # Fall back to original method if sequential calling fails
                return self.generate_response(
                    query, conversation_history, tools, tool_manager
                )

        # After max_rounds, make final call without tools for summary
        try:
            final_response = self._make_api_call(messages, system_content, tools=None)
            return self._extract_text_response(final_response)
        except Exception as e:
            print(f"Final API call failed: {e}")
            return "I gathered some information but encountered an issue generating the final response. Please try rephrasing your question."

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # Get final response
        final_response = self.client.messages.create(**final_params)

        # Handle empty content gracefully
        if not final_response.content:
            return "I apologize, but I encountered an issue generating a response. Please try rephrasing your question."

        # Return the first content block's text
        return final_response.content[0].text

    def _make_api_call(
        self,
        messages: List[Dict[str, Any]],
        system_content: str,
        tools: Optional[List] = None,
    ):
        """
        Make API call with consistent parameters.

        Args:
            messages: Conversation messages
            system_content: System prompt
            tools: Optional tools to make available

        Returns:
            API response object
        """
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        return self.client.messages.create(**api_params)

    def _execute_and_append_tools(
        self, response, messages: List[Dict[str, Any]], tool_manager
    ) -> bool:
        """
        Execute tools and append both AI response and tool results to message chain.

        Args:
            response: API response containing tool use requests
            messages: Message list to append to
            tool_manager: Manager to execute tools

        Returns:
            True if tool execution succeeded, False otherwise
        """
        try:
            # Add AI's tool use response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tool calls
            tool_results = []
            for content_block in response.content:
                if content_block.type == "tool_use":
                    result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": result,
                        }
                    )

            # Add tool results to messages
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            return True
        except Exception as e:
            print(f"Tool execution error: {e}")
            return False

    def _extract_text_response(self, response) -> str:
        """
        Extract text response from API response, handling empty content gracefully.

        Args:
            response: API response object

        Returns:
            Response text or error message
        """
        if not response.content:
            return "I apologize, but I encountered an issue generating a response. Please try rephrasing your question."

        return response.content[0].text
