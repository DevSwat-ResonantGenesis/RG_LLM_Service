"""Agent reasoning engine with ReAct-style tool use."""

import json
import logging
from typing import List, Optional

from .config import settings
from .models import (
    AgentAction,
    AgentFinish,
    AgentRequest,
    AgentResponse,
    AgentStep,
    ChatCompletionRequest,
    Message,
    MessageRole,
    Tool,
)
from .providers.openai_provider import OpenAIProvider
from .providers.anthropic_provider import AnthropicProvider
from .tool_executor import tool_registry, execute_tool

logger = logging.getLogger(__name__)


AGENT_SYSTEM_PROMPT = """You are a DevSwat AI agent — an autonomous tool-using agent that executes tasks end-to-end.

<execution>
- Think step-by-step: analyze what's needed, use tools to gather information, take actions, verify results.
- Use tools aggressively. Every claim must be backed by tool output. Never fabricate data or results.
- If a tool call fails, try a different approach. At least 3 strategies before reporting failure.
- When done, provide a clear structured summary of results.
</execution>

<output>
- Be concise and direct. Use Markdown: **bold** for key terms, `code` for technical values, tables for data.
- End with a clear status of what was accomplished.
</output>
"""


class Agent:
    """ReAct-style agent with tool use capabilities."""

    def __init__(self, provider: str = "openai"):
        if provider == "anthropic":
            self.llm = AnthropicProvider()
        else:
            self.llm = OpenAIProvider()

    async def run(
        self,
        request: AgentRequest,
    ) -> AgentResponse:
        """Run the agent on a task."""
        messages = [
            Message(role=MessageRole.SYSTEM, content=AGENT_SYSTEM_PROMPT),
            Message(role=MessageRole.USER, content=request.task),
        ]

        steps: List[AgentStep] = []
        total_tokens = 0
        iteration = 0

        # Tool definitions: use request-provided tools or agent-accessible tools from registry
        tools = request.tools or tool_registry.get_all_tools()

        while iteration < request.max_iterations:
            iteration += 1

            # Get LLM response with tools
            completion_request = ChatCompletionRequest(
                messages=messages,
                tools=tools,
                temperature=0.7,
                max_tokens=16384,
            )

            response = await self.llm.chat_completion(completion_request)
            total_tokens += response.usage.total_tokens

            choice = response.choices[0]
            assistant_message = choice.message

            # Check if agent wants to use tools
            if choice.tool_calls:
                # Add assistant message with tool calls
                messages.append(assistant_message)

                # Execute each tool call
                for tool_call in choice.tool_calls:
                    tool_name = tool_call.function.get("name", "")
                    tool_args_raw = tool_call.function.get("arguments", "{}")

                    # Execute tool via unified registry
                    observation = await execute_tool(
                        tool_call,
                        user_id=request.user_id,
                        conversation_id=request.conversation_id,
                    )

                    # Parse args for step recording
                    try:
                        tool_input = json.loads(tool_args_raw) if isinstance(tool_args_raw, str) else tool_args_raw
                    except (json.JSONDecodeError, TypeError):
                        tool_input = {"_raw": str(tool_args_raw)}

                    # Record step
                    steps.append(
                        AgentStep(
                            action=AgentAction(
                                tool=tool_name,
                                tool_input=tool_input,
                                log=f"Using tool: {tool_name}",
                            ),
                            observation=observation,
                        )
                    )

                    # Add tool result to messages
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=observation,
                            tool_call_id=tool_call.id,
                        )
                    )

            else:
                # No tool calls - agent is done
                return AgentResponse(
                    result=assistant_message.content,
                    steps=steps,
                    total_tokens=total_tokens,
                )

        # Max iterations reached
        return AgentResponse(
            result="I was unable to complete the task within the allowed number of steps. Here's what I found so far: "
            + (messages[-1].content if messages else "No results."),
            steps=steps,
            total_tokens=total_tokens,
        )


agent = Agent()
