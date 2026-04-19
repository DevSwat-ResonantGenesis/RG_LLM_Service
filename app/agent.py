"""Agent reasoning engine with ReAct-style tool use."""

import json
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
from .tool_executor import tool_registry


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

        while iteration < request.max_iterations:
            iteration += 1

            # Get LLM response with tools
            completion_request = ChatCompletionRequest(
                messages=messages,
                tools=request.tools or tool_registry.get_all_tools(),
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
                    tool_args = tool_call.function.get("arguments", "{}")

                    # Execute tool
                    observation = await tool_registry.execute(
                        tool_call,
                        user_id=request.user_id,
                        conversation_id=request.conversation_id,
                    )

                    # Record step
                    steps.append(
                        AgentStep(
                            action=AgentAction(
                                tool=tool_name,
                                tool_input=json.loads(tool_args) if tool_args else {},
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
