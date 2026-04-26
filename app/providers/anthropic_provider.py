import json
import time
import uuid
from typing import AsyncIterator, List, Optional

import httpx

from ..config import settings
from ..models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    MessageRole,
    StreamChunk,
    ToolCall,
    Usage,
)
from .base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider using httpx for full URL control."""

    def __init__(self):
        self.base_url = settings.ANTHROPIC_BASE_URL or "https://api.anthropic.com"
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.api_key = settings.ANTHROPIC_API_KEY
        self.default_model = settings.ANTHROPIC_MODEL

    def _convert_messages(self, messages: List[Message]) -> tuple:
        """Convert internal message format to Anthropic format.
        
        Returns (system_prompt, messages_list)
        """
        system_prompt = None
        converted = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            else:
                role = "user" if msg.role == MessageRole.USER else "assistant"
                converted.append({"role": role, "content": msg.content})

        return system_prompt, converted

    def _convert_tools(self, tools: Optional[List]) -> Optional[List[dict]]:
        """Convert internal tool format to Anthropic format."""
        if not tools:
            return None

        result = []
        for tool in tools:
            result.append({
                "name": tool.function.name,
                "description": tool.function.description,
                "input_schema": tool.function.parameters,
            })
        return result

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Generate a chat completion using Anthropic."""
        model = request.model or self.default_model
        system_prompt, messages = self._convert_messages(request.messages)
        tools = self._convert_tools(request.tools)

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if tools:
            payload["tools"] = tools

        stream_id = str(uuid.uuid4())
        created = int(time.time())

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            content = ""
            tool_calls = []

            for block in data.get("content", []):
                if block.get("type") == "text":
                    content = block.get("text", "")
                elif block.get("type") == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.get("id", ""),
                            type="function",
                            function={
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        )
                    )

            usage = data.get("usage", {})
            return ChatCompletionResponse(
                id=data.get("id", stream_id),
                created=created,
                model=data.get("model", model),
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=Message(role=MessageRole.ASSISTANT, content=content),
                        finish_reason=data.get("stop_reason", "stop"),
                        tool_calls=tool_calls if tool_calls else None,
                    )
                ],
                usage=Usage(
                    prompt_tokens=usage.get("input_tokens", 0),
                    completion_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                ),
            )
        except httpx.HTTPStatusError as exc:
            raise Exception(f"Anthropic API error: {exc.response.status_code} - {exc.response.text}")
        except Exception as exc:
            raise Exception(f"Anthropic error: {exc}")

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming chat completion using Anthropic."""
        model = request.model or self.default_model
        system_prompt, messages = self._convert_messages(request.messages)
        tools = self._convert_tools(request.tools)

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "stream": True,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if tools:
            payload["tools"] = tools

        stream_id = str(uuid.uuid4())
        created = int(time.time())

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as stream:
                    async for line in stream.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                yield StreamChunk(
                                    id=stream_id,
                                    created=created,
                                    model=model,
                                    delta={},
                                    finish_reason="stop",
                                )
                                continue
                            try:
                                data = json.loads(data_str)
                                event_type = data.get("type", "")
                                if event_type == "content_block_delta":
                                    delta = data.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        yield StreamChunk(
                                            id=stream_id,
                                            created=created,
                                            model=model,
                                            delta={"content": delta.get("text", "")},
                                            finish_reason=None,
                                        )
                                elif event_type == "message_stop":
                                    yield StreamChunk(
                                        id=stream_id,
                                        created=created,
                                        model=model,
                                        delta={},
                                        finish_reason="stop",
                                    )
                            except json.JSONDecodeError:
                                continue
        except Exception as exc:
            yield StreamChunk(
                id=stream_id,
                created=created,
                model=model,
                delta={"error": str(exc)},
                finish_reason="error",
            )

    def count_tokens(self, text: str) -> int:
        """Estimate token count for Anthropic models."""
        return len(text) // 4