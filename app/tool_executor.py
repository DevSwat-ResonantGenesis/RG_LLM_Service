"""Tool execution system for agent reasoning — backed by unified rg_tool_registry."""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

import httpx

from .config import settings
from .models import Tool, ToolCall, ToolFunction
from .rg_tool_registry.registry import (
    ToolRegistry as UnifiedRegistry,
    ToolAccess,
    ToolDef,
    ToolParam,
    ParamType,
    ToolCategory,
)
from .rg_tool_registry.builtin_tools import build_registry

logger = logging.getLogger(__name__)


# ── Local handler implementations ──────────────────────────────────────────

async def _search_memory(
    query: str,
    limit: int = 5,
    _user_id: Optional[str] = None,
    **kwargs,
) -> Any:
    """Search memory service."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{settings.MEMORY_SERVICE_URL}/memory/retrieve",
                json={
                    "user_id": _user_id,
                    "query": query,
                    "limit": limit,
                },
            )
            if response.status_code == 200:
                return response.json()
    except httpx.RequestError as e:
        return {"error": str(e)}
    return []


async def _get_conversation_context(
    conversation_id: str,
    _user_id: Optional[str] = None,
    **kwargs,
) -> dict:
    """Get conversation context from chat service."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{settings.CHAT_SERVICE_URL}/chat/conversations/{conversation_id}/context",
                headers={"x-user-id": _user_id} if _user_id else {},
            )
            if response.status_code == 200:
                return response.json()
    except httpx.RequestError as e:
        return {"error": str(e)}
    return {"error": "Failed to get conversation context"}


async def _create_workflow(
    name: str,
    trigger: str,
    actions: List[dict],
    _user_id: Optional[str] = None,
    **kwargs,
) -> dict:
    """Create a workflow in workflow service."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"http://workflow_service:8000/workflow/workflows",
                json={
                    "name": name,
                    "trigger": trigger,
                    "actions": actions,
                    "user_id": _user_id,
                },
            )
            if response.status_code in (200, 201):
                return response.json()
    except httpx.RequestError as e:
        return {"error": str(e)}
    return {"error": "Failed to create workflow"}


async def _http_request(
    method: str = "GET",
    url: str = "",
    headers: Optional[Dict] = None,
    body: Optional[Dict] = None,
    **kwargs,
) -> dict:
    """Make an HTTP request."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=body if method in ("POST", "PUT") else None,
            )
            return {
                "status_code": response.status_code,
                "body": response.text[:1000],
            }
    except httpx.RequestError as e:
        return {"error": str(e)}


# ── Handler wiring ─────────────────────────────────────────────────────────

_HANDLER_MAP: Dict[str, Callable] = {
    "memory_search": _search_memory,
    "memory_read": _search_memory,
    "http_request": _http_request,
    "get_conversation_context": _get_conversation_context,
    "create_workflow": _create_workflow,
}

# Additional ToolDefs for tools not yet in the shared builtin_tools registry
_EXTRA_TOOLS = [
    ToolDef(
        name="get_conversation_context",
        description="Get conversation context and history",
        category=ToolCategory.MEMORY,
        params=[
            ToolParam("conversation_id", ParamType.STRING, "conversation UUID", required=True),
        ],
        access={ToolAccess.AGENT, ToolAccess.REGISTERED},
        handler_fn=_get_conversation_context,
        priority=15,
    ),
    ToolDef(
        name="create_workflow",
        description="Create a new workflow automation",
        category=ToolCategory.WORKFLOWS,
        params=[
            ToolParam("name", ParamType.STRING, "workflow name", required=True),
            ToolParam("trigger", ParamType.STRING, "trigger event", required=True),
            ToolParam("actions", ParamType.ARRAY, "list of action objects", required=True),
        ],
        access={ToolAccess.AGENT, ToolAccess.REGISTERED},
        handler_fn=_create_workflow,
        priority=20,
    ),
]


# ── Build the unified registry ─────────────────────────────────────────────

def _build_agent_registry() -> UnifiedRegistry:
    """Build unified registry with agent-accessible tools and local handlers."""
    registry = build_registry()

    # Wire local handlers for tools already in the shared registry
    for tool_name, handler in _HANDLER_MAP.items():
        if tool_name in registry:
            registry.register_handler(tool_name, handler)

    # Register additional tools not yet in the shared registry
    for tool in _EXTRA_TOOLS:
        registry.register(tool)

    return registry


_registry = _build_agent_registry()


# ── Conversion helpers ──────────────────────────────────────────────────────

def _tooldef_to_model(td: ToolDef) -> Tool:
    """Convert a ToolDef to a Pydantic Tool model (OpenAI format)."""
    openai_fmt = td.to_openai()
    return Tool(
        type="function",
        function=ToolFunction(
            name=openai_fmt["function"]["name"],
            description=openai_fmt["function"]["description"],
            parameters=openai_fmt["function"]["parameters"],
        ),
    )


# ── Public execution function ───────────────────────────────────────────────

async def execute_tool(
    tool_call: ToolCall,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> str:
    """Execute a tool call and return the result as a string."""
    tool_name = tool_call.function.get("name")
    arguments_raw = tool_call.function.get("arguments", "{}")

    try:
        if isinstance(arguments_raw, str):
            arguments = json.loads(arguments_raw)
        else:
            arguments = dict(arguments_raw)
    except (json.JSONDecodeError, TypeError):
        return f"Error: Invalid arguments: {arguments_raw}"

    handler = _registry.get_handler(tool_name)
    if not handler:
        return f"Error: Unknown tool: {tool_name}"

    try:
        # Inject context for handlers that need it
        arguments["_user_id"] = user_id
        arguments["_conversation_id"] = conversation_id

        result = await handler(**arguments)
        return json.dumps(result) if isinstance(result, (dict, list)) else str(result)
    except Exception as e:
        logger.exception("Tool execution failed: %s", tool_name)
        return f"Error executing tool {tool_name}: {str(e)}"


# ── Backward-compatible wrapper ─────────────────────────────────────────────

class ToolRegistryCompat:
    """Backward-compatible wrapper around unified rg_tool_registry.

    Provides the same interface consumed by agent.py and routers.py:
      - get_all_tools() → List[Tool]
      - execute(tool_call, user_id, conversation_id) → str
      - get_tool(name) → Optional[Callable]
      - get_tool_definition(name) → Optional[Tool]
    """

    def get_all_tools(self) -> List[Tool]:
        """Get all agent-accessible tools as Pydantic Tool models."""
        return [
            _tooldef_to_model(td)
            for td in _registry.get_tools(access=ToolAccess.AGENT)
        ]

    async def execute(
        self,
        tool_call: ToolCall,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> str:
        """Execute a tool call."""
        return await execute_tool(tool_call, user_id, conversation_id)

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool handler by name."""
        return _registry.get_handler(name)

    def get_tool_definition(self, name: str) -> Optional[Tool]:
        """Get a tool definition by name."""
        td = _registry.get(name)
        return _tooldef_to_model(td) if td else None

    @property
    def unified(self) -> UnifiedRegistry:
        """Access the underlying unified registry directly."""
        return _registry


tool_registry = ToolRegistryCompat()
