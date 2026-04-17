"""
Web search handler using Tavily API.

Intercepts Anthropic's built-in web_search_20250305 tool and replaces it
with a custom tool that NIM understands. When the model calls the tool,
executes the search via Tavily and streams the result back.
"""

import json
import os
import uuid
from typing import AsyncIterator

import httpx
from loguru import logger

from .models.anthropic import Tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAVILY_API_URL = "https://api.tavily.com/search"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# The built-in tool type Claude Code sends
ANTHROPIC_WEB_SEARCH_TYPE = "web_search_20250305"

# The custom tool name we inject so NIM understands it
CUSTOM_TOOL_NAME = "web_search"

CUSTOM_TOOL_DEFINITION = {
    "name": CUSTOM_TOOL_NAME,
    "description": (
        "Search the web for current, up-to-date information. "
        "Use this when you need recent news, documentation, or any information "
        "that may have changed since your training cutoff."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Request preprocessing
# ---------------------------------------------------------------------------

def strip_and_replace_web_search_tool(request_data):
    """
    Remove Anthropic's built-in web_search tool from the tools list and
    inject our custom Tavily-backed tool instead.

    Returns True if the built-in tool was found (meaning we're handling search).
    """
    print(f"[WEB_SEARCH] Handler called. tools count: {len(request_data.tools) if request_data.tools else 0}")
    if not request_data.tools:
        print("[WEB_SEARCH] No tools, returning False")
        return False

    had_builtin = False
    filtered = []
    for tool in request_data.tools:
        # Get both name and type
        tool_name = getattr(tool, "name", None) or (tool.get("name") if isinstance(tool, dict) else None)
        tool_type = getattr(tool, "type", None) or (tool.get("type") if isinstance(tool, dict) else None)
        # Detect built-in web_search by its name (regardless of type)
        if tool_name == ANTHROPIC_WEB_SEARCH_TYPE:
            had_builtin = True
            print(f"[WEB_SEARCH] Replaced built-in web_search tool with Tavily custom tool")
        elif tool_name == CUSTOM_TOOL_NAME:
            # Custom tool already present - keep it and enable interception
            had_builtin = True
            print(f"[WEB_SEARCH] Detected custom web_search tool, interception enabled")
            filtered.append(tool)
        else:
            # Not a web search tool, keep it
            filtered.append(tool)

    if had_builtin:
        # Ensure custom tool is present in the filtered list
        custom_already_present = any(
            getattr(t, "name", None) == CUSTOM_TOOL_NAME or (isinstance(t, dict) and t.get("name") == CUSTOM_TOOL_NAME)
            for t in filtered
        )
        if not custom_already_present:
            filtered.append(Tool(**CUSTOM_TOOL_DEFINITION))
            print("[WEB_SEARCH] Injected custom Tavily tool")
        request_data.tools = filtered
        # Also adjust tool_choice if it references the built-in tool
        if hasattr(request_data, "tool_choice") and request_data.tool_choice:
            tc = request_data.tool_choice
            if isinstance(tc, dict):
                if tc.get("name") == ANTHROPIC_WEB_SEARCH_TYPE:
                    tc["name"] = CUSTOM_TOOL_NAME
                    tc["type"] = "tool"
            # Si es objeto Pydantic, también podemos modificar atributos

    print(f"[WEB_SEARCH] Final: had_builtin={had_builtin}, final tools count={len(request_data.tools) if request_data.tools else 0}")
    return had_builtin


# ---------------------------------------------------------------------------
# Tavily search
# ---------------------------------------------------------------------------

async def tavily_search(query: str, max_results: int = 5) -> str:
    """Call Tavily search API and return formatted results as a string."""
    if not TAVILY_API_KEY:
        print("[WEB_SEARCH] TAVILY_API_KEY not set")
        return "Error: TAVILY_API_KEY is not configured. Add it to your .env file."

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": True,
        "include_raw_content": False,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(TAVILY_API_URL, json=payload)
            response.raise_for_status()
            data = response.json()

        parts = []

        # Quick answer if available
        if data.get("answer"):
            parts.append(f"**Summary:** {data['answer']}\n")

        # Individual results
        for i, result in enumerate(data.get("results", []), 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "")
            parts.append(f"{i}. **{title}**\n   URL: {url}\n   {content}\n")

        if not parts:
            return "No results found."

        return "\n".join(parts)

    except httpx.HTTPStatusError as e:
        print(f"[WEB_SEARCH] Tavily HTTP error {e.response.status_code}: {e.response.text}")
        return f"Search error: HTTP {e.response.status_code} from Tavily."
    except Exception as e:
        print(f"[WEB_SEARCH] Tavily error: {e}")
        return f"Search error: {e}"


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_web_search_response(
    model: str,
    tool_use_id: str,
    tool_name: str,
    tool_input: dict,
    original_request,
    provider,
    input_tokens: int,
    request_id: str,
) -> AsyncIterator[str]:
    """
    Execute Tavily search and stream back a complete Anthropic-format
    SSE response that includes:
      1. The tool_use block (model "decided" to call the tool)
      2. The tool_result injected back
      3. A follow-up completion from NIM with the search results in context
    """
    query = tool_input.get("query", "")
    max_results = tool_input.get("max_results", 5)

    print(f"[WEB_SEARCH] Executing search for query='{query}'")
    search_result = await tavily_search(query, max_results)
    print(f"[WEB_SEARCH] Got results ({len(search_result)} chars)")

    # -----------------------------------------------------------------------
    # Build a new request that includes the tool call + result in messages,
    # so the model can synthesize a natural language response.
    # -----------------------------------------------------------------------
    import copy
    followup_request = copy.deepcopy(original_request)

    # Append assistant tool_use + user tool_result to the conversation
    followup_request.messages = list(original_request.messages) + [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": tool_name,
                    "input": tool_input,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": search_result,
                }
            ],
        },
    ]

    # Stream the follow-up response directly from the provider
    async for chunk in provider.stream_response(
        followup_request,
        input_tokens=input_tokens,
        request_id=request_id + "_ws",
    ):
        yield chunk
