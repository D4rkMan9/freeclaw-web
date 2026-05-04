"""FastAPI route handlers."""

import json
import traceback
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import Settings
from providers.common import get_user_facing_error_message
from providers.exceptions import InvalidRequestError, ProviderError

from .dependencies import get_provider_for_type, get_settings, require_api_key
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.responses import TokenCountResponse
from .optimization_handlers import try_optimizations
from .request_utils import get_token_count
from .web_search_handler import (
    ANTHROPIC_WEB_SEARCH_TYPE,
    CUSTOM_TOOL_NAME,
    stream_web_search_response,
    strip_and_replace_web_search_tool,
)

router = APIRouter()


# =============================================================================
# Web search stream interceptor
# =============================================================================


async def _intercept_for_web_search(
    stream, request_data, provider, input_tokens, request_id
):
    """
    Wraps a provider stream. If the model emits a tool_use call for web_search,
    intercepts it, runs Tavily, and streams the follow-up response instead.
    Otherwise passes chunks through unchanged.
    """
    buffer = []
    tool_use_id = None
    tool_name: str = CUSTOM_TOOL_NAME
    tool_input_str = ""
    in_tool_use = False

    async for chunk in stream:
        # Buffer only before tool use begins
        if not in_tool_use:
            buffer.append(chunk)

        # Decode chunk
        chunk_str = (
            chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="replace")
        )

        # Parse each SSE data line
        for line in chunk_str.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                data = json.loads(line[5:].strip())
            except Exception:
                continue

            if not in_tool_use:
                # Detect tool_use start
                cb = data.get("content_block", {})
                if data.get("type") == "content_block_start":
                    logger.debug(
                        "[INTERCEPTOR] content_block_start: type={}, name={}",
                        cb.get("type"),
                        cb.get("name"),
                    )
                if cb.get("type") == "tool_use" and cb.get("name") in (
                    ANTHROPIC_WEB_SEARCH_TYPE,
                    CUSTOM_TOOL_NAME,
                ):
                    in_tool_use = True
                    tool_use_id = cb.get("id", f"toolu_{uuid.uuid4().hex[:16]}")
                    tool_name = cb.get("name") or CUSTOM_TOOL_NAME
                    logger.debug(
                        "[INTERCEPTOR] Detected tool_use! id={}, name={}",
                        tool_use_id,
                        tool_name,
                    )
                    buffer = []  # discard everything before tool_use
            else:
                # Process deltas and stop when done
                if data.get("type") == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "input_json_delta":
                        tool_input_str += delta.get("partial_json", "")
                elif data.get("type") == "content_block_stop" and tool_use_id:
                    try:
                        tool_input = (
                            json.loads(tool_input_str) if tool_input_str else {}
                        )
                    except json.JSONDecodeError:
                        tool_input = {"query": tool_input_str}
                    logger.debug("[INTERCEPTOR] tool_input={}", tool_input)
                    async for search_chunk in stream_web_search_response(
                        model=request_data.model,
                        tool_use_id=tool_use_id,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        original_request=request_data,
                        provider=provider,
                        input_tokens=input_tokens,
                        request_id=request_id,
                    ):
                        yield search_chunk
                    return

    # End of stream: never detected tool_use? flush buffer.
    if not in_tool_use:
        logger.debug(
            "[INTERCEPTOR] No tool_use detected, yielding buffered original stream"
        )
        for buffered in buffer:
            yield buffered


# =============================================================================
# Routes
# =============================================================================
@router.post("/v1/messages")
async def create_message(
    request_data: MessagesRequest,
    raw_request: Request,
    settings: Settings = Depends(get_settings),
    _auth=Depends(require_api_key),
):
    """Create a message (always streaming)."""
    logger.debug("[ROUTES] create_message called")

    try:
        if not request_data.messages:
            raise InvalidRequestError("messages cannot be empty")

        optimized = try_optimizations(request_data, settings)
        if optimized is not None:
            return optimized
        logger.debug("No optimization matched, routing to provider")

        # Replace built-in web_search tool with Tavily-backed custom tool
        has_web_search = strip_and_replace_web_search_tool(request_data)
        logger.debug("[ROUTES] has_web_search = {}", has_web_search)

        # Resolve provider from the model-aware mapping
        provider_type = Settings.parse_provider_type(
            request_data.resolved_provider_model or settings.model
        )
        provider = get_provider_for_type(provider_type)

        request_id = f"req_{uuid.uuid4().hex[:12]}"
        logger.info(
            "API_REQUEST: request_id={} model={} messages={} web_search={}",
            request_id,
            request_data.model,
            len(request_data.messages),
            has_web_search,
        )
        logger.debug("FULL_PAYLOAD [{}]: {}", request_id, request_data.model_dump())

        input_tokens = get_token_count(
            request_data.messages, request_data.system, request_data.tools
        )

        raw_stream = provider.stream_response(
            request_data,
            input_tokens=input_tokens,
            request_id=request_id,
        )

        # If web search is active, wrap the stream with our interceptor
        final_stream = (
            _intercept_for_web_search(
                raw_stream, request_data, provider, input_tokens, request_id
            )
            if has_web_search
            else raw_stream
        )

        return StreamingResponse(
            final_stream,
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except ProviderError:
        raise
    except Exception as e:
        logger.error(f"Error: {e!s}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=getattr(e, "status_code", 500),
            detail=get_user_facing_error_message(e),
        ) from e


@router.post("/v1/messages/count_tokens")
async def count_tokens(request_data: TokenCountRequest, _auth=Depends(require_api_key)):
    """Count tokens for a request."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    with logger.contextualize(request_id=request_id):
        try:
            tokens = get_token_count(
                request_data.messages, request_data.system, request_data.tools
            )
            logger.info(
                "COUNT_TOKENS: request_id={} model={} messages={} input_tokens={}",
                request_id,
                getattr(request_data, "model", "unknown"),
                len(request_data.messages),
                tokens,
            )
            return TokenCountResponse(input_tokens=tokens)
        except Exception as e:
            logger.error(
                "COUNT_TOKENS_ERROR: request_id={} error={}\n{}",
                request_id,
                get_user_facing_error_message(e),
                traceback.format_exc(),
            )
            raise HTTPException(
                status_code=500, detail=get_user_facing_error_message(e)
            ) from e


@router.get("/")
async def root(
    settings: Settings = Depends(get_settings), _auth=Depends(require_api_key)
):
    """Root endpoint."""
    return {
        "status": "ok",
        "provider": settings.provider_type,
        "model": settings.model,
    }


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.post("/stop")
async def stop_cli(request: Request, _auth=Depends(require_api_key)):
    """Stop all CLI sessions and pending tasks."""
    handler = getattr(request.app.state, "message_handler", None)
    if not handler:
        cli_manager = getattr(request.app.state, "cli_manager", None)
        if cli_manager:
            await cli_manager.stop_all()
            logger.info("STOP_CLI: source=cli_manager cancelled_count=N/A")
            return {"status": "stopped", "source": "cli_manager"}
        raise HTTPException(status_code=503, detail="Messaging system not initialized")

    count = await handler.stop_all_tasks()
    logger.info("STOP_CLI: source=handler cancelled_count={}", count)
    return {"status": "stopped", "cancelled_count": count}
