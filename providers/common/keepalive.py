"""SSE keep-alive heartbeat wrapper.

Injects SSE comment lines (": keep-alive\\n\\n") at regular intervals
to prevent clients and intermediaries from closing idle connections.
"""

import asyncio
import time
from collections.abc import AsyncIterator

SSE_KEEPALIVE_INTERVAL_S = 15.0
SSE_KEEPALIVE_LINE = ": keep-alive\n\n"


async def with_keepalive(
    stream: AsyncIterator[str],
    interval_s: float = SSE_KEEPALIVE_INTERVAL_S,
) -> AsyncIterator[str]:
    """Wrap an SSE stream with periodic keep-alive comment lines.

    Sends an SSE comment (``: keep-alive``) every *interval_s* seconds
    when the upstream has not produced any data.  SSE comments are
    ignored by spec-compliant clients but keep the TCP connection alive
    through proxies, load-balancers, and the Claude Code CLI timeout
    logic.

    The wrapper uses ``asyncio.wait_for`` instead of spawning a
    background timer so that there is only one coroutine per stream.
    """
    last_yield = time.monotonic()

    while True:
        timeout = interval_s - (time.monotonic() - last_yield)
        timeout = max(0.1, min(timeout, interval_s))

        try:
            chunk = await asyncio.wait_for(stream.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            return
        except TimeoutError:
            # No data from upstream within the interval → send heartbeat
            yield SSE_KEEPALIVE_LINE
            last_yield = time.monotonic()
            continue

        last_yield = time.monotonic()
        yield chunk

        # Drain any immediately available chunks without waiting
        # (preserves throughput when the provider is actively streaming)
        while True:
            try:
                chunk = await asyncio.wait_for(stream.__anext__(), timeout=0.05)
            except StopAsyncIteration:
                return
            except TimeoutError:
                break
            last_yield = time.monotonic()
            yield chunk
