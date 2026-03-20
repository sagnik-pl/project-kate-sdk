"""kate.run() async context manager."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from kate_sdk._state import KateSDK
from kate_sdk.context import RunContext
from kate_sdk.output import print_eval_summary


@asynccontextmanager
async def run(*, trigger: str = "automatic") -> AsyncIterator[RunContext]:
    """Async context manager that captures spans and runs eval on exit.

    Usage::

        async with kate.run():
            result = summarize(email_body)
        # ← Evals run here
    """
    sdk = KateSDK.get()
    ctx = RunContext()
    sdk._active_ctx = ctx

    if sdk.mode == "remote" and sdk._remote_runner:
        await sdk._remote_runner.start_run(ctx.run_id, trigger)

    try:
        yield ctx
    finally:
        sdk._active_ctx = None
        summary = await sdk.finish_run(ctx)
        if summary:
            print_eval_summary(summary)
