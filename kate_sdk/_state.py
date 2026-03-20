"""KateSDK singleton — mode routing and global state."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kate_sdk._llm import LLMClient
    from kate_sdk.context import RunContext, SpanRecord
    from kate_sdk.remote.runner import RemoteEvalRunner


class KateSDK:
    """Singleton holding SDK configuration and active run context."""

    _instance: KateSDK | None = None

    def __init__(self) -> None:
        self.mode: str = "local"  # "local" or "remote"
        self._active_ctx: RunContext | None = None
        self._llm_client: LLMClient | None = None
        self._remote_runner: RemoteEvalRunner | None = None
        self._store_results: bool = True
        self._tracer_provider = None

    @classmethod
    def get(cls) -> KateSDK:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        if cls._instance is not None and cls._instance._tracer_provider is not None:
            try:
                cls._instance._tracer_provider.shutdown()
            except Exception:
                pass
        cls._instance = None

    def init(
        self,
        *,
        llm_api_key: str | None = None,
        llm_provider: str = "anthropic",
        llm_model: str | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
        agent_id: str | None = None,
        store_results: bool = True,
        auto_instrument: bool = False,
    ) -> None:
        """Configure the SDK. Auto-detects local vs remote mode."""
        api_url = api_url or os.environ.get("KATE_API_URL")
        api_key = api_key or os.environ.get("KATE_API_KEY")
        agent_id = agent_id or os.environ.get("KATE_AGENT_ID")
        self._store_results = store_results

        if api_url:
            # Remote mode
            self.mode = "remote"
            from kate_sdk.remote.runner import RemoteEvalRunner

            self._remote_runner = RemoteEvalRunner(
                api_url=api_url,
                api_key=api_key or "",
                agent_id=agent_id or "",
            )
        else:
            # Local mode — needs an LLM key for the eval judge
            self.mode = "local"
            llm_api_key = llm_api_key or os.environ.get("KATE_LLM_API_KEY")
            if not llm_api_key:
                raise ValueError(
                    "Local mode requires an LLM API key. "
                    "Pass llm_api_key= or set KATE_LLM_API_KEY."
                )

            if llm_provider == "openai":
                from kate_sdk._llm import OpenAILLMClient

                self._llm_client = OpenAILLMClient(
                    api_key=llm_api_key, model=llm_model or "gpt-4o"
                )
            else:
                from kate_sdk._llm import AnthropicLLMClient

                self._llm_client = AnthropicLLMClient(
                    api_key=llm_api_key, model=llm_model or "claude-sonnet-4-20250514"
                )

        if auto_instrument:
            from kate_sdk.instrument import setup_auto_instrumentation

            setup_auto_instrumentation(self)

    def record_span(self, span: SpanRecord) -> None:
        if self._active_ctx is not None:
            self._active_ctx.add_span(span)

    async def finish_run(self, ctx: RunContext) -> dict | None:
        """Route to local or remote runner. Returns eval summary dict."""
        if self.mode == "remote" and self._remote_runner:
            await self._remote_runner.upload_spans(ctx.run_id, ctx.spans)
            await self._remote_runner.complete_run(ctx.run_id)
            return {"mode": "remote", "run_id": ctx.run_id, "status": "submitted"}

        # Local mode
        from kate_sdk.local.runner import LocalEvalRunner

        runner = LocalEvalRunner(
            llm_client=self._llm_client,
            store_results=self._store_results,
        )
        return await runner.run(ctx.spans)
