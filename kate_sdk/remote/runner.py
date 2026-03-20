"""RemoteEvalRunner — uploads spans to KATE API server."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from kate_sdk.context import SpanRecord


class RemoteEvalRunner:
    def __init__(self, api_url: str, api_key: str, agent_id: str):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.agent_id = agent_id
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
        return self._client

    async def start_run(self, run_id: str, trigger: str = "automatic") -> str:
        """POST /agents/{id}/runs — create a new eval run."""
        resp = await self._get_client().post(
            f"/agents/{self.agent_id}/runs",
            json={"run_id": run_id, "trigger": trigger},
        )
        resp.raise_for_status()
        return resp.json().get("id", run_id)

    async def upload_spans(self, run_id: str, spans: list[SpanRecord]) -> None:
        """POST /agents/{id}/runs/{run_id}/spans — upload traced spans."""
        payload = [
            {
                "node_name": s.name,
                "span_kind": s.span_kind,
                "input_text": s.input,
                "output_text": s.output,
                "duration_ms": s.duration_ms,
                "error": s.error,
                "model": s.model,
                "token_count": s.token_count,
            }
            for s in spans
        ]
        resp = await self._get_client().post(
            f"/agents/{self.agent_id}/runs/{run_id}/spans",
            json={"spans": payload},
        )
        resp.raise_for_status()

    async def complete_run(self, run_id: str) -> None:
        """POST /agents/{id}/runs/{run_id}/complete — trigger server-side eval."""
        resp = await self._get_client().post(
            f"/agents/{self.agent_id}/runs/{run_id}/complete",
        )
        resp.raise_for_status()

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
