"""Tests for RemoteEvalRunner with mocked httpx."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from kate_sdk.context import SpanRecord
from kate_sdk.remote.runner import RemoteEvalRunner

_DUMMY_REQUEST = httpx.Request("POST", "http://kate.test:8000/test")


@pytest.fixture
def runner():
    return RemoteEvalRunner(
        api_url="http://kate.test:8000",
        api_key="test-key",
        agent_id="agent-123",
    )


@pytest.mark.asyncio
async def test_start_run(runner):
    mock_response = httpx.Response(200, json={"id": "run-456"}, request=_DUMMY_REQUEST)
    with patch.object(runner, "_get_client") as mock_client:
        mock_client.return_value.post = AsyncMock(return_value=mock_response)
        result = await runner.start_run("run-456", "automatic")

    assert result == "run-456"
    mock_client.return_value.post.assert_called_once_with(
        "/agents/agent-123/runs",
        json={"run_id": "run-456", "trigger": "automatic"},
    )


@pytest.mark.asyncio
async def test_upload_spans(runner):
    mock_response = httpx.Response(200, json={"ok": True}, request=_DUMMY_REQUEST)
    spans = [
        SpanRecord(name="node1", input="in", output="out", span_kind="LLM", duration_ms=50.0),
    ]
    with patch.object(runner, "_get_client") as mock_client:
        mock_client.return_value.post = AsyncMock(return_value=mock_response)
        await runner.upload_spans("run-456", spans)

    call_args = mock_client.return_value.post.call_args
    assert call_args[0][0] == "/agents/agent-123/runs/run-456/spans"
    payload = call_args[1]["json"]["spans"]
    assert len(payload) == 1
    assert payload[0]["node_name"] == "node1"
    assert payload[0]["input_text"] == "in"
    assert payload[0]["output_text"] == "out"


@pytest.mark.asyncio
async def test_complete_run(runner):
    mock_response = httpx.Response(
        200, json={"status": "completed"}, request=_DUMMY_REQUEST
    )
    with patch.object(runner, "_get_client") as mock_client:
        mock_client.return_value.post = AsyncMock(return_value=mock_response)
        await runner.complete_run("run-456")

    mock_client.return_value.post.assert_called_once_with(
        "/agents/agent-123/runs/run-456/complete",
    )


@pytest.mark.asyncio
async def test_error_handling(runner):
    mock_response = httpx.Response(
        500, text="Internal Server Error", request=_DUMMY_REQUEST
    )
    with patch.object(runner, "_get_client") as mock_client:
        mock_client.return_value.post = AsyncMock(return_value=mock_response)
        with pytest.raises(httpx.HTTPStatusError):
            await runner.start_run("run-456", "automatic")
