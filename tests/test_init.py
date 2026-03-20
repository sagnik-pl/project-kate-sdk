"""Tests for kate_sdk.init() mode detection."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

import kate_sdk as kate
from kate_sdk._state import KateSDK


def test_local_mode_with_api_key():
    kate.init(llm_api_key="sk-test-123", llm_provider="anthropic")
    sdk = KateSDK.get()
    assert sdk.mode == "local"
    assert sdk._llm_client is not None
    assert sdk._remote_runner is None


def test_local_mode_openai():
    kate.init(llm_api_key="sk-test-123", llm_provider="openai")
    sdk = KateSDK.get()
    assert sdk.mode == "local"
    assert sdk._llm_client is not None


def test_local_mode_missing_key_raises():
    with pytest.raises(ValueError, match="LLM API key"):
        kate.init()


def test_local_mode_from_env():
    with patch.dict(os.environ, {"KATE_LLM_API_KEY": "sk-env-key"}):
        kate.init()
    sdk = KateSDK.get()
    assert sdk.mode == "local"


def test_remote_mode_with_kwargs():
    kate.init(api_url="http://localhost:8000", api_key="abc", agent_id="uuid-1")
    sdk = KateSDK.get()
    assert sdk.mode == "remote"
    assert sdk._remote_runner is not None
    assert sdk._remote_runner.api_url == "http://localhost:8000"


def test_remote_mode_from_env():
    env = {
        "KATE_API_URL": "http://myserver:8000",
        "KATE_API_KEY": "mykey",
        "KATE_AGENT_ID": "agent-1",
    }
    with patch.dict(os.environ, env):
        kate.init()
    sdk = KateSDK.get()
    assert sdk.mode == "remote"
