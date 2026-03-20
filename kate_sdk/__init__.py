"""KATE SDK — local-first auto-eval for AI agents."""

from kate_sdk._state import KateSDK
from kate_sdk.decorators import trace
from kate_sdk.run_context import run


def init(**kwargs) -> None:
    """Initialize the KATE SDK. See KateSDK.init() for parameters."""
    KateSDK.get().init(**kwargs)


__all__ = ["init", "trace", "run"]
