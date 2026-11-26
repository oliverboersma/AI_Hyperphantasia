"""Perception backends: how a problem statement becomes a Scene."""

from .offline import OfflineBackend

__all__ = ["OfflineBackend", "ClaudeBackend"]


def __getattr__(name):
    if name == "ClaudeBackend":  # imported lazily so the package works without the SDK
        from .claude import ClaudeBackend
        return ClaudeBackend
    raise AttributeError(name)
