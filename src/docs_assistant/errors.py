"""Exceptions shared by the provider modules."""

from __future__ import annotations


class ProviderError(RuntimeError):
    """An upstream AI provider failed (network, auth, rate limit, bad response)."""

    def __init__(self, provider: str, cause: Exception) -> None:
        status = getattr(cause, "status_code", None)
        detail = type(cause).__name__ + (f" (HTTP {status})" if status else "")
        # The provider's own message is deliberately not copied: it can echo request details.
        super().__init__(f"{provider} request failed: {detail}")
        self.provider = provider
        self.status_code = status
