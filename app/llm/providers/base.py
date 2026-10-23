"""Provider interface: one JSON-producing completion per call."""
from __future__ import annotations

from typing import Protocol


class ProviderError(Exception):
    """Transport/API failure (timeout, auth, rate limit, refusal)."""


class Provider(Protocol):
    name: str
    model: str

    def generate(self, *, kind: str, system: str, user: str, schema: dict, max_tokens: int, timeout: float) -> str:
        """Return the raw model text, which should be one JSON object matching `schema`."""
        ...
