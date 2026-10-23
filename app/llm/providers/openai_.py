"""OpenAI chat completions with a strict JSON schema (falls back to JSON mode)."""
from __future__ import annotations

from app.llm.providers.base import ProviderError

REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str = "") -> None:
        import openai

        self._openai = openai
        self.model = model or "gpt-4.1"
        self._client = openai.OpenAI(api_key=api_key, max_retries=0)
        self._schema_mode = True

    def generate(self, *, kind: str, system: str, user: str, schema: dict, max_tokens: int, timeout: float) -> str:
        params: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": max_tokens,
        }
        if self.model.startswith(REASONING_PREFIXES):
            params["reasoning_effort"] = "low"
        else:
            params["temperature"] = 0
            params["seed"] = 42
        client = self._client.with_options(timeout=timeout)
        try:
            if self._schema_mode:
                try:
                    return self._call(client, params, {"type": "json_schema", "json_schema": {"name": kind, "schema": schema, "strict": True}})
                except self._openai.BadRequestError as exc:
                    if "response_format" not in str(exc) and "json_schema" not in str(exc):
                        raise
                    self._schema_mode = False  # model without structured outputs: use JSON mode
            return self._call(client, params, {"type": "json_object"})
        except self._openai.OpenAIError as exc:
            raise ProviderError(f"openai: {type(exc).__name__}: {str(exc)[:200]}") from exc

    @staticmethod
    def _call(client, params: dict, response_format: dict) -> str:
        resp = client.chat.completions.create(**params, response_format=response_format)
        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            raise ProviderError("openai: refusal")
        return choice.message.content or ""
