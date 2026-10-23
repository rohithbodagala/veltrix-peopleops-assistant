"""Google Gemini (google-genai SDK) with JSON output constrained by a JSON schema."""
from __future__ import annotations

from app.llm.providers.base import ProviderError


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str = "") -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self.model = model or "gemini-2.5-flash"
        self._client = genai.Client(api_key=api_key)

    def generate(self, *, kind: str, system: str, user: str, schema: dict, max_tokens: int, timeout: float) -> str:
        types = self._types
        http_options = types.HttpOptions(timeout=int(timeout * 1000))
        if hasattr(types, "HttpRetryOptions"):
            http_options = types.HttpOptions(timeout=int(timeout * 1000), retry_options=types.HttpRetryOptions(attempts=1))
        config: dict = {
            "system_instruction": system,
            "temperature": 0,
            "seed": 42,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
            "response_json_schema": schema,
            "http_options": http_options,
        }
        if "2.5-flash" in self.model:
            config["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        try:
            resp = self._client.models.generate_content(
                model=self.model, contents=user, config=types.GenerateContentConfig(**config)
            )
        except Exception as exc:  # google-genai raises several unrelated error types
            raise ProviderError(f"gemini: {type(exc).__name__}: {str(exc)[:200]}") from exc
        return resp.text or ""
