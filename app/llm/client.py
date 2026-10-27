"""The single gateway to the LLM.

Every call goes through `LLMClient.complete_json`, which enforces, before anything is
sent: the system prompt is a known constant (no corpus text), every corpus block
belongs to a document the requester may read, and corpus text is at most 3,000
cl100k_base tokens. Every call is logged with its corpus-token count.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from pydantic import BaseModel

from app.config import Settings
from app.llm.prompts import KNOWN_SYSTEM_PROMPTS
from app.llm.providers.base import Provider, ProviderError
from app.llm.schemas import LLMOutputError, parse_json_lenient, strict_json_schema
from app.logging_setup import log_event
from app.retrieval.tokenizer import count_tokens

logger = logging.getLogger(__name__)
MAX_CORPUS_TOKENS = 3000


class LLMUnavailable(Exception):
    pass


class CorpusBudgetExceeded(Exception):
    pass


class RestrictedDocumentLeak(Exception):
    pass


class DeadlineExceeded(Exception):
    pass


@dataclass(frozen=True)
class CorpusBlock:
    label: str
    doc_id: str
    text: str  # the full block as sent, label line included; all of it counts as corpus text


def build_provider(settings: Settings) -> Provider | None:
    name = settings.llm_provider
    if name == "auto":
        if settings.openai_api_key:
            name = "openai"
        elif settings.google_api_key:
            name = "gemini"
        else:
            # No key at all: run the offline stand-in so `docker compose up` works for a dry run.
            name = "fake"
    if name == "openai" and settings.openai_api_key:
        from app.llm.providers.openai_ import OpenAIProvider

        return OpenAIProvider(settings.openai_api_key, settings.llm_model)
    if name in ("gemini", "google") and settings.google_api_key:
        from app.llm.providers.gemini_ import GeminiProvider

        return GeminiProvider(settings.google_api_key, settings.llm_model)
    if name == "fake":
        from app.llm.providers.fake import FakeProvider

        return FakeProvider()
    return None


class LLMClient:
    def __init__(self, provider: Provider | None, timeout: float = 20.0) -> None:
        self.provider = provider
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return self.provider is not None

    def complete_json(
        self,
        *,
        kind: str,
        system: str,
        user_prefix: str,
        corpus_blocks: list[CorpusBlock],
        user_suffix: str,
        schema_model: type[BaseModel],
        allowed_doc_ids: set[str] | None,
        deadline: float,
        request_id: str,
        max_tokens: int,
    ) -> BaseModel:
        if system not in KNOWN_SYSTEM_PROMPTS:
            raise ValueError("system prompt must be one of the static prompts")
        if corpus_blocks and allowed_doc_ids is None:
            raise RestrictedDocumentLeak("corpus blocks sent without an allow-list")
        leaked = sorted({b.doc_id for b in corpus_blocks if b.doc_id not in (allowed_doc_ids or set())})
        corpus_tokens = sum(count_tokens(b.text) for b in corpus_blocks)
        base = {
            "request_id": request_id,
            "kind": kind,
            "provider": getattr(self.provider, "name", "none"),
            "model": getattr(self.provider, "model", "none"),
            "corpus_tokens": corpus_tokens,
            "n_blocks": len(corpus_blocks),
            "doc_ids": sorted({b.doc_id for b in corpus_blocks}),
        }
        if leaked:
            log_event(logger, "llm_call_blocked", logging.ERROR, reason="restricted_document", **base)
            raise RestrictedDocumentLeak(",".join(leaked))
        if corpus_tokens > MAX_CORPUS_TOKENS:
            log_event(logger, "llm_call_blocked", logging.ERROR, reason="corpus_budget", **base)
            raise CorpusBudgetExceeded(str(corpus_tokens))
        if self.provider is None:
            log_event(logger, "llm_call_blocked", logging.WARNING, reason="no_provider", **base)
            raise LLMUnavailable("no LLM provider configured")

        corpus_text = "\n".join(b.text for b in corpus_blocks)
        user = "\n".join(part for part in (user_prefix, corpus_text, user_suffix) if part)
        schema = strict_json_schema(schema_model)
        retried = False
        started = time.monotonic()
        while True:
            remaining = deadline - time.monotonic()
            timeout = min(self.timeout, remaining - 1.0)
            if timeout < 3.0:
                log_event(logger, "llm_call", logging.WARNING, ok=False, error="deadline", retried=retried, latency_ms=int((time.monotonic() - started) * 1000), **base)
                raise DeadlineExceeded(kind)
            try:
                text = self.provider.generate(kind=kind, system=system, user=user, schema=schema, max_tokens=max_tokens, timeout=timeout)
                result = parse_json_lenient(text, schema_model)
            except (ProviderError, LLMOutputError) as exc:
                log_event(logger, "llm_call", logging.WARNING, ok=False, error=str(exc)[:200], retried=retried, latency_ms=int((time.monotonic() - started) * 1000), **base)
                if retried or deadline - time.monotonic() < 12.0:
                    raise LLMUnavailable(str(exc)) from exc
                retried = True
                continue
            log_event(logger, "llm_call", ok=True, retried=retried, latency_ms=int((time.monotonic() - started) * 1000), **base)
            return result
