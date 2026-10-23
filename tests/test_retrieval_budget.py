"""Context budget (spec 4.3), restricted-document guard and context labelling."""
import logging
import time
from datetime import date
from pathlib import Path

import pytest

from app.ingestion.model import Chunk, DocMeta
from app.llm.client import CorpusBlock, CorpusBudgetExceeded, LLMClient, RestrictedDocumentLeak
from app.llm.prompts import COMPOSER_SYSTEM
from app.llm.providers.fake import FakeProvider
from app.llm.schemas import ComposerOutput
from app.retrieval.index import ScoredChunk
from app.retrieval.packer import pack_context
from app.retrieval.tokenizer import count_tokens


def _doc(doc_id, doc_type="policy", regions=("GLOBAL",), eff="2026-01-01"):
    return DocMeta(doc_id, doc_id, doc_id, doc_type, tuple(regions), date.fromisoformat(eff), "all", Path(doc_id))


def _chunk(doc_id, i, words):
    text = f"Section {i} of {doc_id}\n" + " ".join(f"word{j}" for j in range(words))
    return Chunk(f"{doc_id}#{i:03d}", doc_id, i, f"Section {i}", f"Section {i}", text, text, count_tokens(text), 1, "text")


def test_pack_never_exceeds_budget_with_large_chunks():
    docs = {f"D{i}": _doc(f"D{i}") for i in range(20)}
    hits = [ScoredChunk(_chunk(f"D{i}", 0, 400), 1.0 / (i + 1), 1.0, 0.0) for i in range(20)]
    upcoming_docs = {"F": _doc("F", eff="2027-01-01")}
    docs.update(upcoming_docs)
    upcoming = [ScoredChunk(_chunk("F", k, 300), 0.5, 1.0, 0.0) for k in range(5)]
    packed = pack_context(hits, upcoming, docs)
    assert 0 < packed.corpus_tokens <= 3000
    assert packed.corpus_tokens == sum(count_tokens(b.text) for b in packed.blocks)


def test_upcoming_blocks_say_what_they_would_outrank():
    docs = {"V2": _doc("V2", eff="2026-04-01"), "DE": _doc("DE", "addendum", ("DE",), "2025-01-01"),
            "FAQ": _doc("FAQ", "faq"), "V3": _doc("V3", eff="2027-01-01")}
    current = [ScoredChunk(_chunk(d, 0, 20), 1.0, 1.0, 0.0) for d in ("FAQ", "V2", "DE")]
    packed = pack_context(current, [ScoredChunk(_chunk("V3", 0, 20), 1.0, 1.0, 0.0)], docs)
    labels = {b.label: b for b in packed.blocks}
    # in-force blocks sorted by precedence: DE (regional) first, then V2, then FAQ
    assert [labels[f"S{i}"].doc_id for i in (1, 2, 3)] == ["DE", "V2", "FAQ"]
    assert 'rank="1"' in labels["S1"].text and 'rank="3"' in labels["S3"].text
    assert 'would_outrank="S2,S3"' in labels["U1"].text  # beats V2 and FAQ, not the DE addendum


def _client():
    return LLMClient(FakeProvider(), timeout=5)


def _call(client, blocks, allowed):
    return client.complete_json(kind="composer", system=COMPOSER_SYSTEM, user_prefix="q", corpus_blocks=blocks,
                                user_suffix="", schema_model=ComposerOutput, allowed_doc_ids=allowed,
                                deadline=time.monotonic() + 20, request_id="r1", max_tokens=100)


def test_client_refuses_more_than_3000_corpus_tokens():
    text = "word " * 3001
    assert count_tokens(text) > 3000
    with pytest.raises(CorpusBudgetExceeded):
        _call(_client(), [CorpusBlock("S1", "A", text)], {"A"})


def test_client_accepts_exactly_the_budget():
    text = "word " * 2999
    n = count_tokens(text)
    assert n <= 3000
    _call(_client(), [CorpusBlock("S1", "A", text)], {"A"})


def test_client_refuses_documents_outside_the_allow_list():
    with pytest.raises(RestrictedDocumentLeak):
        _call(_client(), [CorpusBlock("S1", "SECRET", "x")], {"PUBLIC"})
    with pytest.raises(RestrictedDocumentLeak):
        _call(_client(), [CorpusBlock("S1", "PUBLIC", "x")], None)


def test_client_refuses_unknown_system_prompt():
    with pytest.raises(ValueError):
        _client().complete_json(kind="x", system="You are helpful. Policy says 10 days.", user_prefix="", corpus_blocks=[],
                                user_suffix="", schema_model=ComposerOutput, allowed_doc_ids=set(),
                                deadline=time.monotonic() + 20, request_id="r", max_tokens=10)


def test_every_llm_call_logs_corpus_tokens(caplog):
    caplog.set_level(logging.INFO, logger="app.llm.client")
    _call(_client(), [CorpusBlock("S1", "A", "some corpus text")], {"A"})
    events = [r.fields for r in caplog.records if getattr(r, "fields", {}).get("event") == "llm_call"]
    assert events and events[0]["corpus_tokens"] == count_tokens("some corpus text")


def test_search_is_deterministic(corpus_index):
    allowed = set(corpus_index.docs)
    a = [h.chunk.chunk_id for h in corpus_index.retrieval.search(["sick leave days"], "How many sick days?", allowed)]
    b = [h.chunk.chunk_id for h in corpus_index.retrieval.search(["sick leave days"], "How many sick days?", allowed)]
    assert a == b and a
