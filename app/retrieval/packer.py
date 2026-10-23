"""Turn retrieved chunks into labelled excerpt blocks within the corpus-token budget.

In-force excerpts (S1..) are ordered by precedence rank, then relevance. Upcoming
excerpts (U1..) carry a would_outrank list: the S labels whose documents they would
beat on their effective date (spec 5.1 "Upcoming changes").
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.ingestion.model import Chunk, DocMeta
from app.llm.client import MAX_CORPUS_TOKENS, CorpusBlock
from app.retrieval.index import ScoredChunk
from app.retrieval.precedence import dense_ranks, describe, outranks
from app.retrieval.tokenizer import count_tokens

CURRENT_BUDGET = 2400
UPCOMING_BUDGET = MAX_CORPUS_TOKENS - CURRENT_BUDGET
MAX_CURRENT_CHUNKS = 14
MAX_UPCOMING_CHUNKS = 3
MAX_CHUNKS_PER_DOC = 3


@dataclass
class PackedContext:
    blocks: list[CorpusBlock] = field(default_factory=list)
    label_to_chunk: dict[str, Chunk] = field(default_factory=dict)
    corpus_tokens: int = 0

    @property
    def doc_ids(self) -> set[str]:
        return {b.doc_id for b in self.blocks}


def _attr(value: str) -> str:
    return value.replace('"', "'").replace("\n", " ")


def _render(label: str, chunk: Chunk, doc: DocMeta, extra: dict[str, str]) -> str:
    attrs = {
        "label": label,
        "doc_id": doc.doc_id,
        "title": doc.title,
        "type": doc.doc_type,
        **extra,
        **describe(doc),
        "section": chunk.locator,
    }
    head = " ".join(f'{k}="{_attr(str(v))}"' for k, v in attrs.items())
    body = chunk.text.replace("</excerpt>", "</ excerpt>")
    return f"<excerpt {head}>\n{body}\n</excerpt>"


def _select(hits: list[ScoredChunk], budget: int, max_chunks: int, seen_text: set[str]) -> list[Chunk]:
    chosen: list[Chunk] = []
    per_doc: dict[str, int] = {}
    used = 0
    for hit in hits:
        chunk = hit.chunk
        norm = " ".join(chunk.text.lower().split())
        if norm in seen_text or per_doc.get(chunk.doc_id, 0) >= MAX_CHUNKS_PER_DOC:
            continue
        cost = chunk.tokens + 80  # label line allowance; exact count is checked below
        if used + cost > budget:
            continue
        chosen.append(chunk)
        seen_text.add(norm)
        per_doc[chunk.doc_id] = per_doc.get(chunk.doc_id, 0) + 1
        used += cost
        if len(chosen) >= max_chunks:
            break
    return chosen


def pack_context(
    current: list[ScoredChunk],
    upcoming: list[ScoredChunk],
    docs: dict[str, DocMeta],
    budget: int = MAX_CORPUS_TOKENS,
) -> PackedContext:
    seen: set[str] = set()
    cur_chunks = _select(current, min(CURRENT_BUDGET, budget), MAX_CURRENT_CHUNKS, seen)
    up_chunks = _select(upcoming, UPCOMING_BUDGET, MAX_UPCOMING_CHUNKS, seen)

    relevance = {c.chunk_id: i for i, c in enumerate(cur_chunks)}
    ranks = dense_ranks([docs[d] for d in {c.doc_id for c in cur_chunks}])
    cur_chunks.sort(key=lambda c: (ranks[c.doc_id], relevance[c.chunk_id]))

    packed = PackedContext()
    for i, chunk in enumerate(cur_chunks, start=1):
        label = f"S{i}"
        text = _render(label, chunk, docs[chunk.doc_id], {"rank": str(ranks[chunk.doc_id])})
        packed.blocks.append(CorpusBlock(label, chunk.doc_id, text))
        packed.label_to_chunk[label] = chunk
    s_labels = [(b.label, docs[b.doc_id]) for b in packed.blocks]
    for i, chunk in enumerate(up_chunks, start=1):
        label = f"U{i}"
        doc = docs[chunk.doc_id]
        beats = [lbl for lbl, cur in s_labels if outranks(doc, cur)]
        extra = {"status": "UPCOMING - not in force", "would_outrank": ",".join(beats)}
        text = _render(label, chunk, doc, extra)
        packed.blocks.append(CorpusBlock(label, chunk.doc_id, text))
        packed.label_to_chunk[label] = chunk

    # Exact budget enforcement: drop the lowest-priority blocks until it fits.
    while packed.blocks:
        packed.corpus_tokens = sum(count_tokens(b.text) for b in packed.blocks)
        if packed.corpus_tokens <= budget:
            break
        dropped = packed.blocks.pop()
        packed.label_to_chunk.pop(dropped.label, None)
    else:
        packed.corpus_tokens = 0
    return packed
