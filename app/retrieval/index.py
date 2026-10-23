"""Hybrid retrieval: BM25 over every query plus dense cosine, fused with RRF.

Callers pass the set of document ids the requester may use; chunks of any other
document are never scored or returned (access control happens before ranking).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from app.ingestion.model import Chunk
from app.retrieval.embeddings import Embedder

STOPWORDS = set(
    """a an and are as at be by can do does for from get got have how i if in is it its me my
    of on or our so that the their them there these this to was we what when where which who why
    will with would you your much many any about into than then also per""".split()
)
RRF_K = 60
BM25_FLOOR = 0.25  # fraction of the best BM25 score
DENSE_MARGIN = 0.12  # cosine distance from the best dense match
EXPANSION_SEEDS = 3
EXPANSION_PER_SEED = 4
EXPANSION_MIN_COSINE = 0.70
EXPANSION_BM25_RATIO = 0.2


def _stem(token: str) -> str:
    for suffix in ("ies", "es", "s"):
        if len(token) > 4 and token.endswith(suffix) and not token.endswith("ss"):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    bm25: float
    cosine: float


class RetrievalIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None) -> None:
        self.chunks = chunks
        self.embedder = embedder
        corpus = [tokenize(c.index_text) for c in chunks] or [[""]]
        self._bm25 = BM25Okapi(corpus)
        self._vectors: np.ndarray | None = None
        if embedder is not None and chunks:
            self._vectors = embedder.embed_passages([c.index_text for c in chunks])

    def search(self, queries: list[str], question: str, allowed_doc_ids: set[str], k: int = 24) -> list[ScoredChunk]:
        allowed = [i for i, c in enumerate(self.chunks) if c.doc_id in allowed_doc_ids]
        if not allowed:
            return []
        rrf = {i: 0.0 for i in allowed}
        bm25_best = {i: 0.0 for i in allowed}
        texts = [q for q in dict.fromkeys([*queries, question]) if q and q.strip()]
        for q in texts:
            tokens = tokenize(q)
            if not tokens:
                continue
            scores = self._bm25.get_scores(tokens)
            ranked = sorted((i for i in allowed if scores[i] > 0), key=lambda i: (-scores[i], i))
            for rank, i in enumerate(ranked[:k], start=1):
                rrf[i] += 1.0 / (RRF_K + rank)
                bm25_best[i] = max(bm25_best[i], float(scores[i]))
        cosines = {i: 0.0 for i in allowed}
        if self._vectors is not None and self.embedder is not None:
            qv = self.embedder.embed_query(question or " ".join(queries))
            sims = self._vectors[allowed] @ qv
            for i, s in zip(allowed, sims):
                cosines[i] = float(s)
            ranked = sorted(allowed, key=lambda i: (-cosines[i], i))
            for rank, i in enumerate(ranked[:k], start=1):
                rrf[i] += 1.0 / (RRF_K + rank)
        # Relevance floor: keep chunks reasonably close to the best lexical or dense match.
        top_bm25 = max(bm25_best.values(), default=0.0)
        top_cos = max(cosines.values(), default=0.0)
        has_dense = self._vectors is not None and self.embedder is not None

        def relevant(i: int) -> bool:
            lexical = top_bm25 > 0 and bm25_best[i] >= BM25_FLOOR * top_bm25
            dense = has_dense and cosines[i] >= top_cos - DENSE_MARGIN
            return lexical or dense

        hits = [i for i in allowed if rrf[i] > 0 and relevant(i)]
        hits.sort(key=lambda i: (-rrf[i], self.chunks[i].doc_id, self.chunks[i].ordinal))
        hits = self._expand(hits, allowed)
        return [ScoredChunk(self.chunks[i], rrf[i], bm25_best[i], cosines[i]) for i in hits[:k]]

    def _expand(self, hits: list[int], allowed: list[int]) -> list[int]:
        """Conflict expansion: the best hits are used as queries to pull in passages from
        OTHER documents that state the same rule (an outdated FAQ must arrive together with
        the policy that overrides it, or precedence cannot be applied)."""
        seeds = hits[:EXPANSION_SEEDS]
        if not seeds:
            return hits
        extra: list[int] = []
        for seed in seeds:
            for j in self._neighbours(seed, allowed):
                if j not in hits and j not in extra:
                    extra.append(j)
        return seeds + extra + [i for i in hits[EXPANSION_SEEDS:] if i not in extra]

    def _neighbours(self, seed: int, allowed: list[int]) -> list[int]:
        seed_doc = self.chunks[seed].doc_id
        others = [i for i in allowed if self.chunks[i].doc_id != seed_doc]
        if not others:
            return []
        if self._vectors is not None:
            sims = self._vectors[others] @ self._vectors[seed]
            ranked = sorted(zip(others, sims), key=lambda p: (-p[1], p[0]))
            return [i for i, sim in ranked[:EXPANSION_PER_SEED] if sim >= EXPANSION_MIN_COSINE]
        scores = self._bm25.get_scores(tokenize(self.chunks[seed].index_text))
        floor = EXPANSION_BM25_RATIO * scores[seed]
        ranked = sorted(others, key=lambda i: (-scores[i], i))
        return [i for i in ranked[:EXPANSION_PER_SEED] if scores[i] > 0 and scores[i] >= floor]
