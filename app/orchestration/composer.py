"""Retrieval + the composer LLM call for document-grounded answers (routes rag/hybrid)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

from app.data.repo import Employee
from app.ingestion.pipeline import CorpusIndex
from app.llm.client import LLMClient
from app.llm.prompts import COMPOSER_SYSTEM
from app.llm.schemas import ComposerOutput, RouterOutput
from app.retrieval.eligibility import is_eligible, is_upcoming
from app.retrieval.packer import PackedContext, pack_context


@dataclass
class ComposedAnswer:
    answer: str
    citations: list[dict]
    coverage: str
    packed: PackedContext
    ref_date: date
    used_labels: list[str] = field(default_factory=list)


def eligible_doc_ids(corpus: CorpusIndex, requester: Employee, ref_date: date) -> set[str]:
    return {d.doc_id for d in corpus.docs.values() if is_eligible(d, requester.role, requester.region, ref_date)}


def upcoming_doc_ids(corpus: CorpusIndex, requester: Employee, ref_date: date) -> set[str]:
    return {d.doc_id for d in corpus.docs.values() if is_upcoming(d, requester.role, requester.region, ref_date)}


def retrieve(corpus: CorpusIndex, requester: Employee, router: RouterOutput, ref_date: date, today: date,
             extra_queries: list[str] | None = None) -> PackedContext:
    queries = [*router.search_queries, *(extra_queries or [])]
    question = router.standalone_question
    allowed = eligible_doc_ids(corpus, requester, ref_date)
    current = corpus.retrieval.search(queries, question, allowed, k=24)
    upcoming = []
    if ref_date >= today:  # upcoming changes are only relevant to questions about now
        future = upcoming_doc_ids(corpus, requester, ref_date)
        if future:
            upcoming = [h for h in corpus.retrieval.search(queries, question, future, k=6) if h.bm25 > 0 or h.cosine >= 0.6]
    return pack_context(current, upcoming, corpus.docs)


def compose(
    llm: LLMClient,
    *,
    corpus: CorpusIndex,
    requester: Employee,
    router: RouterOutput,
    message: str,
    today: date,
    ref_date: date,
    records: dict,
    deadline: float,
    request_id: str,
    extra_queries: list[str] | None = None,
) -> ComposedAnswer:
    packed = retrieve(corpus, requester, router, ref_date, today, extra_queries)
    if not packed.blocks:
        return ComposedAnswer("", [], "none", packed, ref_date)
    allowed = eligible_doc_ids(corpus, requester, ref_date) | upcoming_doc_ids(corpus, requester, ref_date)
    header = (
        f"Employee: {requester.employee_id} (role: {requester.role}, region: {requester.region})\n"
        f"Today: {today.isoformat()}. Reference date for the rules: {ref_date.isoformat()}.\n"
        f"Question: {router.standalone_question}\n"
        f"Original message:\n<<<\n{message}\n>>>\n"
        f"Records: {json.dumps(records, default=str) if records else '(none)'}\n"
        "Document excerpts (S = in force on the reference date, U = upcoming):"
    )
    out: ComposerOutput = llm.complete_json(
        kind="composer",
        system=COMPOSER_SYSTEM,
        user_prefix=header,
        corpus_blocks=packed.blocks,
        user_suffix="Answer the question following your instructions. Return only the JSON object.",
        schema_model=ComposerOutput,
        allowed_doc_ids=allowed,
        deadline=deadline,
        request_id=request_id,
        max_tokens=8000,
    )
    used = [label.strip().upper() for label in out.used_sources if label.strip().upper() in packed.label_to_chunk]
    citations: list[dict] = []
    for label in used:
        chunk = packed.label_to_chunk[label]
        citation = {"doc_id": chunk.doc_id, "locator": chunk.locator}
        if citation not in citations:
            citations.append(citation)
    answer = re.sub(r"\s*\[(?:[SU]\d+(?:,\s*)?)+\]", "", out.answer).strip()
    coverage = out.coverage if citations else "none"
    return ComposedAnswer(answer, citations, coverage, packed, ref_date, used)
