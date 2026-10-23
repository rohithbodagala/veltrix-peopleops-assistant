"""Section 5.1 precedence: authority, then scope, then recency.

Everything mechanical is computed here so the LLM only has to judge whether two
passages state the same rule. Each document gets a key (authority, scope, date); a
higher key wins, an equal key is an unresolved tie.
"""
from __future__ import annotations

from datetime import date

from app.ingestion.model import DocMeta

LOW_AUTHORITY_TYPES = {"faq", "wiki"}


def authority_class(doc: DocMeta) -> int:
    """1 = policy/addendum/memo/handbook/calendar (and any unknown type), 0 = faq/wiki."""
    return 0 if doc.doc_type in LOW_AUTHORITY_TYPES else 1


def scope_class(doc: DocMeta) -> int:
    """1 = written for specific regions, 0 = GLOBAL."""
    return 0 if "GLOBAL" in doc.regions else 1


def precedence_key(doc: DocMeta) -> tuple[int, int, date]:
    return (authority_class(doc), scope_class(doc), doc.effective_date)


def outranks(a: DocMeta, b: DocMeta) -> bool:
    return precedence_key(a) > precedence_key(b)


def dense_ranks(docs: list[DocMeta]) -> dict[str, int]:
    """Rank 1 = strongest key; documents with identical keys share a rank (a tie group)."""
    keys = sorted({precedence_key(d) for d in docs}, reverse=True)
    position = {k: i + 1 for i, k in enumerate(keys)}
    return {d.doc_id: position[precedence_key(d)] for d in docs}


def describe(doc: DocMeta) -> dict[str, str]:
    return {
        "authority": "HIGH" if authority_class(doc) else "LOW",
        "scope": "GLOBAL" if scope_class(doc) == 0 else "REGIONAL(" + ",".join(doc.regions) + ")",
        "effective": doc.effective_date.isoformat(),
    }
