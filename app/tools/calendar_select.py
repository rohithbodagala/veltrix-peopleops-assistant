"""Which holiday calendar applies to a requester (Section 5 applied to calendars).

Eligible calendar documents (readable, for the requester's region or GLOBAL, effective
on the reference date) are compared per calendar year: for each year, only the
documents with the strongest (scope, effective_date) key contribute their dates.
So a regional calendar beats a GLOBAL one, a re-issued calendar for the same year
beats the older one, and last year's calendar only contributes last year's dates.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

from app.ingestion.model import DocMeta, HolidayEntry
from app.retrieval.eligibility import is_eligible
from app.retrieval.precedence import scope_class


def applicable_holiday_entries(
    role: str, region: str, ref_date: date, docs: dict[str, DocMeta], entries: list[HolidayEntry], kinds: tuple[str, ...] = ("public",)
) -> list[HolidayEntry]:
    eligible = {
        doc_id
        for doc_id, d in docs.items()
        if d.doc_type == "calendar" and is_eligible(d, role, region, ref_date)
    }
    by_year: dict[int, list[HolidayEntry]] = defaultdict(list)
    for e in entries:
        if e.doc_id in eligible:
            by_year[e.day.year].append(e)
    chosen: list[HolidayEntry] = []
    for year, year_entries in by_year.items():
        keys = {e.doc_id: (scope_class(docs[e.doc_id]), docs[e.doc_id].effective_date) for e in year_entries}
        best = max(keys.values())
        winners = {doc_id for doc_id, k in keys.items() if k == best}
        chosen.extend(e for e in year_entries if e.doc_id in winners and e.kind in kinds)
    return sorted(chosen, key=lambda e: (e.day, e.doc_id))


def applicable_public_holidays(
    role: str, region: str, ref_date: date, docs: dict[str, DocMeta], entries: list[HolidayEntry]
) -> set[date]:
    return {e.day for e in applicable_holiday_entries(role, region, ref_date, docs, entries)}
