"""Section 5.1 eligibility: access level, region and effective date."""
from __future__ import annotations

from datetime import date

from app.ingestion.model import DocMeta

ROLE_RANK = {"employee": 0, "manager": 1, "hr": 2}
ACCESS_RANK = {"all": 0, "managers": 1, "hr": 2}


def can_read(role: str, access_level: str) -> bool:
    """employee: all; manager: all+managers; hr: all+managers+hr. Unknown levels are never readable."""
    if access_level not in ACCESS_RANK:
        return False
    return ROLE_RANK.get(role, 0) >= ACCESS_RANK[access_level]


def region_applies(regions: tuple[str, ...], region: str) -> bool:
    return "GLOBAL" in regions or region.upper() in regions


def is_eligible(doc: DocMeta, role: str, region: str, ref_date: date) -> bool:
    return can_read(role, doc.access_level) and region_applies(doc.regions, region) and doc.effective_date <= ref_date


def is_upcoming(doc: DocMeta, role: str, region: str, ref_date: date) -> bool:
    """Readable, for the requester's region, but not yet in force on ref_date."""
    return can_read(role, doc.access_level) and region_applies(doc.regions, region) and doc.effective_date > ref_date


def readable_doc_ids(docs: dict[str, DocMeta], role: str, region: str) -> set[str]:
    return {d.doc_id for d in docs.values() if can_read(role, d.access_level) and region_applies(d.regions, region)}
