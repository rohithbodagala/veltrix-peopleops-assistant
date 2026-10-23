"""Section 5: eligibility and precedence, including the spec's worked example (5.2)."""
from datetime import date
from pathlib import Path

from app.ingestion.model import DocMeta
from app.retrieval.eligibility import can_read, is_eligible, is_upcoming
from app.retrieval.precedence import dense_ranks, outranks, precedence_key

D = date.fromisoformat


def doc(doc_id, doc_type, regions, eff, access="all"):
    return DocMeta(doc_id, doc_id, doc_id, doc_type, tuple(regions), D(eff), access, Path(doc_id))


# Worked example of spec 5.2
V1 = doc("V1", "policy", ["GLOBAL"], "2025-01-01")
V2 = doc("V2", "policy", ["GLOBAL"], "2026-04-01")
FAQ = doc("FAQ", "faq", ["GLOBAL"], "2026-05-01")
DE = doc("DE", "addendum", ["DE"], "2025-06-01")
V3 = doc("V3", "policy", ["GLOBAL"], "2027-01-01")
ALL = [V1, V2, FAQ, DE, V3]
TODAY = D("2026-09-15")


def eligible(region, ref):
    return [d for d in ALL if is_eligible(d, "employee", region, ref)]


def winner(docs):
    ranks = dense_ranks(docs)
    best = min(ranks.values())
    return sorted(d.doc_id for d in docs if ranks[d.doc_id] == best)


def test_india_employee_gets_v2_and_v3_is_upcoming():
    docs = eligible("IN", TODAY)
    assert {d.doc_id for d in docs} == {"V1", "V2", "FAQ"}  # DE not for region, V3 not yet effective
    assert winner(docs) == ["V2"]  # FAQ loses on authority, V1 on recency
    assert is_upcoming(V3, "employee", "IN", TODAY)
    assert outranks(V3, V2)  # so it is mentioned as an upcoming change


def test_germany_employee_gets_addendum_and_v3_not_mentioned():
    docs = eligible("DE", TODAY)
    assert winner(docs) == ["DE"]  # wins on scope
    assert not outranks(V3, DE)  # the addendum would still win in 2027


def test_india_employee_asking_about_past_date_only_v1():
    docs = eligible("IN", D("2025-08-01"))
    assert [d.doc_id for d in docs] == ["V1"]


def test_authority_beats_scope_and_recency():
    regional_faq = doc("RFAQ", "faq", ["IN"], "2026-08-01")
    assert outranks(V1, regional_faq)


def test_memo_for_region_outranks_global_policy():
    memo = doc("MEMO", "memo", ["IN", "US"], "2026-07-01")
    remote = doc("REMOTE", "policy", ["GLOBAL"], "2025-01-01")
    assert outranks(memo, remote)
    assert not is_eligible(memo, "employee", "DE", TODAY)


def test_same_type_scope_and_date_is_an_unresolved_tie():
    travel = doc("TRAVEL", "policy", ["GLOBAL"], "2026-01-01")
    finance = doc("FIN", "handbook", ["GLOBAL"], "2026-01-01")
    assert precedence_key(travel) == precedence_key(finance)
    assert dense_ranks([travel, finance]) == {"TRAVEL": 1, "FIN": 1}


def test_wiki_and_faq_rank_below_every_other_type():
    for t in ("policy", "addendum", "memo", "handbook", "calendar", "something_new"):
        assert outranks(doc("X", t, ["GLOBAL"], "2020-01-01"), doc("W", "wiki", ["IN"], "2026-01-01"))


def test_role_access_levels():
    assert can_read("employee", "all") and not can_read("employee", "managers") and not can_read("employee", "hr")
    assert can_read("manager", "managers") and not can_read("manager", "hr")
    assert can_read("hr", "hr") and can_read("hr", "managers") and can_read("hr", "all")
    assert not can_read("hr", "board-only")  # unknown access levels are never readable


def test_restricted_future_doc_is_not_upcoming_for_employee():
    future_hr = doc("FHR", "policy", ["GLOBAL"], "2027-01-01", access="hr")
    assert not is_upcoming(future_hr, "employee", "IN", TODAY)
    assert is_upcoming(future_hr, "hr", "IN", TODAY)
