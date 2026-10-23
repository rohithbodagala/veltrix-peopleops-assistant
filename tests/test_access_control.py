"""Access control per role, for documents (retrieval) and leave data (tools)."""
from datetime import date

import pytest

from app.data.repo import EmployeeRepo
from app.orchestration.targets import resolve_target
from app.retrieval.eligibility import is_eligible
from app.tools import leave_tools as tools
from app.tools.authz import PermissionDenied

TODAY = date(2026, 10, 22)


def allowed_docs(corpus_index, role, region, ref=TODAY):
    return {d.doc_id for d in corpus_index.docs.values() if is_eligible(d, role, region, ref)}


# ---------------------------------------------------------------- documents


@pytest.mark.parametrize(
    "role,can_mgr,can_hr", [("employee", False, False), ("manager", True, False), ("hr", True, True)]
)
def test_document_access_by_role(corpus_index, role, can_mgr, can_hr):
    docs = allowed_docs(corpus_index, role, "IN")
    assert ("HB-MGR" in docs) is can_mgr
    assert ("HR-COMP" in docs) is can_hr
    assert "FAQ" in docs and "POL-V2" in docs


def test_restricted_chunks_never_returned_by_search_even_on_exact_match(corpus_index):
    for role, secret in (("employee", "zebracode"), ("employee", "quokkaband"), ("manager", "quokkaband")):
        hits = corpus_index.retrieval.search([secret], secret, allowed_docs(corpus_index, role, "IN"), k=50)
        assert all(h.chunk.doc_id not in ("HR-COMP",) + (("HB-MGR",) if role == "employee" else ()) for h in hits)
        assert all(secret not in h.chunk.text for h in hits)


def test_restricted_chunks_are_found_for_roles_that_may_read_them(corpus_index):
    hits = corpus_index.retrieval.search(["quokkaband"], "quokkaband", allowed_docs(corpus_index, "hr", "IN"))
    assert hits and hits[0].chunk.doc_id == "HR-COMP"


def test_regional_documents_filtered_by_region(corpus_index):
    assert "ADD-DE" not in allowed_docs(corpus_index, "employee", "IN")
    assert "ADD-DE" in allowed_docs(corpus_index, "employee", "DE")
    assert "MEMO-SCAN" not in allowed_docs(corpus_index, "employee", "DE")


# ---------------------------------------------------------------- leave data


@pytest.mark.parametrize(
    "requester,target,allowed",
    [
        ("E2002", "E2002", True),  # employee: self
        ("E2002", "E2003", False),  # employee: colleague
        ("E2002", "E2001", False),  # employee: own manager
        ("E2001", "E2002", True),  # manager: direct report
        ("E2001", "E2001", True),  # manager: self
        ("E2001", "E3002", False),  # manager: someone else's report
        ("E2001", "E1001", False),  # manager: HR person
        ("E1001", "E3002", True),  # HR: any one employee
        ("E1001", "E4001", True),
    ],
)
def test_balance_and_requests_permissions(tool_ctx, requester, target, allowed):
    ctx = tool_ctx(requester)
    for fn in (tools.get_leave_balance, tools.list_leave_requests):
        if allowed:
            assert fn(ctx, target)["employee_id"] == target
        else:
            with pytest.raises(PermissionDenied):
                fn(ctx, target)


def test_nonexistent_employee_is_denied_like_a_forbidden_one(tool_ctx):
    with pytest.raises(PermissionDenied):
        tools.get_leave_balance(tool_ctx("E2002"), "E9999")


def test_team_leave_only_for_managers_and_only_direct_reports(tool_ctx):
    rows = tools.get_team_leave(tool_ctx("E2001"), "2026-11-01", "2026-11-30")["leave"]
    assert {r["employee_id"] for r in rows} == {"E2002", "E2003"}
    assert all(r["status"] in ("pending", "approved") for r in rows)
    for who in ("E2002", "E1001"):  # employee and HR are not managers
        with pytest.raises(PermissionDenied):
            tools.get_team_leave(tool_ctx(who), "2026-11-01", "2026-11-30")


def test_team_leave_excludes_cancelled_and_non_overlapping(tool_ctx):
    rows = tools.get_team_leave(tool_ctx("E2001"), "2026-12-14", "2026-12-15")["leave"]
    assert rows == []  # only Asha's cancelled request overlaps


def test_tool_results_never_contain_email_or_phone(tool_ctx):
    result = tools.get_leave_balance(tool_ctx("E1001"), "E2002")
    team = tools.get_team_leave(tool_ctx("E2001"), "2026-11-01", "2026-12-31")
    text = repr(result) + repr(team)
    assert "@" not in text and "+91" not in text


def test_name_resolution_never_leaks_existence(conn):
    repo = EmployeeRepo(conn)
    employee = repo.get("E2002")
    manager = repo.get("E2001")
    hr = repo.get("E1001")
    # employee: any other person, existing or not -> not_permitted
    assert resolve_target("name", "Arjun", employee, repo).status == "not_permitted"
    assert resolve_target("name", "Nobody Atall", employee, repo).status == "not_permitted"
    assert resolve_target("name", "Asha Kumar", employee, repo).status == "self"
    # manager: direct report ok; outsider and unknown both not_permitted
    assert resolve_target("name", "Arjun", manager, repo).employee.employee_id == "E2003"
    assert resolve_target("name", "Ben Ford", manager, repo).status == "not_permitted"
    assert resolve_target("name", "Nobody", manager, repo).status == "not_permitted"
    assert resolve_target("employee_id", "E3002", manager, repo).status == "not_permitted"
    # HR: anyone; ambiguous first names ask; unknown is not found
    assert resolve_target("name", "Ben", hr, repo).employee.employee_id == "E3002"
    ambiguous = resolve_target("name", "Asha", hr, repo)
    assert ambiguous.status == "ambiguous" and {e.employee_id for e in ambiguous.candidates} == {"E2002", "E2004"}
    assert resolve_target("name", "Nobody", hr, repo).status == "not_found"
