"""Resolve "whose data?" to one employee, without leaking existence.

The candidate pool is limited to the employees the requester may see: employees see
only themselves, managers themselves and their direct reports, HR everyone. A name or
id outside the pool is "not permitted" for non-HR requesters whether or not that
person exists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.data.repo import Employee, EmployeeRepo
from app.tools.authz import can_view_employee_data

EMP_ID = re.compile(r"^E\d+$", re.I)


@dataclass
class TargetResult:
    status: str  # "self" | "ok" | "not_permitted" | "ambiguous" | "not_found"
    employee: Employee | None = None
    candidates: list[Employee] = field(default_factory=list)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower().replace("'s", ""))


def _matches(query: str, name: str) -> bool:
    q, n = _tokens(query), _tokens(name)
    if not q:
        return False
    return all(any(nt.startswith(qt) for nt in n) for qt in q) and any(qt in n for qt in q)


def _pool(requester: Employee, repo: EmployeeRepo) -> list[Employee]:
    if requester.role == "hr":
        return repo.all()
    if requester.role == "manager":
        return [requester, *repo.direct_reports(requester.employee_id)]
    return [requester]


def resolve_target(kind: str, value: str, requester: Employee, repo: EmployeeRepo) -> TargetResult:
    value = (value or "").strip()
    if kind in ("self", "none", "") or not value:
        return TargetResult("self", requester)
    if kind == "employee_id" or EMP_ID.match(value):
        employee_id = value.upper()
        if employee_id == requester.employee_id:
            return TargetResult("self", requester)
        target = repo.get(employee_id)
        if can_view_employee_data(requester, target):
            return TargetResult("ok", target)
        if requester.role == "hr" and target is None:
            return TargetResult("not_found")
        return TargetResult("not_permitted")
    # A name
    matches = [e for e in _pool(requester, repo) if _matches(value, e.full_name)]
    exact = [e for e in matches if e.full_name.lower() == value.lower()]
    if len(exact) == 1:
        matches = exact
    if len(matches) == 1:
        target = matches[0]
        if target.employee_id == requester.employee_id:
            return TargetResult("self", requester)
        return TargetResult("ok", target)
    if len(matches) > 1:
        return TargetResult("ambiguous", candidates=matches)
    if requester.role == "hr":
        return TargetResult("not_found")
    return TargetResult("not_permitted")
