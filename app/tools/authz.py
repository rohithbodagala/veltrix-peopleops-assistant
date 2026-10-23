"""Leave-data access rules (spec 3.1 / 6.1). Identity always comes from the database."""
from __future__ import annotations

from app.data.repo import Employee


class PermissionDenied(Exception):
    """Raised inside a tool when the requester may not access the target."""


def can_view_employee_data(requester: Employee, target: Employee | None) -> bool:
    """Self always; a manager for direct reports; HR for any one existing employee."""
    if target is None:
        return False
    if target.employee_id == requester.employee_id:
        return True
    if requester.role == "hr":
        return True
    if requester.role == "manager":
        return target.manager_id == requester.employee_id
    return False


def can_use_team_leave(requester: Employee) -> bool:
    return requester.role == "manager"
