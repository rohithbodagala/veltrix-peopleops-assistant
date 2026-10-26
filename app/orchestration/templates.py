"""Deterministic answers for tool-only routes (no LLM involved, no extra personal data)."""
from __future__ import annotations

from datetime import date

from app.data.repo import Employee, LeaveRequest
from app.tools.leave_tools import ApplyCheck, fmt_days


def _day(iso: str | date) -> str:
    d = iso if isinstance(iso, date) else date.fromisoformat(iso)
    return f"{d.strftime('%a')} {d.isoformat()}"


def _is_self(result: dict, requester: Employee) -> bool:
    return result["employee_id"] == requester.employee_id


def _label(result: dict) -> str:
    return f"{result['full_name']} ({result['employee_id']})"


def render_balance(result: dict, requester: Employee) -> str:
    balances = result["balances"]
    parts = [f"{t} leave: {fmt_days(v)} days" for t, v in balances.items()]
    if _is_self(result, requester):
        if len(balances) == 1:
            (leave_type, value), = balances.items()
            return f"Your remaining {leave_type} leave balance is {fmt_days(value)} days."
        return "Your remaining leave balances: " + "; ".join(parts) + "." if parts else "No leave balance is recorded for you."
    if not parts:
        return f"No leave balance is recorded for {_label(result)}."
    return f"Remaining leave balance for {_label(result)}: " + "; ".join(parts) + "."


def _request_line(r: dict) -> str:
    return (
        f"- {r['leave_type']} leave {_day(r['start_date'])} to {_day(r['end_date'])}, "
        f"{r['working_days']} working day{'s' if r['working_days'] != 1 else ''}, {r['status']} "
        f"(request ID {r['request_id']})"
    )


def render_requests(result: dict, requester: Employee, status: str | None) -> str:
    requests = result["requests"]
    qualifier = f"{status} " if status else ""
    if _is_self(result, requester):
        if not requests:
            return f"You have no {qualifier}leave requests."
        heading = f"Your {qualifier}leave requests:"
    else:
        if not requests:
            return f"{_label(result)} has no {qualifier}leave requests."
        heading = f"{qualifier}leave requests for {_label(result)}:"
        heading = heading[0].upper() + heading[1:]
    return heading + "\n" + "\n".join(_request_line(r) for r in requests)


def render_team_leave(result: dict) -> str:
    rows = result["leave"]
    period = f"{_day(result['start_date'])} and {_day(result['end_date'])}"
    if not rows:
        return f"None of your direct reports has pending or approved leave between {period}."
    lines = "\n".join(
        f"- {r['full_name']} ({r['employee_id']}): {r['leave_type']} leave {_day(r['start_date'])} to "
        f"{_day(r['end_date'])}, {r['status']}"
        for r in rows
    )
    return f"Direct reports with pending or approved leave between {period}:\n{lines}"


def render_apply_proposal(check: ApplyCheck, holidays: list[date]) -> str:
    excluded = ""
    if holidays:
        excluded = " Public holidays in the range are not counted: " + ", ".join(_day(h) for h in holidays) + "."
    plural = "s" if check.working_days != 1 else ""
    return (
        f"Please confirm this leave request: {check.leave_type} leave from {_day(check.start)} to "
        f"{_day(check.end)}, {check.working_days} working day{plural}.{excluded} Your {check.leave_type} "
        f"balance would go from {fmt_days(check.balance)} to {fmt_days(check.balance_after)} days. "
        "Reply \"yes\" to submit it, or tell me what to change."
    )


def render_apply_result(result: dict) -> str:
    plural = "s" if result["working_days"] != 1 else ""
    return (
        f"Done. Your {result['leave_type']} leave request for {_day(result['start_date'])} to "
        f"{_day(result['end_date'])} ({result['working_days']} working day{plural}) has been submitted "
        f"with status pending. Request ID: {result['request_id']}. Your {result['leave_type']} balance is now "
        f"{fmt_days(result['balance_after'])} days."
    )


def render_cancel_proposal(req: LeaveRequest) -> str:
    plural = "s" if req.working_days != 1 else ""
    return (
        f"Please confirm that you want to cancel your {req.status} {req.leave_type} leave from "
        f"{_day(req.start_date)} to {_day(req.end_date)} ({req.working_days} working day{plural}, request ID "
        f"{req.request_id}). The {req.working_days} day{plural} will be returned to your {req.leave_type} balance. "
        "Reply \"yes\" to cancel it."
    )


def render_cancel_result(result: dict) -> str:
    return (
        f"Done. Your {result['leave_type']} leave from {_day(result['start_date'])} to {_day(result['end_date'])} "
        f"(request ID {result['request_id']}) is cancelled. {result['working_days']} day"
        f"{'s' if result['working_days'] != 1 else ''} returned; your {result['leave_type']} balance is now "
        f"{fmt_days(result['balance_after'])} days."
    )


def render_cancel_choices(requests: list[LeaveRequest]) -> str:
    lines = "\n".join(_request_line(r.as_dict()) for r in requests)
    return f"Which request would you like to cancel?\n{lines}"


def render_employee_choices(candidates: list[Employee]) -> str:
    names = ", ".join(f"{e.full_name} ({e.employee_id})" for e in candidates)
    return f"Which employee do you mean: {names}?"
