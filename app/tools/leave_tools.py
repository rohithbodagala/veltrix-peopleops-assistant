"""The five leave tools. Permission checks run here, against the requester taken from the
X-Employee-Id header (ToolContext.requester), never against LLM-chosen arguments."""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone

from app.data.db import write_transaction
from app.data.repo import Employee, EmployeeRepo, LeaveRepo
from app.tools.authz import PermissionDenied, can_use_team_leave, can_view_employee_data
from app.tools.working_days import count_working_days

LEAVE_TYPES = ("annual", "sick")
STATUSES = ("pending", "approved", "cancelled")
LEAVE_YEAR = 2026
NOT_FOUND_REQUEST = "I can't find that request among your leave requests."


class ToolError(Exception):
    """A rule rejected the action; the message is the specific, user-facing reason."""


@dataclass
class ToolContext:
    requester: Employee
    today: date
    holidays: set[date]
    conn: sqlite3.Connection

    @property
    def employees(self) -> EmployeeRepo:
        return EmployeeRepo(self.conn)

    @property
    def leave(self) -> LeaveRepo:
        return LeaveRepo(self.conn)


@dataclass(frozen=True)
class ApplyCheck:
    leave_type: str
    start: date
    end: date
    working_days: int
    balance: float
    balance_after: float


def fmt_days(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text


def _parse_iso(value: str) -> date:
    try:
        return date.fromisoformat(str(value).strip())
    except (TypeError, ValueError):
        raise ToolError("Dates must be given as YYYY-MM-DD.") from None


def _target(ctx: ToolContext, employee_id: str) -> Employee:
    target = ctx.employees.get(str(employee_id).strip().upper())
    if not can_view_employee_data(ctx.requester, target):
        raise PermissionDenied(employee_id)
    return target


# ---------------------------------------------------------------- read tools


def get_leave_balance(ctx: ToolContext, employee_id: str, leave_type: str | None = None) -> dict:
    target = _target(ctx, employee_id)
    balances = ctx.leave.balances(target.employee_id)
    if leave_type:
        if leave_type not in LEAVE_TYPES:
            raise ToolError("leave_type must be 'annual' or 'sick'.")
        balances = {leave_type: balances.get(leave_type, 0.0)}
    return {"employee_id": target.employee_id, "full_name": target.full_name, "balances": balances}


def list_leave_requests(ctx: ToolContext, employee_id: str, status: str | None = None) -> dict:
    target = _target(ctx, employee_id)
    if status and status not in STATUSES:
        raise ToolError("status must be 'pending', 'approved' or 'cancelled'.")
    requests = [r.as_dict() for r in ctx.leave.requests(target.employee_id, status)]
    return {"employee_id": target.employee_id, "full_name": target.full_name, "requests": requests}


def get_team_leave(ctx: ToolContext, start_date: str, end_date: str) -> dict:
    if not can_use_team_leave(ctx.requester):
        raise PermissionDenied("get_team_leave")
    start, end = _parse_iso(start_date), _parse_iso(end_date)
    if end < start:
        raise ToolError("The end date is before the start date.")
    rows = ctx.leave.team_leave(ctx.requester.employee_id, start, end)
    return {"start_date": start.isoformat(), "end_date": end.isoformat(), "leave": rows}


# ---------------------------------------------------------------- write tools


def validate_apply(ctx: ToolContext, leave_type: str, start_date: str, end_date: str) -> ApplyCheck:
    """All rules of spec 6.2, in a fixed order; raises ToolError with the first failure."""
    if leave_type not in LEAVE_TYPES:
        raise ToolError("Leave type must be 'annual' or 'sick'.")
    start, end = _parse_iso(start_date), _parse_iso(end_date)
    if start.year != LEAVE_YEAR or end.year != LEAVE_YEAR:
        raise ToolError(f"Leave can only be requested for dates in {LEAVE_YEAR}.")
    if start < ctx.today:
        raise ToolError(f"The start date {start.isoformat()} is before today ({ctx.today.isoformat()}).")
    if end < start:
        raise ToolError("The end date is before the start date.")
    working_days = count_working_days(start, end, ctx.holidays)
    if working_days == 0:
        raise ToolError(
            f"{start.isoformat()} to {end.isoformat()} has no working days (only weekends or public holidays)."
        )
    return _check_balance_and_overlap(ctx, leave_type, start, end, working_days)


def _check_balance_and_overlap(ctx: ToolContext, leave_type: str, start: date, end: date, working_days: int) -> ApplyCheck:
    balance = ctx.leave.balances(ctx.requester.employee_id).get(leave_type, 0.0)
    if working_days > balance:
        raise ToolError(
            f"This request needs {working_days} working day{'s' if working_days != 1 else ''} but your "
            f"{leave_type} leave balance is {fmt_days(balance)} days."
        )
    overlaps = ctx.leave.active_overlapping(ctx.requester.employee_id, start, end)
    if overlaps:
        o = overlaps[0]
        raise ToolError(
            f"It overlaps your {o.status} {o.leave_type} leave from {o.start_date} to {o.end_date}."
        )
    return ApplyCheck(leave_type, start, end, working_days, balance, balance - working_days)


def apply_leave(ctx: ToolContext, leave_type: str, start_date: str, end_date: str) -> dict:
    check = validate_apply(ctx, leave_type, start_date, end_date)
    request_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    with write_transaction(ctx.conn):
        # Re-check inside the transaction: the data may have changed since the proposal.
        check = _check_balance_and_overlap(ctx, check.leave_type, check.start, check.end, check.working_days)
        ctx.conn.execute(
            "INSERT INTO leave_requests (request_id, employee_id, leave_type, start_date, end_date, "
            "working_days, status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
            (request_id, ctx.requester.employee_id, check.leave_type, check.start.isoformat(),
             check.end.isoformat(), check.working_days, created_at),
        )
        updated = ctx.conn.execute(
            "UPDATE leave_balances SET balance_days = balance_days - ? WHERE employee_id = ? AND leave_type = ?",
            (check.working_days, ctx.requester.employee_id, check.leave_type),
        ).rowcount
        if updated != 1:
            raise ToolError(f"No {check.leave_type} leave balance is recorded for you.")
    return {
        "request_id": request_id,
        "leave_type": check.leave_type,
        "start_date": check.start.isoformat(),
        "end_date": check.end.isoformat(),
        "working_days": check.working_days,
        "status": "pending",
        "balance_after": check.balance_after,
    }


def validate_cancel(ctx: ToolContext, request_id: str):
    """Only the requester's own pending/approved requests that start after today."""
    req = ctx.leave.get_own_request(str(request_id).strip(), ctx.requester.employee_id)
    if req is None:
        raise ToolError(NOT_FOUND_REQUEST)
    if req.status == "cancelled":
        raise ToolError("That request is already cancelled.")
    if req.status not in ("pending", "approved"):
        raise ToolError(f"Only pending or approved requests can be cancelled; this one is {req.status}.")
    if date.fromisoformat(req.start_date) <= ctx.today:
        raise ToolError(
            f"Only requests that start after today ({ctx.today.isoformat()}) can be cancelled; "
            f"this one starts on {req.start_date}."
        )
    return req


def cancel_leave(ctx: ToolContext, request_id: str) -> dict:
    validate_cancel(ctx, request_id)
    with write_transaction(ctx.conn):
        req = validate_cancel(ctx, request_id)
        ctx.conn.execute(
            "UPDATE leave_requests SET status = 'cancelled' WHERE request_id = ? AND employee_id = ?",
            (req.request_id, ctx.requester.employee_id),
        )
        ctx.conn.execute(
            "UPDATE leave_balances SET balance_days = balance_days + ? WHERE employee_id = ? AND leave_type = ?",
            (req.working_days, ctx.requester.employee_id, req.leave_type),
        )
    balance = ctx.leave.balances(ctx.requester.employee_id).get(req.leave_type, 0.0)
    return {
        "request_id": req.request_id,
        "leave_type": req.leave_type,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "working_days": req.working_days,
        "status": "cancelled",
        "balance_after": balance,
    }
