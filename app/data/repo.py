"""Repositories over the existing tables. Queries never select email or phone."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Employee:
    employee_id: str
    full_name: str
    region: str
    join_date: str
    level: str
    role: str
    manager_id: str | None


@dataclass(frozen=True)
class LeaveRequest:
    request_id: str
    employee_id: str
    leave_type: str
    start_date: str
    end_date: str
    working_days: int
    status: str
    created_at: str

    def as_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "employee_id": self.employee_id,
            "leave_type": self.leave_type,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "working_days": self.working_days,
            "status": self.status,
        }


_EMP_COLS = "employee_id, full_name, region, join_date, level, role, manager_id"
_REQ_COLS = "request_id, employee_id, leave_type, start_date, end_date, working_days, status, created_at"


def _emp(row: sqlite3.Row) -> Employee:
    return Employee(**{k: row[k] for k in row.keys()})


def _req(row: sqlite3.Row) -> LeaveRequest:
    return LeaveRequest(**{k: row[k] for k in row.keys()})


class EmployeeRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def get(self, employee_id: str) -> Employee | None:
        row = self.conn.execute(f"SELECT {_EMP_COLS} FROM employees WHERE employee_id = ?", (employee_id,)).fetchone()
        return _emp(row) if row else None

    def direct_reports(self, manager_id: str) -> list[Employee]:
        rows = self.conn.execute(
            f"SELECT {_EMP_COLS} FROM employees WHERE manager_id = ? ORDER BY employee_id", (manager_id,)
        ).fetchall()
        return [_emp(r) for r in rows]

    def all(self) -> list[Employee]:
        rows = self.conn.execute(f"SELECT {_EMP_COLS} FROM employees ORDER BY employee_id").fetchall()
        return [_emp(r) for r in rows]

    def name_pairs(self) -> list[tuple[str, str]]:
        return [(r[0], r[1]) for r in self.conn.execute("SELECT employee_id, full_name FROM employees")]


class LeaveRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def balances(self, employee_id: str) -> dict[str, float]:
        rows = self.conn.execute(
            "SELECT leave_type, balance_days FROM leave_balances WHERE employee_id = ? ORDER BY leave_type",
            (employee_id,),
        ).fetchall()
        return {r["leave_type"]: float(r["balance_days"]) for r in rows}

    def requests(self, employee_id: str, status: str | None = None) -> list[LeaveRequest]:
        sql = f"SELECT {_REQ_COLS} FROM leave_requests WHERE employee_id = ?"
        params: list = [employee_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY start_date, request_id"
        return [_req(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_own_request(self, request_id: str, employee_id: str) -> LeaveRequest | None:
        row = self.conn.execute(
            f"SELECT {_REQ_COLS} FROM leave_requests WHERE request_id = ? AND employee_id = ?",
            (request_id, employee_id),
        ).fetchone()
        return _req(row) if row else None

    def active_overlapping(self, employee_id: str, start: date, end: date) -> list[LeaveRequest]:
        rows = self.conn.execute(
            f"SELECT {_REQ_COLS} FROM leave_requests WHERE employee_id = ? "
            "AND status IN ('pending', 'approved') AND start_date <= ? AND end_date >= ? "
            "ORDER BY start_date",
            (employee_id, end.isoformat(), start.isoformat()),
        ).fetchall()
        return [_req(r) for r in rows]

    def cancellable(self, employee_id: str, today: date) -> list[LeaveRequest]:
        rows = self.conn.execute(
            f"SELECT {_REQ_COLS} FROM leave_requests WHERE employee_id = ? "
            "AND status IN ('pending', 'approved') AND start_date > ? ORDER BY start_date",
            (employee_id, today.isoformat()),
        ).fetchall()
        return [_req(r) for r in rows]

    def team_leave(self, manager_id: str, start: date, end: date) -> list[dict]:
        rows = self.conn.execute(
            "SELECT r.request_id, r.employee_id, e.full_name, r.leave_type, r.start_date, r.end_date, "
            "r.working_days, r.status FROM leave_requests r JOIN employees e ON e.employee_id = r.employee_id "
            "WHERE e.manager_id = ? AND r.status IN ('pending', 'approved') "
            "AND r.start_date <= ? AND r.end_date >= ? ORDER BY r.start_date, r.employee_id",
            (manager_id, end.isoformat(), start.isoformat()),
        ).fetchall()
        return [dict(r) for r in rows]
