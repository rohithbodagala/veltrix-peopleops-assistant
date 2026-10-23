"""apply_leave and cancel_leave rules (spec 6.2, 6.3). Today = Thursday 2026-10-22."""
import uuid

import pytest

from app.data.repo import LeaveRepo
from app.tools import leave_tools as tools
from tests import fixtures as fx


def balance(conn, emp, leave_type="annual"):
    return LeaveRepo(conn).balances(emp)[leave_type]


@pytest.mark.parametrize(
    "args,reason",
    [
        (("vacation", "2026-11-02", "2026-11-02"), "annual' or 'sick"),
        (("annual", "2027-01-04", "2027-01-05"), "dates in 2026"),
        (("annual", "2026-12-31", "2027-01-02"), "dates in 2026"),
        (("annual", "2026-10-20", "2026-10-21"), "before today"),
        (("annual", "2026-11-03", "2026-11-02"), "end date is before the start date"),
        (("annual", "2026-10-24", "2026-10-25"), "no working days"),  # weekend
        (("annual", "2026-11-09", "2026-11-09"), "no working days"),  # public holiday (IN)
        (("annual", "2026-11-23", "2026-12-04"), "balance is 7.25 days"),  # 10 days > 7.25
        (("annual", "2026-11-18", "2026-11-19"), "overlaps your approved annual leave"),
        (("sick", "2026-12-08", "2026-12-08"), "overlaps your pending annual leave"),
        (("annual", "bad", "2026-11-02"), "YYYY-MM-DD"),
    ],
)
def test_apply_rejections(tool_ctx, conn, args, reason):
    before = balance(conn, "E2002")
    with pytest.raises(tools.ToolError) as exc:
        tools.apply_leave(tool_ctx("E2002"), *args)
    assert reason in str(exc.value)
    assert balance(conn, "E2002") == before


def test_cancelled_request_does_not_block_new_leave(tool_ctx):
    check = tools.validate_apply(tool_ctx("E2002"), "annual", "2026-12-14", "2026-12-15")
    assert check.working_days == 2


def test_start_today_is_allowed(tool_ctx):
    assert tools.validate_apply(tool_ctx("E2002"), "annual", fx.TODAY, fx.TODAY).working_days == 1


def test_apply_success_creates_pending_request_and_subtracts_balance(tool_ctx, conn):
    # Mon 30 Nov .. Wed 2 Dec = 3 working days; 7.25 - 3 = 4.25
    result = tools.apply_leave(tool_ctx("E2002"), "annual", "2026-11-30", "2026-12-02")
    assert result["working_days"] == 3 and result["status"] == "pending"
    assert uuid.UUID(result["request_id"]).version == 4
    assert balance(conn, "E2002") == pytest.approx(4.25)
    row = conn.execute("SELECT * FROM leave_requests WHERE request_id = ?", (result["request_id"],)).fetchone()
    assert (row["employee_id"], row["status"], row["working_days"]) == ("E2002", "pending", 3)


def test_apply_working_days_skip_holidays(tool_ctx, conn):
    # Mon 9 Nov is a public holiday in IN: Mon 9 .. Fri 13 = 4 days
    result = tools.apply_leave(tool_ctx("E2003"), "sick", "2026-11-09", "2026-11-13")
    assert result["working_days"] == 4
    assert balance(conn, "E2003", "sick") == 8


def test_apply_exactly_the_balance_is_allowed(tool_ctx, conn):
    # E2003 has 2 annual days; Mon 7 + Tue 8 Dec
    tools.apply_leave(tool_ctx("E2003"), "annual", "2026-12-07", "2026-12-08")
    assert balance(conn, "E2003") == 0


def test_apply_is_atomic(tool_ctx, conn, monkeypatch):
    """If the balance update fails after the INSERT, the INSERT is rolled back too."""
    count = conn.execute("SELECT COUNT(*) FROM leave_requests").fetchone()[0]
    conn.execute("DELETE FROM leave_balances WHERE employee_id = 'E2003' AND leave_type = 'annual'")
    monkeypatch.setattr(tools, "_check_balance_and_overlap",
                        lambda c, lt, s, e, wd: tools.ApplyCheck(lt, s, e, wd, 5.0, 5.0 - wd))
    with pytest.raises(tools.ToolError):
        tools.apply_leave(tool_ctx("E2003"), "annual", "2026-12-07", "2026-12-08")
    assert conn.execute("SELECT COUNT(*) FROM leave_requests").fetchone()[0] == count


def test_cancel_future_request_restores_balance(tool_ctx, conn):
    result = tools.cancel_leave(tool_ctx("E2002"), fx.R_ASHA_APPROVED)
    assert result["status"] == "cancelled"
    assert balance(conn, "E2002") == pytest.approx(7.25 + 3)
    status = conn.execute("SELECT status FROM leave_requests WHERE request_id = ?", (fx.R_ASHA_APPROVED,)).fetchone()[0]
    assert status == "cancelled"


def test_cancel_pending_request(tool_ctx, conn):
    tools.cancel_leave(tool_ctx("E2002"), fx.R_ASHA_PENDING)
    assert balance(conn, "E2002") == pytest.approx(9.25)


@pytest.mark.parametrize(
    "employee,request_id,reason",
    [
        ("E2002", fx.R_ASHA_PAST, "start after today"),
        ("E2002", fx.R_ASHA_CANCELLED, "already cancelled"),
        ("E2003", fx.R_ARJUN_TODAY, "start after today"),  # starts today: not after today
        ("E2002", fx.R_ARJUN_PENDING, "can't find that request"),  # someone else's request
        ("E2002", str(uuid.uuid4()), "can't find that request"),  # does not exist
    ],
)
def test_cancel_rejections(tool_ctx, conn, employee, request_id, reason):
    before = balance(conn, employee)
    with pytest.raises(tools.ToolError) as exc:
        tools.cancel_leave(tool_ctx(employee), request_id)
    assert reason in str(exc.value)
    assert balance(conn, employee) == before


def test_foreign_and_unknown_request_ids_get_identical_messages(tool_ctx):
    messages = []
    for rid in (fx.R_BEN_APPROVED, str(uuid.uuid4())):
        with pytest.raises(tools.ToolError) as exc:
            tools.cancel_leave(tool_ctx("E2002"), rid)
        messages.append(str(exc.value))
    assert messages[0] == messages[1]
