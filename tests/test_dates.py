"""Date phrases (spec 8.4). Today = Thursday 2026-10-22."""
from datetime import date

import pytest

from app.orchestration.dates import month_ends_between, resolve_phrase, resolve_reference_date

TODAY = date(2026, 10, 22)
D = date.fromisoformat


@pytest.mark.parametrize(
    "phrase,start,end",
    [
        ("tomorrow", "2026-10-23", "2026-10-23"),
        ("next week", "2026-10-26", "2026-10-30"),
        ("next Friday", "2026-10-23", "2026-10-23"),  # first Friday after today
        ("next Thursday", "2026-10-29", "2026-10-29"),
        ("this Friday", "2026-10-23", "2026-10-23"),
        ("Monday", "2026-10-26", "2026-10-26"),
        ("Monday to Wednesday next week", "2026-10-26", "2026-10-28"),
        ("from Tuesday to Thursday next week", "2026-10-27", "2026-10-29"),
        ("Wednesday next week", "2026-10-28", "2026-10-28"),
        ("tomorrow to next Tuesday", "2026-10-23", "2026-10-27"),
        ("Monday to Friday", "2026-10-26", "2026-10-30"),
    ],
)
def test_relative_phrases(phrase, start, end):
    res = resolve_phrase(phrase, TODAY)
    assert res.status == "ok", res
    assert (res.start, res.end) == (D(start), D(end))


def test_until_weekday_uses_the_anchor_start():
    res = resolve_phrase("until Friday instead", TODAY, anchor=D("2026-11-02"))
    assert res.status == "ok" and res.start is None and res.end == D("2026-11-06")


def test_this_weekday_already_passed_is_reported():
    assert resolve_phrase("this Monday", TODAY).status == "past"


def test_bare_weekday_equal_to_today_is_ambiguous():
    assert resolve_phrase("Thursday", TODAY).status == "ambiguous"


def test_explicit_dates_are_left_to_the_router():
    assert resolve_phrase("3 to 5 November", TODAY).status == "unrecognized"
    assert resolve_phrase("2026-11-03", TODAY).status == "unrecognized"


def test_reference_date_for_past_periods():
    assert resolve_reference_date("2025", TODAY) == D("2025-12-31")
    assert resolve_reference_date("2026-03", TODAY) == D("2026-03-31")
    assert resolve_reference_date("2025-08-01", TODAY) == D("2025-08-01")
    assert resolve_reference_date("2026", TODAY) == TODAY  # current year -> today
    assert resolve_reference_date("2027", TODAY) == TODAY  # future -> today (upcoming changes)
    assert resolve_reference_date("", TODAY) == TODAY


def test_month_ends_between():
    assert month_ends_between(TODAY, D("2026-12-31")) == [D("2026-10-31"), D("2026-11-30"), D("2026-12-31")]
    assert month_ends_between(D("2026-10-31"), D("2026-11-30")) == [D("2026-11-30")]
