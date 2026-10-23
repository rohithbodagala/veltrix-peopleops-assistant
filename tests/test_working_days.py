"""Working-day calculation and holiday-calendar selection (spec 6.2)."""
from datetime import date

from app.ingestion.model import DocMeta, HolidayEntry
from app.tools.calendar_select import applicable_public_holidays
from app.tools.working_days import count_working_days

D = date.fromisoformat


def test_weekend_only_range_has_zero_working_days():
    assert count_working_days(D("2026-10-24"), D("2026-10-25"), set()) == 0


def test_full_week_monday_to_friday():
    assert count_working_days(D("2026-10-26"), D("2026-10-30"), set()) == 5


def test_range_spanning_a_weekend():
    # Thu 29 Oct .. Tue 3 Nov = Thu, Fri, Mon, Tue
    assert count_working_days(D("2026-10-29"), D("2026-11-03"), set()) == 4


def test_two_weeks_spanning_weekend_is_ten():
    assert count_working_days(D("2026-11-02"), D("2026-11-13"), set()) == 10


def test_holiday_inside_range_is_not_counted():
    assert count_working_days(D("2026-11-09"), D("2026-11-13"), {D("2026-11-09")}) == 4


def test_holiday_on_a_weekend_is_not_subtracted_twice():
    # Sat 15 Aug 2026 is a holiday; the week around it still has 5 working days.
    assert count_working_days(D("2026-08-10"), D("2026-08-16"), {D("2026-08-15")}) == 5


def test_range_spanning_weekend_and_holiday():
    # Thu 26 Nov (holiday) .. Tue 1 Dec with Fri 27 Nov (holiday): Mon 30, Tue 1
    assert count_working_days(D("2026-11-26"), D("2026-12-01"), {D("2026-11-26"), D("2026-11-27")}) == 2


def test_single_day_holiday_has_zero_working_days():
    assert count_working_days(D("2026-12-25"), D("2026-12-25"), {D("2026-12-25")}) == 0


def test_corpus_holidays_extracted_and_restricted_ones_excluded(corpus_index):
    in_2026 = applicable_public_holidays("employee", "IN", D("2026-10-22"), corpus_index.docs, corpus_index.holidays)
    assert D("2026-11-09") in in_2026  # public holiday from the PDF calendar
    assert D("2026-11-24") not in in_2026  # restricted holiday: office stays open
    assert count_working_days(D("2026-11-23"), D("2026-11-27"), in_2026) == 5


def test_regional_calendar_beats_global_and_old_year_does_not_leak(corpus_index):
    in_hols = applicable_public_holidays("employee", "IN", D("2026-10-22"), corpus_index.docs, corpus_index.holidays)
    assert D("2026-12-31") not in in_hols  # GLOBAL company calendar loses to the IN calendar for 2026
    assert D("2025-10-20") in in_hols  # 2025 calendar only contributes 2025 dates
    de_hols = applicable_public_holidays("employee", "DE", D("2026-10-22"), corpus_index.docs, corpus_index.holidays)
    assert de_hols == {D("2026-12-31"), D("2026-12-30")}  # no DE calendar: GLOBAL applies
    us_hols = applicable_public_holidays("employee", "US", D("2026-10-22"), corpus_index.docs, corpus_index.holidays)
    assert {D("2026-11-26"), D("2026-11-27"), D("2026-12-25")} <= us_hols
    assert D("2026-11-09") not in us_hols


def _cal(doc_id, regions, eff):
    from pathlib import Path

    return DocMeta(doc_id, f"{doc_id}.md", doc_id, "calendar", tuple(regions), D(eff), "all", Path("x"))


def test_reissued_calendar_for_same_year_replaces_older_one():
    docs = {"A": _cal("A", ["IN"], "2026-01-01"), "B": _cal("B", ["IN"], "2026-03-01")}
    entries = [
        HolidayEntry("A", ("IN",), D("2026-11-09"), "Old date", "public", "x"),
        HolidayEntry("B", ("IN",), D("2026-11-10"), "New date", "public", "x"),
    ]
    assert applicable_public_holidays("employee", "IN", D("2026-06-01"), docs, entries) == {D("2026-11-10")}
    # before the re-issue is effective, the old calendar applies
    assert applicable_public_holidays("employee", "IN", D("2026-02-01"), docs, entries) == {D("2026-11-09")}
