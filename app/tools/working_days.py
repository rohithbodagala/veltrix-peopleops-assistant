"""Working days = Monday-Friday dates in the inclusive range that are not public holidays."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterator


def iter_dates(start: date, end: date) -> Iterator[date]:
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def count_working_days(start: date, end: date, holidays: set[date]) -> int:
    return sum(1 for d in iter_dates(start, end) if d.weekday() < 5 and d not in holidays)


def holidays_in_range(start: date, end: date, holidays: set[date]) -> list[date]:
    return [d for d in iter_dates(start, end) if d in holidays]
