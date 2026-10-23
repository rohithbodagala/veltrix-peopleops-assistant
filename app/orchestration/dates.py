"""Deterministic date handling (spec 8.4).

- Weeks start on Monday. "next week" = Monday-Friday of the following week.
- "next Friday" = the first Friday after today.
- A phrase that is ambiguous, or resolves to a day that has already passed, is
  reported back so the orchestrator can ask instead of guessing.
Explicit calendar dates are taken from the router; this module handles the relative
phrases whose meaning the spec fixes.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}
_WD = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tues?|wed|thu(?:rs?)?|fri|sat|sun)"
_RANGE_SEP = r"(?:\s+(?:to|until|till|through|thru)\s+|\s*[-–—]\s*)"


@dataclass(frozen=True)
class DateResolution:
    status: str  # "ok" | "ambiguous" | "past" | "unrecognized"
    start: date | None = None
    end: date | None = None
    reason: str = ""


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _next_weekday_after(d: date, wd: int) -> date:
    delta = (wd - d.weekday()) % 7 or 7
    return d + timedelta(days=delta)


def _on_or_after(d: date, wd: int) -> date:
    return d + timedelta(days=(wd - d.weekday()) % 7)


def _clean(phrase: str) -> str:
    p = phrase.lower().strip()
    p = re.sub(r"[?.!,]", " ", p)
    p = re.sub(r"\bday\s+after\s+tomorrow\b", "overmorrow", p)
    p = re.sub(r"\bday\s+before\s+yesterday\b", "before yesterday", p)
    p = re.sub(r"\b(from|on|the|of|day|days|please|leave|off|for|instead)\b", " ", p)
    return re.sub(r"\s+", " ", p).strip()


def _single(p: str, today: date) -> DateResolution:
    """One-day expressions."""
    if p in ("yesterday", "before yesterday") or re.fullmatch(rf"last\s+{_WD}|last week|previous week", p):
        return DateResolution("past", reason=f"'{p}' is in the past")
    if p == "today":
        return DateResolution("ok", today, today)
    if p == "tomorrow":
        d = today + timedelta(days=1)
        return DateResolution("ok", d, d)
    if p == "overmorrow":
        d = today + timedelta(days=2)
        return DateResolution("ok", d, d)
    m = re.fullmatch(rf"(next|this|coming)\s+{_WD}", p)
    if m:
        wd = WEEKDAYS[m.group(2)]
        if m.group(1) in ("next", "coming"):
            d = _next_weekday_after(today, wd)
            return DateResolution("ok", d, d)
        d = _monday(today) + timedelta(days=wd)
        if d < today:
            return DateResolution("past", d, d, f"this {m.group(2)} ({d.isoformat()}) has already passed")
        return DateResolution("ok", d, d)
    m = re.fullmatch(rf"{_WD}\s+next\s+week|next\s+week\s+{_WD}", p)
    if m:
        wd = WEEKDAYS[m.group(1) or m.group(2)]
        d = _monday(today) + timedelta(days=7 + wd)
        return DateResolution("ok", d, d)
    m = re.fullmatch(_WD, p)
    if m:
        wd = WEEKDAYS[m.group(1)]
        if wd == today.weekday():
            return DateResolution("ambiguous", reason=f"'{m.group(1)}' could mean today or next week")
        d = _next_weekday_after(today, wd)
        return DateResolution("ok", d, d)
    return DateResolution("unrecognized")


def resolve_phrase(phrase: str, today: date, anchor: date | None = None) -> DateResolution:
    """Resolve a relative date expression. `anchor` is the start date for "until <day>"."""
    p = _clean(phrase)
    if not p:
        return DateResolution("unrecognized")
    if p in ("last week", "previous week"):
        return DateResolution("past", reason=f"'{p}' is in the past")
    if p == "next week":
        mon = _monday(today) + timedelta(days=7)
        return DateResolution("ok", mon, mon + timedelta(days=4))
    if p == "this week":
        mon = _monday(today)
        fri = mon + timedelta(days=4)
        if today > fri:
            return DateResolution("past", mon, fri, "this week's working days have already passed")
        return DateResolution("ok", max(today, mon), fri)
    m = re.fullmatch(rf"(?:next week\s+)?{_WD}{_RANGE_SEP}{_WD}(\s+next week)?", p)
    if m and (p.startswith("next week") or m.group(3)):
        mon = _monday(today) + timedelta(days=7)
        start = mon + timedelta(days=WEEKDAYS[m.group(1)])
        end = mon + timedelta(days=WEEKDAYS[m.group(2)])
        if end < start:
            return DateResolution("ambiguous", reason="the end day is before the start day")
        return DateResolution("ok", start, end)
    m = re.fullmatch(rf"(?:until|till|through|thru|to|up to)\s+(next\s+|this\s+)?{_WD}", p)
    if m:
        base = anchor or today
        wd = WEEKDAYS[m.group(2)]
        if m.group(1) and m.group(1).strip() == "next":
            end = _next_weekday_after(today, wd)
        else:
            end = _on_or_after(base, wd)
        return DateResolution("ok", None, end)
    parts = re.split(_RANGE_SEP, p, maxsplit=1)
    if len(parts) == 2:
        first = _single(parts[0].strip(), today)
        if first.status != "ok":
            return first
        second_raw = parts[1].strip()
        m = re.fullmatch(_WD, second_raw)
        if m:
            end = _on_or_after(first.start, WEEKDAYS[m.group(1)])
            return DateResolution("ok", first.start, end)
        second = _single(second_raw, today)
        if second.status != "ok":
            return second
        if second.start < first.start:
            return DateResolution("ambiguous", reason="the end date is before the start date")
        return DateResolution("ok", first.start, second.start)
    return _single(p, today)


def parse_iso(value: str) -> date | None:
    try:
        return date.fromisoformat(value.strip()) if value and value.strip() else None
    except ValueError:
        return None


def resolve_reference_date(past_reference: str, today: date) -> date:
    """Rules "in force at" a past period = its last day; current/future periods -> today."""
    ref = (past_reference or "").strip()
    try:
        if re.fullmatch(r"\d{4}", ref):
            d = date(int(ref), 12, 31)
        elif re.fullmatch(r"\d{4}-\d{2}", ref):
            y, m = int(ref[:4]), int(ref[5:7])
            d = date(y, m, calendar.monthrange(y, m)[1])
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", ref):
            d = date.fromisoformat(ref)
        else:
            return today
    except ValueError:
        return today
    return d if d < today else today


def month_ends_between(today: date, target: date) -> list[date]:
    """Month-end dates strictly after today and on or before target."""
    out: list[date] = []
    y, m = today.year, today.month
    while True:
        d = date(y, m, calendar.monthrange(y, m)[1])
        if d > target:
            break
        if d > today:
            out.append(d)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def describe_calendar(today: date, days: int = 28) -> str:
    """A small calendar for the router so it can map explicit day names to dates."""
    lines = []
    for i in range(days):
        d = today + timedelta(days=i)
        lines.append(f"{d.isoformat()} {d.strftime('%a')}")
    return ", ".join(lines)
