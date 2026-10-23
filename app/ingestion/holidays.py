"""Automatic holiday extraction from calendar documents (manifest doc_type == "calendar").

A table counts as a holiday table when most cells of one column parse as dates. A
table (or its section) labelled "restricted"/"optional" lists days the office stays
open; those are stored with kind "restricted" and never reduce working days.
"""
from __future__ import annotations

import re
from datetime import date, datetime

from dateutil import parser as dateparser

from app.ingestion.model import Block, DocMeta, Heading, HolidayEntry, Paragraph, Table

WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun"}
NOT_CLOSURE = re.compile(r"\b(restricted|optional|floating)\b", re.I)
LINE_DATE = re.compile(
    r"^\s*[•\-*]?\s*(\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})"
    r"\s*(?:\([A-Za-z]+\))?\s*[:\-–—|,]?\s*(.+)$"
)
_DEFAULT = datetime(1900, 1, 1)


def parse_date(text: str) -> date | None:
    text = text.strip()
    if not text or not re.search(r"\d", text) or len(text) > 40:
        return None
    try:
        dt = dateparser.parse(text, dayfirst=True, default=_DEFAULT)
    except (ValueError, OverflowError):
        return None
    if dt.year < 1990 or not re.search(r"\b\d{4}\b", text):
        return None  # a date without an explicit year is not trusted
    return dt.date()


def _date_column(table: Table) -> int | None:
    if not table.rows:
        return None
    width = max(len(r) for r in table.rows)
    best, best_ratio = None, 0.0
    for col in range(width):
        values = [r[col] for r in table.rows if col < len(r) and r[col]]
        if not values:
            continue
        ratio = sum(parse_date(v) is not None for v in values) / len(values)
        if ratio > best_ratio:
            best, best_ratio = col, ratio
    return best if best_ratio >= 0.6 else None


def _name_column(table: Table, date_col: int) -> int | None:
    width = max(len(r) for r in table.rows)
    best, best_len = None, -1.0
    for col in range(width):
        if col == date_col:
            continue
        values = [r[col] for r in table.rows if col < len(r) and r[col]]
        if not values or all(v.strip().lower().rstrip(".") in WEEKDAYS for v in values):
            continue
        avg = sum(len(v) for v in values) / len(values)
        if avg > best_len:
            best, best_len = col, avg
    return best


def extract_holidays(meta: DocMeta, blocks: list[Block]) -> list[HolidayEntry]:
    if meta.doc_type != "calendar":
        return []
    entries: list[HolidayEntry] = []
    section = ""
    title = next((b for b in blocks if isinstance(b, Heading)), None)
    title = title if title is not None and title.level == 1 else None
    for block in blocks:
        if isinstance(block, Heading):
            if block is not title:
                section = block.text
            continue
        locator = section or meta.title
        if isinstance(block, Table):
            date_col = _date_column(block)
            if date_col is None:
                continue
            name_col = _name_column(block, date_col)
            label = " ".join([section] + (block.header or []))
            kind = "restricted" if NOT_CLOSURE.search(label) else "public"
            for row in block.rows:
                if date_col >= len(row):
                    continue
                day = parse_date(row[date_col])
                if day is None:
                    continue
                name = row[name_col] if name_col is not None and name_col < len(row) else ""
                entries.append(HolidayEntry(meta.doc_id, meta.regions, day, name, kind, locator))
        elif isinstance(block, Paragraph):
            m = LINE_DATE.match(block.text.lstrip("• "))
            if m:
                day = parse_date(m.group(1))
                if day is not None:
                    kind = "restricted" if NOT_CLOSURE.search(section + " " + block.text) else "public"
                    entries.append(HolidayEntry(meta.doc_id, meta.regions, day, m.group(2).strip(), kind, locator))
    return entries
