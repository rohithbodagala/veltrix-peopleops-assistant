"""Personal-data helpers shared by logging, tool-call args and answers.

Personal data in this service = an employee's full_name, email and phone.
- Emails are always masked: first character, then ***, then the domain.
- Phone numbers are never shown; they are replaced by a placeholder.
- In logs, known employee names are replaced by their employee_id.
"""
from __future__ import annotations

import re
import threading
from typing import Any

EMAIL_RE = re.compile(r"(?<![\w.%+-])([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})")
# '+'-prefixed international numbers, or US-style 3-3-4 groups. The look-arounds keep
# ISO dates, UUIDs and employee ids from matching.
PHONE_RE = re.compile(
    r"(?<![\w+-])(?:\+\d[\d\s().-]{7,}\d|\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4})(?![\w-])"
)
PHONE_PLACEHOLDER = "[phone number removed]"


def mask_email(match: re.Match) -> str:
    local, domain = match.group(1), match.group(2)
    return f"{local[0]}***@{domain}"


def mask_emails(text: str) -> str:
    return EMAIL_RE.sub(mask_email, text)


def strip_phones(text: str, placeholder: str = PHONE_PLACEHOLDER) -> str:
    def repl(m: re.Match) -> str:
        digits = sum(ch.isdigit() for ch in m.group(0))
        return placeholder if digits >= 9 else m.group(0)

    return PHONE_RE.sub(repl, text)


def scrub_answer(text: str) -> str:
    """Final pass over every answer: mask emails, remove phone numbers."""
    return strip_phones(mask_emails(text))


def scrub_value(value: Any) -> Any:
    """Recursively scrub strings inside tool-call args."""
    if isinstance(value, str):
        return scrub_answer(value)
    if isinstance(value, dict):
        return {k: scrub_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_value(v) for v in value]
    return value


class NameRegistry:
    """Maps employee names to ids so log lines never carry names."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pattern: re.Pattern | None = None
        self._lookup: dict[str, str] = {}

    def load(self, pairs: list[tuple[str, str]]) -> None:
        lookup: dict[str, str] = {}
        for employee_id, full_name in pairs:
            if not full_name:
                continue
            lookup[full_name] = employee_id
            for token in full_name.split():
                if len(token) >= 3 and token not in lookup:
                    lookup[token] = employee_id
        keys = sorted(lookup, key=len, reverse=True)
        pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b") if keys else None
        with self._lock:
            self._lookup = lookup
            self._pattern = pattern

    def redact(self, text: str) -> str:
        with self._lock:
            pattern, lookup = self._pattern, self._lookup
        if pattern is None:
            return text
        return pattern.sub(lambda m: lookup.get(m.group(1), "[name]"), text)


NAMES = NameRegistry()


def redact_for_log(text: str) -> str:
    return NAMES.redact(strip_phones(mask_emails(text), "[phone]"))
