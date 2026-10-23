"""Deterministic offline provider for tests and key-less demos (LLM_PROVIDER=fake).

The router is a keyword heuristic; the composer cites every in-force excerpt and
quotes the first one. Tests can script either side per message.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

ISO = re.compile(r"\b(2\d{3}-\d{2}-\d{2})\b")
UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
EMP = re.compile(r"\b(E\d{3,})\b", re.I)


@dataclass
class CapturedCall:
    kind: str
    system: str
    user: str


def latest_message(user: str) -> str:
    m = re.search(r"Latest user message:\n<<<\n(.*)\n>>>", user, re.S)
    return m.group(1) if m else user


def heuristic_route(user: str) -> dict:
    msg = latest_message(user)
    low = msg.lower()
    out: dict = {"standalone_question": msg, "search_queries": [msg]}
    dates = ISO.findall(msg)
    if dates:
        out["start_date"] = dates[0]
        out["end_date"] = dates[1] if len(dates) > 1 else dates[0]
    if "annual" in low or "vacation" in low:
        out["leave_type"] = "annual"
    elif "sick" in low:
        out["leave_type"] = "sick"
    if m := UUID.search(msg):
        out["request_id"] = m.group(0)
    if m := EMP.search(msg):
        out["target_kind"], out["target_value"] = "employee_id", m.group(1).upper()
    if "half day" in low or "half-day" in low:
        out["half_day"] = True
    if any(w in low for w in ("everyone", "all employees", "everybody", "all staff")):
        out["bulk_request"] = True

    if re.search(r"ignore (all |your |the )?(previous|prior)|system prompt|i am (in )?hr\b", low):
        intent = "manipulation"
    elif any(w in low for w in ("python", "capital of", "joke", "weather", "recipe")):
        intent = "out_of_scope"
    elif "phone" in low or "email address" in low:
        intent = "contact_info"
    elif "cancel" in low:
        intent = "cancel_leave"
    elif "instead" in low or low.startswith("make it"):
        intent, out["refers_to_pending"] = "apply_leave", True
    elif re.search(r"\b(apply|book|request|take)\b.*\b(leave|off|vacation)\b", low):
        intent = "apply_leave"
    elif "team" in low and "leave" in low:
        intent = "team_leave_query"
    elif "will i have" in low or "by december" in low:
        intent, out["target_date"] = "future_balance", "2026-12-31"
    elif "balance" in low or "days left" in low:
        intent = "balance_query"
    elif "request" in low:
        intent = "requests_query"
    elif low.strip(" !.") in ("hi", "hello", "thanks", "thank you"):
        intent = "smalltalk"
    else:
        intent = "policy_question"
    out["intent"] = intent
    return out


def heuristic_compose(user: str) -> dict:
    labels = re.findall(r'<excerpt label="(S\d+)"', user)
    upcoming = re.findall(r'<excerpt label="(U\d+)"[^>]*would_outrank="([^"]+)"', user)
    if not labels:
        return {"answer": "", "used_sources": [], "coverage": "none"}
    first = re.search(r'<excerpt label="S1"[^>]*>\n(.*?)\n</excerpt>', user, re.S)
    snippet = first.group(1).strip().splitlines()[-1][:300] if first else ""
    used = labels + [u for u, outranks in upcoming if outranks.strip()]
    return {"answer": f"According to the documents: {snippet}", "used_sources": used, "coverage": "full"}


@dataclass
class FakeProvider:
    router: Callable[[str], dict] = heuristic_route
    composer: Callable[[str], dict] = heuristic_compose
    calls: list[CapturedCall] = field(default_factory=list)
    name: str = "fake"
    model: str = "fake"

    def generate(self, *, kind: str, system: str, user: str, schema: dict, max_tokens: int, timeout: float) -> str:
        self.calls.append(CapturedCall(kind, system, user))
        result = self.router(user) if kind == "router" else self.composer(user)
        return json.dumps(result)
