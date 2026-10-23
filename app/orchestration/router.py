"""Message classification: regex pre-filters for the cases the spec fixes exactly, then
one LLM router call that sees no corpus text."""
from __future__ import annotations

import re
from datetime import date

from app.llm.client import LLMClient
from app.llm.prompts import ROUTER_SYSTEM
from app.llm.schemas import RouterOutput
from app.orchestration.dates import describe_calendar
from app.orchestration.sessions import Proposal
from app.pii import NAMES

_YES = r"(yes|y|yep|yeah|yup|sure|ok|okay|confirm|confirmed|go ahead|do it|proceed|submit|submit it|correct|please do|yes please|absolutely|that's right|that is right|sounds good|looks good)"
_POLITE = r"(please|thanks|thank you|go ahead|do it|confirm|proceed)"
CONFIRM_RE = re.compile(rf"^\s*{_YES}(\s*[,!.]?\s*{_YES})*(\s*[,!.]?\s*{_POLITE})*\s*[.!]*\s*$", re.I)
NEGATION_RE = re.compile(
    r"^\s*(no|nope|nah|no thanks|no thank you|don'?t|do not|stop|never ?mind|forget it|abort|not now)\s*[.!]*\s*$", re.I
)
MANIPULATION_RE = re.compile(
    r"(ignore|disregard|forget|override)\s+(all\s+|any\s+|your\s+|the\s+|my\s+)*(previous|prior|above|earlier|system)?\s*"
    r"(instructions|rules|prompt|guidelines)"
    r"|(system|developer|hidden|initial)\s+(prompt|instructions|message)"
    r"|reveal\s+(your|the)\s+(prompt|instructions|rules|configuration)"
    r"|jailbreak|developer mode|\bDAN\b|pretend (to be|you are)|you are now",
    re.I,
)
SKIP_CONFIRM_RE = re.compile(
    r"(don'?t|do not|no need to|without)\s+(ask(ing)?\s+(me\s+)?(to|for)\s+)?confirm|skip\s+(the\s+)?confirm", re.I
)
EMP_ID_RE = re.compile(r"\b(E\d{3,})\b", re.I)
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)


def is_confirmation(message: str) -> bool:
    return bool(CONFIRM_RE.match(message))


def is_negation(message: str) -> bool:
    return bool(NEGATION_RE.match(message))


def is_manipulation(message: str) -> bool:
    return bool(MANIPULATION_RE.search(message))


def wants_to_skip_confirmation(message: str) -> bool:
    return bool(SKIP_CONFIRM_RE.search(message))


def _pending_summary(pending: Proposal | None) -> str:
    if pending is None:
        return "none"
    return f"{pending.kind} " + ", ".join(f"{k}={v}" for k, v in pending.args.items())


def build_router_input(today: date, history: list[tuple[str, str]], pending: Proposal | None, message: str) -> str:
    turns = []
    for role, text in history[-6:]:
        text = NAMES.redact(text)  # earlier turns go to the LLM with ids instead of names
        if role == "assistant" and len(text) > 300:
            text = text[:300] + " ..."
        turns.append(f"{role.capitalize()}: {text}")
    return (
        f"Today: {today.isoformat()} ({today.strftime('%A')}). Dates without a year are in 2026.\n"
        f"Calendar: {describe_calendar(today)}\n"
        f"Recent conversation:\n{chr(10).join(turns) if turns else '(none)'}\n"
        f"Pending proposal: {_pending_summary(pending)}\n"
        f"Latest user message:\n<<<\n{message}\n>>>"
    )


def route(llm: LLMClient, *, today: date, history: list[tuple[str, str]], pending: Proposal | None,
          message: str, deadline: float, request_id: str) -> RouterOutput:
    out: RouterOutput = llm.complete_json(
        kind="router",
        system=ROUTER_SYSTEM,
        user_prefix=build_router_input(today, history, pending, message),
        corpus_blocks=[],
        user_suffix="Return the JSON object.",
        schema_model=RouterOutput,
        allowed_doc_ids=set(),
        deadline=deadline,
        request_id=request_id,
        max_tokens=4000,
    )
    # Deterministic overrides for identifiers present verbatim in the message.
    m = EMP_ID_RE.search(message)
    if m and out.target_kind in ("self", "none", "name") and out.intent in (
        "balance_query", "requests_query", "apply_leave", "cancel_leave", "future_balance"
    ):
        out.target_kind, out.target_value = "employee_id", m.group(1).upper()
    u = UUID_RE.search(message)
    if u:
        out.request_id = u.group(0).lower()
    if not out.standalone_question:
        out.standalone_question = message
    out.search_queries = [q for q in out.search_queries if isinstance(q, str) and q.strip()][:3]
    return out
