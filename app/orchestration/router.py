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

NEGATION_RE = re.compile(
    r"^\s*(no|nope|nah|no thanks|no thank you|don'?t|do not|stop|never ?mind|forget it|abort|not now)\s*[.!]*\s*$", re.I
)
# Only unambiguous attempts to change the assistant itself; everything else goes to the router.
MANIPULATION_RE = re.compile(
    r"\b(ignore|disregard|forget)\s+(all\s+|any\s+)?(of\s+)?(your|previous|prior|above|earlier|these|the\s+above|the\s+previous)\s+"
    r"(instructions|rules|prompts?|guidelines|directions)\b"
    r"|\b(system|developer|hidden)\s+(prompt|instructions)\b"
    r"|\breveal\s+(your|the)\s+(system\s+)?(prompt|instructions|configuration)\b"
    r"|\bjailbreak|\bdeveloper mode\b|\bpretend (to be|you are)\b|\byou are now (a|an|in)\b",
    re.I,
)
# A confirmation is a short message made only of these words, containing at least one
# affirmative word. "cancel" affirms only a pending cancellation.
_CONFIRM_WORDS = set(
    """yes yep yeah yup y sure ok okay k confirm confirmed confirming i go ahead please proceed do it submit
    book apply that thats that's is correct right sounds looks good thanks thank you absolutely definitely
    fine great perfect request leave the go-ahead approve approved""".split()
)
_AFFIRMATIVE = {"yes", "yep", "yeah", "yup", "y", "sure", "ok", "okay", "k", "confirm", "confirmed", "confirming",
                "proceed", "ahead", "correct", "absolutely", "definitely", "submit", "book", "apply", "approve", "approved"}
SKIP_CONFIRM_RE = re.compile(
    r"(don'?t|do not|no need to|without)\s+(ask(ing)?\s+(me\s+)?(to|for)\s+)?confirm|skip\s+(the\s+)?confirm", re.I
)
EMP_ID_RE = re.compile(r"\b(E\d{3,})\b", re.I)
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)


def is_confirmation(message: str, pending_kind: str = "") -> bool:
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", message.lower())
    if not words or len(words) > 8:
        return False
    vocabulary = _CONFIRM_WORDS | ({"cancel", "cancellation"} if pending_kind == "cancel_leave" else set())
    affirmative = _AFFIRMATIVE | ({"cancel"} if pending_kind == "cancel_leave" else set())
    return all(w in vocabulary for w in words) and any(w in affirmative for w in words)


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
