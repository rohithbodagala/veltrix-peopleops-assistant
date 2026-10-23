"""Per-message state machine: confirmation handling, intent dispatch, route invariants.

The LLM classifies (router) and writes document-grounded prose (composer). Everything
that touches data or permissions is deterministic code: tools are only ever invoked
here, from the router's reading of the user's message, never from document text.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date

from app.config import Settings
from app.data.db import connection
from app.data.repo import Employee, EmployeeRepo, LeaveRepo
from app.ingestion.pipeline import CorpusIndex
from app.llm.client import CorpusBudgetExceeded, DeadlineExceeded, LLMClient, LLMUnavailable, RestrictedDocumentLeak
from app.llm.schemas import RouterOutput
from app.logging_setup import log_event
from app.orchestration import messages as M
from app.orchestration import templates as T
from app.orchestration.composer import compose
from app.orchestration.dates import month_ends_between, parse_iso, resolve_phrase, resolve_reference_date
from app.orchestration.router import (
    is_confirmation,
    is_manipulation,
    is_negation,
    route,
    wants_to_skip_confirmation,
)
from app.orchestration.sessions import Proposal, Session
from app.orchestration.targets import resolve_target
from app.pii import scrub_answer, scrub_value
from app.retrieval.eligibility import readable_doc_ids
from app.tools import leave_tools as tools
from app.tools.authz import PermissionDenied
from app.tools.calendar_select import applicable_holiday_entries, applicable_public_holidays
from app.tools.working_days import count_working_days, holidays_in_range

logger = logging.getLogger(__name__)
LLM_ERRORS = (LLMUnavailable, DeadlineExceeded, CorpusBudgetExceeded, RestrictedDocumentLeak)


@dataclass
class ToolCall:
    name: str
    args: dict
    status: str  # "pending_confirmation" | "success" | "error"


@dataclass
class ChatResult:
    answer: str
    route: str
    citations: list[dict] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    pending_confirmation: bool = False
    intent: str = ""


@dataclass
class Turn:
    requester: Employee
    session: Session
    message: str
    today: date
    deadline: float
    request_id: str
    conn: sqlite3.Connection
    holidays: set[date]
    calls: list[ToolCall] = field(default_factory=list)

    def ctx(self) -> tools.ToolContext:
        return tools.ToolContext(self.requester, self.today, self.holidays, self.conn)


def _refuse(text: str) -> ChatResult:
    return ChatResult(text, "refuse")


def _clarify(text: str) -> ChatResult:
    return ChatResult(text, "clarify")


class Orchestrator:
    def __init__(self, settings: Settings, corpus: CorpusIndex, llm: LLMClient) -> None:
        self.settings = settings
        self.corpus = corpus
        self.llm = llm

    # ------------------------------------------------------------------ entry point

    def handle(self, requester: Employee, session: Session, message: str, request_id: str) -> ChatResult:
        started = time.monotonic()
        today = self.settings.today()
        with connection(self.settings.db_path) as conn:
            holidays = applicable_public_holidays(
                requester.role, requester.region, today, self.corpus.docs, self.corpus.holidays
            )
            turn = Turn(requester, session, message, today, started + self.settings.request_deadline, request_id, conn, holidays)
            try:
                result = self._handle(turn)
            except PermissionDenied:
                result = _refuse(M.REFUSE_EMPLOYEE_DATA)
            except LLM_ERRORS as exc:
                log_event(logger, "llm_failed", logging.WARNING, request_id=request_id, error=type(exc).__name__)
                result = ChatResult(M.LLM_UNAVAILABLE, "no_answer")
        result = self._finalize(result, requester)
        session.history.append(("user", message))
        session.history.append(("assistant", result.answer))
        log_event(
            logger, "chat_handled", request_id=request_id, employee_id=requester.employee_id, intent=result.intent,
            route=result.route, citations=[c["doc_id"] for c in result.citations],
            tool_calls=[f"{c.name}:{c.status}" for c in result.tool_calls],
            pending_confirmation=result.pending_confirmation, latency_ms=int((time.monotonic() - started) * 1000),
        )
        return result

    def _finalize(self, result: ChatResult, requester: Employee) -> ChatResult:
        """Route invariants (spec 8.1) and the personal-data pass (spec 7.4)."""
        readable = readable_doc_ids(self.corpus.docs, requester.role, requester.region)
        result.citations = [c for c in result.citations if c["doc_id"] in readable]
        if result.route == "rag":
            result.tool_calls = []
            if not result.citations:
                result.route, result.answer = "no_answer", M.NO_ANSWER
        elif result.route == "hybrid":
            if not result.citations:
                result.route = "tool" if result.tool_calls else "no_answer"
            elif not result.tool_calls:
                result.route = "rag"
        if result.route == "tool":
            result.citations = []
            if not result.tool_calls:
                log_event(logger, "route_invariant_fixed", logging.ERROR, route="tool")
                result.route = "clarify"
        if result.route in ("clarify", "no_answer", "refuse"):
            result.citations, result.tool_calls = [], []
        if result.route != "tool":
            result.pending_confirmation = False
        result.answer = scrub_answer(result.answer)
        for call in result.tool_calls:
            call.args = scrub_value(call.args)
        return result

    # ------------------------------------------------------------------ dispatch

    def _handle(self, t: Turn) -> ChatResult:
        session = t.session
        pending = session.pending
        # A proposal lives for exactly one following message: it is executed, replaced by
        # a new proposal, or discarded -- also when that message fails or is rejected.
        session.pending = None
        if pending is not None:
            if is_confirmation(t.message, pending.kind):
                return self._execute(t, pending)
            if is_negation(t.message):
                return _clarify(M.PROPOSAL_DISCARDED)
        if is_manipulation(t.message):
            return ChatResult(M.REFUSE_MANIPULATION, "refuse", intent="manipulation")

        r = route(self.llm, today=t.today, history=list(session.history), pending=pending,
                  message=t.message, deadline=t.deadline, request_id=t.request_id)
        if pending is not None and r.confirms_pending and not self._changes_proposal(r, pending):
            return self._execute(t, pending)
        modifies = pending is not None and r.refers_to_pending and r.intent == pending.kind
        result = self._dispatch(t, r, pending if modifies else None)
        result.intent = r.intent
        return result

    @staticmethod
    def _changes_proposal(r: RouterOutput, pending: Proposal) -> bool:
        if pending.kind == "apply_leave":
            return bool(r.start_date or r.end_date or r.date_phrase or r.half_day
                        or (r.leave_type and r.leave_type != pending.args.get("leave_type")))
        return bool(r.request_id and r.request_id != pending.args.get("request_id"))

    def _dispatch(self, t: Turn, r: RouterOutput, pending: Proposal | None) -> ChatResult:
        intent = r.intent
        if intent == "manipulation":
            return _refuse(M.REFUSE_MANIPULATION)
        if intent == "out_of_scope":
            return _refuse(M.REFUSE_OUT_OF_SCOPE)
        if intent == "contact_info":
            return _refuse(M.REFUSE_CONTACT)
        if intent == "smalltalk":
            return _clarify(M.GREETING)
        if r.bulk_request and intent != "team_leave_query":
            return _refuse(M.REFUSE_BULK)
        if intent in ("balance_query", "requests_query"):
            return self._employee_data(t, r)
        if intent == "team_leave_query":
            return self._team_leave(t, r)
        if intent == "apply_leave":
            return self._apply(t, r, pending)
        if intent == "cancel_leave":
            return self._cancel(t, r, pending)
        if intent == "future_balance":
            return self._future_balance(t, r)
        return self._policy(t, r)

    # ------------------------------------------------------------------ helpers

    def _record(self, t: Turn, name: str, args: dict, status: str) -> None:
        t.calls.append(ToolCall(name, dict(args), status))

    def _run_read_tool(self, t: Turn, name: str, args: dict):
        fn = {"get_leave_balance": tools.get_leave_balance, "list_leave_requests": tools.list_leave_requests,
              "get_team_leave": tools.get_team_leave}[name]
        try:
            result = fn(t.ctx(), **args)
        except tools.ToolError as exc:
            self._record(t, name, args, "error")
            return None, str(exc)
        self._record(t, name, args, "success")
        return result, None

    def _composed(self, t: Turn, r: RouterOutput, ref_date: date, records: dict, extra_queries=None):
        return compose(self.llm, corpus=self.corpus, requester=t.requester, router=r, message=t.message,
                       today=t.today, ref_date=ref_date, records=records, deadline=t.deadline,
                       request_id=t.request_id, extra_queries=extra_queries)

    def _dates(self, t: Turn, r: RouterOutput, base_start: date | None = None, base_end: date | None = None):
        """Returns (start, end, problem); problem is a clarification text or None.

        Relative phrases with a fixed meaning (spec 8.4) are resolved here and override
        the router's reading; a phrase that is ambiguous or resolves to a past day
        produces a clarification. Explicit calendar dates come from the router; if they
        are in the past, the tool rejects them with the specific reason. Unspecified
        parts keep the values of the pending proposal (base_*), if any."""
        start = parse_iso(r.start_date) or base_start
        end = parse_iso(r.end_date) or base_end
        if r.date_phrase:
            res = resolve_phrase(r.date_phrase, t.today, anchor=parse_iso(r.start_date) or base_start)
            if res.status == "ok":
                if res.start is not None:
                    start = res.start
                    end = res.end
                else:  # "until Friday": only the end changes
                    end = res.end
            elif res.status == "past":
                return None, None, f"That date has already passed ({res.reason}). Which dates would you like?"
            elif res.status == "ambiguous":
                return None, None, f"I'm not sure which dates you mean ({res.reason}). Please give the exact dates."
        if r.dates_ambiguous:
            return None, None, "I'm not sure which dates you mean. Please give the exact start and end dates."
        return start, end, None

    # ------------------------------------------------------------------ leave data (read)

    def _employee_data(self, t: Turn, r: RouterOutput) -> ChatResult:
        target = resolve_target(r.target_kind, r.target_value, t.requester, EmployeeRepo(t.conn))
        if target.status == "not_permitted":
            return _refuse(M.REFUSE_EMPLOYEE_DATA)
        if target.status == "ambiguous":
            return _clarify(T.render_employee_choices(target.candidates))
        if target.status == "not_found":
            return _clarify("I couldn't find that employee. Could you give their employee ID?")
        employee_id = target.employee.employee_id
        if r.intent == "balance_query":
            args = {"employee_id": employee_id}
            if r.leave_type:
                args["leave_type"] = r.leave_type
            result, error = self._run_read_tool(t, "get_leave_balance", args)
            text = T.render_balance(result, t.requester) if result else error
        else:
            args = {"employee_id": employee_id}
            if r.status_filter:
                args["status"] = r.status_filter
            result, error = self._run_read_tool(t, "list_leave_requests", args)
            text = T.render_requests(result, t.requester, r.status_filter or None) if result else error
        if result and r.needs_documents:
            records = {"tool_result": {k: v for k, v in result.items() if k != "full_name"}}
            composed = self._composed(t, r, t.today, records)
            if composed.citations:
                return ChatResult(composed.answer, "hybrid", composed.citations, t.calls)
        return ChatResult(text, "tool", tool_calls=t.calls)

    def _team_leave(self, t: Turn, r: RouterOutput) -> ChatResult:
        if t.requester.role != "manager":
            return _refuse(M.REFUSE_TEAM_NOT_MANAGER)
        start, end, problem = self._dates(t, r)
        if problem:
            return _clarify(problem)
        if not start or not end:
            return _clarify("For which dates would you like to see your team's leave? Please give a start and end date.")
        result, error = self._run_read_tool(t, "get_team_leave", {"start_date": start.isoformat(), "end_date": end.isoformat()})
        return ChatResult(T.render_team_leave(result) if result else error, "tool", tool_calls=t.calls)

    # ------------------------------------------------------------------ writes: propose

    def _acting_for_other(self, t: Turn, r: RouterOutput) -> bool:
        if r.target_kind not in ("employee_id", "name") or not r.target_value:
            return False
        target = resolve_target(r.target_kind, r.target_value, t.requester, EmployeeRepo(t.conn))
        return target.status != "self"

    def _apply(self, t: Turn, r: RouterOutput, pending: Proposal | None) -> ChatResult:
        if self._acting_for_other(t, r):
            return _refuse(M.REFUSE_ACT_FOR_OTHERS)
        if r.half_day:
            return _refuse(M.REFUSE_HALF_DAY)
        base = pending.args if pending is not None and pending.kind == "apply_leave" else {}
        leave_type = r.leave_type or base.get("leave_type", "")
        start, end, problem = self._dates(t, r, parse_iso(base.get("start_date", "")), parse_iso(base.get("end_date", "")))
        missing = []
        if not leave_type:
            missing.append("which type of leave it is (annual or sick)")
        if problem:
            return _clarify(problem + (f" Also, please tell me {missing[0]}." if missing else ""))
        if not start and not end:
            missing.append("the dates (first and last day of the leave)")
        elif not start:
            missing.append("the first day of the leave")
        elif not end:
            missing.append("the last day of the leave")
        if missing:
            return _clarify("To request leave I need " + " and ".join(missing) + ".")

        args = {"leave_type": leave_type, "start_date": start.isoformat(), "end_date": end.isoformat()}
        try:
            check = tools.validate_apply(t.ctx(), **args)
        except tools.ToolError as exc:
            self._record(t, "apply_leave", args, "error")
            return ChatResult(f"I can't request that leave: {exc}", "tool", tool_calls=t.calls)
        text = T.render_apply_proposal(check, holidays_in_range(check.start, check.end, t.holidays))
        if wants_to_skip_confirmation(t.message):
            text = "I always need your confirmation before submitting leave. " + text
        t.session.pending = Proposal("apply_leave", args, text, {"working_days": check.working_days})
        self._record(t, "apply_leave", args, "pending_confirmation")
        return ChatResult(text, "tool", tool_calls=t.calls, pending_confirmation=True)

    def _cancel(self, t: Turn, r: RouterOutput, pending: Proposal | None) -> ChatResult:
        if self._acting_for_other(t, r):
            return _refuse(M.REFUSE_ACT_FOR_OTHERS)
        leave = LeaveRepo(t.conn)
        request_id = r.request_id.strip()
        if not request_id:
            candidates = leave.cancellable(t.requester.employee_id, t.today)
            if not candidates:
                result, _ = self._run_read_tool(t, "list_leave_requests", {"employee_id": t.requester.employee_id})
                return ChatResult(
                    "You have no pending or approved leave requests that start after today, so there is nothing "
                    "I can cancel.", "tool", tool_calls=t.calls)
            narrowed = self._narrow(t, r, candidates)
            if len(narrowed) == 1:
                request_id = narrowed[0].request_id
            else:
                return _clarify(T.render_cancel_choices(narrowed or candidates))
        args = {"request_id": request_id}
        try:
            req = tools.validate_cancel(t.ctx(), request_id)
        except tools.ToolError as exc:
            self._record(t, "cancel_leave", args, "error")
            return ChatResult(f"I can't cancel that request: {exc}", "tool", tool_calls=t.calls)
        text = T.render_cancel_proposal(req)
        if wants_to_skip_confirmation(t.message):
            text = "I always need your confirmation before cancelling leave. " + text
        t.session.pending = Proposal("cancel_leave", args, text)
        self._record(t, "cancel_leave", args, "pending_confirmation")
        return ChatResult(text, "tool", tool_calls=t.calls, pending_confirmation=True)

    def _narrow(self, t: Turn, r: RouterOutput, candidates):
        hint_dates = [d for d in (parse_iso(r.start_date), parse_iso(r.end_date), parse_iso(r.cancel_hint)) if d]
        if r.date_phrase and not hint_dates:
            res = resolve_phrase(r.date_phrase, t.today)
            if res.status == "ok" and res.start:
                hint_dates = [res.start, res.end or res.start]
        out = candidates
        if r.leave_type:
            out = [c for c in out if c.leave_type == r.leave_type]
        if hint_dates:
            lo, hi = min(hint_dates), max(hint_dates)
            out = [c for c in out if date.fromisoformat(c.start_date) <= hi and date.fromisoformat(c.end_date) >= lo]
        return out

    # ------------------------------------------------------------------ writes: execute

    def _execute(self, t: Turn, proposal: Proposal) -> ChatResult:
        ctx = t.ctx()
        try:
            if proposal.kind == "apply_leave":
                result = tools.apply_leave(ctx, **proposal.args)
                text = T.render_apply_result(result)
            else:
                result = tools.cancel_leave(ctx, **proposal.args)
                text = T.render_cancel_result(result)
        except tools.ToolError as exc:
            self._record(t, proposal.kind, proposal.args, "error")
            verb = "submit that leave request" if proposal.kind == "apply_leave" else "cancel that request"
            return ChatResult(f"I couldn't {verb}: {exc}", "tool", tool_calls=t.calls, intent=proposal.kind)
        self._record(t, proposal.kind, proposal.args, "success")
        return ChatResult(text, "tool", tool_calls=t.calls, intent=proposal.kind)

    # ------------------------------------------------------------------ documents

    def _policy(self, t: Turn, r: RouterOutput) -> ChatResult:
        ref_date = resolve_reference_date(r.past_reference, t.today)
        records: dict = {}
        start, end = parse_iso(r.start_date), parse_iso(r.end_date) or parse_iso(r.start_date)
        if start and end and end >= start and (end - start).days <= 370:
            entries = applicable_holiday_entries(t.requester.role, t.requester.region, ref_date,
                                                 self.corpus.docs, self.corpus.holidays, ("public", "restricted"))
            public = {e.day for e in entries if e.kind == "public"}
            records["date_range"] = {
                "start": start.isoformat(), "end": end.isoformat(),
                "working_days_for_this_employee": count_working_days(start, end, public),
                "public_holidays_in_range": [f"{e.day.isoformat()} {e.name}" for e in entries if e.kind == "public" and start <= e.day <= end],
                "restricted_holidays_in_range": [f"{e.day.isoformat()} {e.name}" for e in entries if e.kind == "restricted" and start <= e.day <= end],
            }
        composed = self._composed(t, r, ref_date, records)
        if composed.coverage == "none" or not composed.citations:
            return ChatResult(M.NO_ANSWER, "no_answer")
        return ChatResult(composed.answer, "rag", composed.citations)

    def _future_balance(self, t: Turn, r: RouterOutput) -> ChatResult:
        who = resolve_target(r.target_kind, r.target_value, t.requester, EmployeeRepo(t.conn))
        if who.status == "not_permitted":
            return _refuse(M.REFUSE_EMPLOYEE_DATA)
        if who.status == "ambiguous":
            return _clarify(T.render_employee_choices(who.candidates))
        if who.status == "not_found":
            return _clarify("I couldn't find that employee. Could you give their employee ID?")
        employee = who.employee
        args = {"employee_id": employee.employee_id}
        if r.leave_type:
            args["leave_type"] = r.leave_type
        result, error = self._run_read_tool(t, "get_leave_balance", args)
        if result is None:
            return ChatResult(error, "tool", tool_calls=t.calls)
        target = parse_iso(r.target_date) or date(t.today.year, 12, 31)
        join = date.fromisoformat(employee.join_date)
        months = (t.today.year - join.year) * 12 + (t.today.month - join.month) - (1 if t.today.day < join.day else 0)
        records = {
            "employee_id": employee.employee_id,
            "employee_region": employee.region,
            "current_balance_days": result["balances"],
            "balance_note": "current remaining balance; pending and approved requests are already subtracted",
            "join_date": join.isoformat(),
            "length_of_service_today": f"{months // 12} years {months % 12} months",
            "projection_target_date": target.isoformat(),
            "month_end_dates_after_today_until_target": [d.isoformat() for d in month_ends_between(t.today, target)],
        }
        composed = self._composed(t, r, t.today, records, extra_queries=["annual leave accrual per month", "carry forward unused annual leave"])
        if composed.citations and composed.coverage != "none":
            return ChatResult(composed.answer, "hybrid", composed.citations, t.calls)
        text = T.render_balance(result, t.requester) + " The HR documents don't let me project it further."
        return ChatResult(text, "tool", tool_calls=t.calls)
