"""HTTP API: POST /chat and GET /health (spec section 9).

Run with: uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import Settings, load_settings
from app.data.db import connection
from app.data.repo import EmployeeRepo
from app.ingestion.pipeline import CorpusIndex, ingest_all
from app.llm.client import LLMClient, build_provider
from app.llm.providers.base import Provider
from app.logging_setup import configure_logging, log_event
from app.orchestration.orchestrator import Orchestrator
from app.orchestration.sessions import SessionOwnershipError, SessionStore
from app.pii import NAMES

logger = logging.getLogger("app.api")
MAX_MESSAGE_CHARS = 4000


class AppState:
    def __init__(self, settings: Settings, provider: Provider | None, corpus: CorpusIndex | None = None) -> None:
        self.settings = settings
        self.provider = provider
        self.prebuilt_corpus = corpus
        self.sessions = SessionStore()
        self.orchestrator: Orchestrator | None = None
        self.ready = threading.Event()
        self.error: str | None = None

    def ingest(self) -> None:
        try:
            with connection(self.settings.db_path) as conn:
                NAMES.load(EmployeeRepo(conn).name_pairs())
            corpus = self.prebuilt_corpus or ingest_all(self.settings)
            llm = LLMClient(self.provider, timeout=self.settings.llm_timeout)
            self.orchestrator = Orchestrator(self.settings, corpus, llm)
            log_event(logger, "service_ready", provider=getattr(self.provider, "name", "none"),
                      model=getattr(self.provider, "model", "none"), today=self.settings.today().isoformat())
            if self.provider is None:
                log_event(logger, "no_llm_provider", logging.WARNING,
                          hint="set LLM_PROVIDER and an API key; only regex-handled messages will work")
            self.ready.set()
        except Exception as exc:  # keep the process alive and observable
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("ingestion_failed")


def create_app(
    settings: Settings | None = None,
    provider: Provider | None = None,
    background: bool = True,
    corpus: CorpusIndex | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings.log_level, settings.log_file)
    state = AppState(settings, provider if provider is not None else build_provider(settings), corpus)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if background:
            threading.Thread(target=state.ingest, name="ingestion", daemon=True).start()
        else:
            state.ingest()
        yield

    app = FastAPI(title="Veltrix PeopleOps Assistant", lifespan=lifespan)
    app.state.assistant = state

    @app.exception_handler(RequestValidationError)
    async def _bad_request(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=400, content={"detail": "invalid request body"})

    @app.get("/health")
    def health():
        if state.ready.is_set():
            return {"status": "ok"}
        return JSONResponse(status_code=503, content={"status": "starting" if state.error is None else "error"})

    @app.post("/chat")
    async def chat(request: Request):
        employee_id = (request.headers.get("x-employee-id") or "").strip()
        if not employee_id:
            return JSONResponse(status_code=401, content={"detail": "missing X-Employee-Id header"})
        with connection(settings.db_path) as conn:
            requester = EmployeeRepo(conn).get(employee_id)
        if requester is None:
            return JSONResponse(status_code=401, content={"detail": "unknown employee"})
        try:
            body = json.loads(await request.body())
        except (ValueError, UnicodeDecodeError):
            return JSONResponse(status_code=400, content={"detail": "body must be JSON"})
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"detail": "body must be a JSON object"})
        session_id, message = body.get("session_id"), body.get("message")
        if not isinstance(session_id, str) or not session_id.strip() or not isinstance(message, str) or not message.strip():
            return JSONResponse(status_code=400, content={"detail": "session_id and message must be non-empty strings"})
        if len(message) > MAX_MESSAGE_CHARS:
            return JSONResponse(status_code=400, content={"detail": f"message longer than {MAX_MESSAGE_CHARS} characters"})
        try:
            session = state.sessions.get_or_create(session_id, requester.employee_id)
        except SessionOwnershipError:
            log_event(logger, "session_owner_mismatch", logging.WARNING, employee_id=requester.employee_id)
            return JSONResponse(status_code=403, content={"detail": "session belongs to another employee"})
        if not state.ready.is_set() or state.orchestrator is None:
            return JSONResponse(status_code=503, content={"detail": "service is starting"})

        request_id = str(uuid.uuid4())

        def run():
            with session.lock:
                return state.orchestrator.handle(requester, session, message.strip(), request_id)

        try:
            result = await anyio.to_thread.run_sync(run)
        except Exception:
            logger.exception("chat_failed", extra={"fields": {"event": "chat_failed", "request_id": request_id}})
            return {
                "request_id": request_id,
                "answer": "Something went wrong while handling your message. Please try again or contact the HR team.",
                "route": "no_answer",
                "citations": [],
                "tool_calls": [],
                "pending_confirmation": False,
            }
        return {
            "request_id": request_id,
            "answer": result.answer,
            "route": result.route,
            "citations": result.citations,
            "tool_calls": [{"name": c.name, "args": c.args, "status": c.status} for c in result.tool_calls],
            "pending_confirmation": result.pending_confirmation,
        }

    return app

