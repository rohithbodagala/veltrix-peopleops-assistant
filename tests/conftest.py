from __future__ import annotations

import shutil
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.data.db import connect
from app.data.repo import EmployeeRepo
from app.ingestion.pipeline import ingest_all
from app.llm.providers.fake import FakeProvider, heuristic_compose, heuristic_route
from app.main import create_app
from app.tools.calendar_select import applicable_public_holidays
from app.tools.leave_tools import ToolContext
from tests import fixtures as fx


@pytest.fixture(scope="session")
def corpus_paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus_root")
    corpus_dir, manifest = fx.build_corpus(root)
    return corpus_dir, manifest


@pytest.fixture(scope="session")
def template_db(tmp_path_factory) -> Path:
    return fx.build_db(tmp_path_factory.mktemp("db") / "template.db")


@pytest.fixture
def db_path(template_db, tmp_path) -> Path:
    path = tmp_path / "veltrix.db"
    shutil.copy(template_db, path)
    return path


@pytest.fixture
def settings(corpus_paths, db_path):
    corpus_dir, manifest = corpus_paths
    return load_settings({
        "CORPUS_DIR": str(corpus_dir), "MANIFEST_PATH": str(manifest), "DB_PATH": str(db_path),
        "APP_TODAY": fx.TODAY, "EMBEDDINGS": "off", "LLM_PROVIDER": "fake", "LOG_LEVEL": "WARNING",
    })


@pytest.fixture(scope="session")
def corpus_index(corpus_paths, tmp_path_factory, template_db):
    corpus_dir, manifest = corpus_paths
    db = tmp_path_factory.mktemp("ingest") / "ingest.db"
    shutil.copy(template_db, db)
    settings = load_settings({"CORPUS_DIR": str(corpus_dir), "MANIFEST_PATH": str(manifest), "DB_PATH": str(db),
                              "EMBEDDINGS": "off", "LOG_LEVEL": "WARNING"})
    return ingest_all(settings)


class ScriptedProvider(FakeProvider):
    """Fake LLM whose router output can be scripted per message; records every call."""

    def __init__(self):
        super().__init__()
        self.routes: dict[str, dict] = {}
        self.compose_override = None
        self.router = self._route
        self.composer = self._compose

    def _route(self, user: str) -> dict:
        from app.llm.providers.fake import latest_message

        msg = latest_message(user)
        out = heuristic_route(user)
        out.update(self.routes.get(msg, {}))
        return out

    def _compose(self, user: str) -> dict:
        if self.compose_override is not None:
            return self.compose_override(user)
        return heuristic_compose(user)


@pytest.fixture
def provider():
    return ScriptedProvider()


@pytest.fixture
def client(settings, provider, corpus_index):
    app = create_app(settings, provider=provider, background=False, corpus=corpus_index)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def chat(client):
    def _chat(employee_id: str, message: str, session_id: str = "s1"):
        resp = client.post("/chat", headers={"X-Employee-Id": employee_id},
                           json={"session_id": session_id, "message": message})
        assert resp.status_code == 200, resp.text
        return resp.json()

    return _chat


@pytest.fixture
def conn(db_path):
    c = connect(db_path)
    yield c
    c.close()


@pytest.fixture
def tool_ctx(conn, corpus_index):
    def _ctx(employee_id: str, today: str = fx.TODAY) -> ToolContext:
        emp = EmployeeRepo(conn).get(employee_id)
        holidays = applicable_public_holidays(emp.role, emp.region, date.fromisoformat(today),
                                              corpus_index.docs, corpus_index.holidays)
        return ToolContext(emp, date.fromisoformat(today), holidays, conn)

    return _ctx


def with_today(settings, today: str):
    return replace(settings, app_today=date.fromisoformat(today))
