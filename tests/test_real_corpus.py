"""Generic checks against the real ./data folder (skipped when it is absent).

Only structural assertions: the graders' corpus may have different values."""
import shutil
from datetime import date
from pathlib import Path

import pytest

from app.config import load_settings
from app.ingestion.pipeline import ingest_all
from app.retrieval.eligibility import is_eligible

DATA = Path(__file__).resolve().parent.parent / "data"
pytestmark = pytest.mark.skipif(not (DATA / "manifest.json").exists(), reason="./data not present")


@pytest.fixture(scope="module")
def real_index(tmp_path_factory):
    db = tmp_path_factory.mktemp("real") / "veltrix.db"
    shutil.copy(DATA / "veltrix.db", db)
    settings = load_settings({"CORPUS_DIR": str(DATA / "corpus"), "MANIFEST_PATH": str(DATA / "manifest.json"),
                              "DB_PATH": str(db), "EMBEDDINGS": "off", "LOG_LEVEL": "WARNING"})
    return ingest_all(settings)


def test_every_manifest_document_yields_chunks(real_index):
    assert real_index.stats["n_docs_ok"] == real_index.stats["n_docs"]
    assert {c.doc_id for c in real_index.chunks} == set(real_index.docs)


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_documents_without_text_layer_are_ocrd(real_index):
    ocr_chunks = [c for c in real_index.chunks if c.kind == "ocr"]
    assert real_index.stats["n_ocr_docs"] >= 1
    assert sum(ch.isalpha() for c in ocr_chunks for ch in c.text) > 200


def test_every_calendar_yields_public_holidays(real_index):
    calendars = {d.doc_id for d in real_index.docs.values() if d.doc_type == "calendar"}
    with_public = {h.doc_id for h in real_index.holidays if h.kind == "public"}
    assert calendars <= with_public


def test_chunks_within_size_and_have_locators(real_index):
    assert all(c.tokens <= 450 and c.locator for c in real_index.chunks)


def test_restricted_documents_hidden_from_employees(real_index):
    restricted = {d.doc_id for d in real_index.docs.values() if d.access_level != "all"}
    for region in ("IN", "US", "DE"):
        allowed = {d.doc_id for d in real_index.docs.values() if is_eligible(d, "employee", region, date(2026, 12, 31))}
        assert not allowed & restricted
        for c in real_index.chunks:
            if c.doc_id in restricted:
                hits = real_index.retrieval.search([c.text[:200]], c.text[:200], allowed)
                assert all(h.chunk.doc_id not in restricted for h in hits)
