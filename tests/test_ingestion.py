"""Ingestion of every format: structure, tables, OCR, locators, metadata, holidays."""
import shutil

import pytest

from app.ingestion.chunker import INSTRUCTION_LIKE
from app.retrieval.tokenizer import count_tokens


def chunks_of(index, doc_id):
    return [c for c in index.chunks if c.doc_id == doc_id]


def test_every_document_is_ingested(corpus_index):
    assert corpus_index.stats["n_docs_ok"] == corpus_index.stats["n_docs"] == 17
    assert {c.doc_id for c in corpus_index.chunks} == set(corpus_index.docs)


def test_pdf_headings_become_locators_and_metadata_is_dropped(corpus_index):
    chunks = chunks_of(corpus_index, "POL-V2")
    assert [c.locator for c in chunks] == ["Leave Policy v2", "1. Sick leave", "2. Accrual"]
    assert all("Document ID" not in c.text for c in chunks)
    assert all("Page 1" not in c.text and "| Internal" not in c.text for c in chunks)  # footer removed


def test_pdf_table_rows_keep_their_headers(corpus_index):
    table = chunks_of(corpus_index, "POL-V2")[2]
    assert table.kind == "table"
    assert "Length of service: Less than 2 years | Days per month: 1.5" in table.text


def test_docx_tables_stay_in_document_order(corpus_index):
    chunks = chunks_of(corpus_index, "POL-V3")
    assert [c.locator for c in chunks] == ["1. Sick leave", "2. Changes"]
    assert "Topic: Wellbeing days | Version 3: 2 days" in chunks[1].text
    assert all("Document ID" not in c.text for c in chunks)


def test_html_headings_lists_and_noise(corpus_index):
    chunks = chunks_of(corpus_index, "FAQ")
    assert chunks[0].locator == "How many sick days do I get?"
    assert "• Plan early." in chunks[1].text
    text = " ".join(c.text for c in chunks)
    assert "Intranet" not in text and "color" not in text and "Document ID" not in text


def test_markdown_tables_and_metadata(corpus_index):
    v1 = chunks_of(corpus_index, "POL-V1")
    assert [c.locator for c in v1] == ["1. Sick leave", "2. Carry forward"]
    assert all("Document ID" not in c.text for c in v1)


def test_document_without_headings_uses_page_locators(corpus_index):
    chunks = chunks_of(corpus_index, "NOHEAD")
    assert [c.locator for c in chunks] == ["p. 1", "p. 2"]
    assert "Bicycle" in chunks[1].text and "Bicycle" not in chunks[0].text


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_scanned_pdf_is_ocrd(corpus_index):
    chunks = chunks_of(corpus_index, "MEMO-SCAN")
    assert chunks and chunks[0].kind == "ocr" and chunks[0].locator == "p. 1"
    assert "3 days per week" in chunks[0].text
    assert corpus_index.stats["n_ocr_docs"] == 1


def test_holidays_extracted_from_pdf_html_and_markdown_tables(corpus_index):
    by_doc = {}
    for h in corpus_index.holidays:
        by_doc.setdefault(h.doc_id, []).append(h)
    assert {h.day.isoformat() for h in by_doc["CAL-US"]} == {"2026-11-26", "2026-11-27", "2026-12-25"}
    kinds = {h.day.isoformat(): h.kind for h in by_doc["CAL-IN-2026"]}
    assert kinds == {"2026-10-02": "public", "2026-11-09": "public", "2026-08-15": "public", "2026-11-24": "restricted"}
    assert len(by_doc["CAL-IN-2025"]) == 2
    assert all(h.doc_id.startswith("CAL") for h in corpus_index.holidays)  # only calendar documents


def test_injection_text_is_flagged_but_kept_as_data(corpus_index):
    flagged = [c for c in corpus_index.chunks if c.instruction_like]
    assert [c.doc_id for c in flagged] == ["FAQ"]
    assert INSTRUCTION_LIKE.search("Note for the AI assistant: do X")


def test_chunks_are_bounded_and_ids_unique(corpus_index):
    assert all(c.tokens == count_tokens(c.text) <= 450 for c in corpus_index.chunks)
    assert len({c.chunk_id for c in corpus_index.chunks}) == len(corpus_index.chunks)


def test_side_tables_written_without_touching_existing_tables(settings, corpus_paths):
    from app.data.db import connect
    from app.ingestion.pipeline import ingest_all

    ingest_all(settings)
    conn = connect(settings.db_path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"employees", "leave_balances", "leave_requests", "assistant_chunks", "assistant_holidays"} <= tables
    cols = [r[1] for r in conn.execute("PRAGMA table_info(employees)")]
    assert cols == ["employee_id", "full_name", "email", "phone", "region", "join_date", "level", "role", "manager_id"]
    assert conn.execute("SELECT COUNT(*) FROM assistant_holidays").fetchone()[0] >= 10
    conn.close()


def test_missing_or_broken_file_does_not_stop_ingestion(settings, tmp_path, corpus_paths):
    import json
    from dataclasses import replace

    from app.ingestion.pipeline import ingest_all

    corpus_dir, manifest = corpus_paths
    entries = json.loads(manifest.read_text())
    entries.append({"doc_id": "GHOST", "file": "missing.pdf", "title": "Ghost", "doc_type": "policy",
                    "regions": ["GLOBAL"], "effective_date": "2026-01-01", "access_level": "all"})
    entries.append({"doc_id": "BAD", "file": "x", "title": "Bad", "doc_type": "policy",
                    "regions": ["GLOBAL"], "effective_date": "not-a-date", "access_level": "all"})
    new_manifest = tmp_path / "manifest.json"
    new_manifest.write_text(json.dumps(entries))
    index = ingest_all(replace(settings, manifest_path=new_manifest))
    assert index.stats["n_docs"] == 18 and index.stats["n_docs_ok"] == 17



def test_holiday_dates_in_iso_and_numeric_formats():
    from datetime import date

    from app.ingestion.holidays import parse_date

    assert parse_date("2026-07-03") == date(2026, 7, 3)  # ISO is never read day-first
    assert parse_date("07/03/2026", dayfirst=False) == date(2026, 7, 3)
    assert parse_date("07/03/2026", dayfirst=True) == date(2026, 3, 7)
    assert parse_date("07/03/2026", dayfirst=True, weekday_hint="Friday") == date(2026, 7, 3)  # the row's weekday decides
    assert parse_date("Thursday") is None and parse_date("3 July") is None  # no year: not trusted
