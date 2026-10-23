"""Ingestion: manifest + corpus files -> chunks, holidays and a retrieval index.

Runs at every start-up (a fresh ./data may be mounted). Derived data is kept in memory
and mirrored best-effort into assistant_* side tables; a failure to write them never
blocks readiness.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.data.db import connection, ensure_side_tables, write_transaction
from app.ingestion.chunker import build_chunks, split_metadata
from app.ingestion.holidays import extract_holidays
from app.ingestion.manifest import load_manifest
from app.ingestion.model import Chunk, DocMeta, HolidayEntry, ParsedDoc
from app.ingestion.parsers.docx import parse_docx
from app.ingestion.parsers.html import parse_html
from app.ingestion.parsers.md import parse_md
from app.ingestion.parsers.ocr import ocr_image_bytes, ocr_text_to_paragraphs
from app.ingestion.parsers.pdf import parse_pdf
from app.logging_setup import log_event
from app.retrieval.embeddings import Embedder, load_embedder
from app.retrieval.index import RetrievalIndex

logger = logging.getLogger(__name__)


@dataclass
class CorpusIndex:
    docs: dict[str, DocMeta]
    chunks: list[Chunk]
    holidays: list[HolidayEntry]
    retrieval: RetrievalIndex
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.chunks_by_id = {c.chunk_id: c for c in self.chunks}


def _parse_image(path: Path) -> ParsedDoc:
    return ParsedDoc(blocks=list(ocr_text_to_paragraphs(ocr_image_bytes(path.read_bytes()), page=1)), ocr_used=True)


PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".html": parse_html,
    ".htm": parse_html,
    ".md": parse_md,
    ".markdown": parse_md,
    ".txt": parse_md,
    ".png": _parse_image,
    ".jpg": _parse_image,
    ".jpeg": _parse_image,
    ".tif": _parse_image,
    ".tiff": _parse_image,
}


def parse_document(path: Path) -> ParsedDoc:
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        raise ValueError(f"unsupported file type: {path.suffix}")
    return parser(path)


def ingest_all(settings: Settings, embedder: Embedder | None = None, use_embeddings: bool | None = None) -> CorpusIndex:
    started = time.monotonic()
    docs = load_manifest(settings.manifest_path, settings.corpus_dir)
    chunks: list[Chunk] = []
    holidays: list[HolidayEntry] = []
    doc_status: list[dict] = []
    ocr_docs = 0
    for meta in docs.values():
        status = {"doc_id": meta.doc_id, "file": meta.file, "n_chunks": 0, "ocr_used": 0, "status": "ok", "error": None, "sha256": None}
        try:
            status["sha256"] = hashlib.sha256(meta.path.read_bytes()).hexdigest()
            parsed = parse_document(meta.path)
            _, blocks = split_metadata(parsed.blocks)
            doc_chunks = build_chunks(meta, blocks)
            if not doc_chunks:
                raise ValueError("no text extracted")
            chunks.extend(doc_chunks)
            holidays.extend(extract_holidays(meta, blocks))
            status["n_chunks"] = len(doc_chunks)
            status["ocr_used"] = int(parsed.ocr_used)
            ocr_docs += int(parsed.ocr_used)
        except Exception as exc:
            status["status"], status["error"] = "error", f"{type(exc).__name__}: {exc}"[:300]
            log_event(logger, "ingest_doc_failed", logging.ERROR, doc_id=meta.doc_id, error=status["error"])
        doc_status.append(status)

    if use_embeddings is None:
        use_embeddings = settings.embeddings
    if embedder is None and use_embeddings:
        embedder = load_embedder(True, settings.embedding_cache)
    retrieval = RetrievalIndex(chunks, embedder)
    _persist(settings.db_path, doc_status, chunks, holidays)
    stats = {
        "n_docs": len(docs),
        "n_docs_ok": sum(s["status"] == "ok" for s in doc_status),
        "n_chunks": len(chunks),
        "n_ocr_docs": ocr_docs,
        "n_holidays": len(holidays),
        "n_instruction_like_chunks": sum(c.instruction_like for c in chunks),
        "embeddings": embedder.name if embedder else "off",
        "seconds": round(time.monotonic() - started, 2),
    }
    log_event(logger, "ingestion_complete", **stats)
    return CorpusIndex(docs=docs, chunks=chunks, holidays=holidays, retrieval=retrieval, stats=stats)


def _persist(db_path: Path, doc_status: list[dict], chunks: list[Chunk], holidays: list[HolidayEntry]) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with connection(db_path) as conn:
            ensure_side_tables(conn)
            with write_transaction(conn):
                conn.execute("DELETE FROM assistant_documents")
                conn.execute("DELETE FROM assistant_chunks")
                conn.execute("DELETE FROM assistant_holidays")
                conn.executemany(
                    "INSERT INTO assistant_documents VALUES (?,?,?,?,?,?,?,?)",
                    [(s["doc_id"], s["file"], s["sha256"], now, s["n_chunks"], s["ocr_used"], s["status"], s["error"]) for s in doc_status],
                )
                conn.executemany(
                    "INSERT INTO assistant_chunks VALUES (?,?,?,?,?,?,?)",
                    [(c.chunk_id, c.doc_id, c.ordinal, c.locator, c.kind, c.tokens, c.text) for c in chunks],
                )
                conn.executemany(
                    "INSERT OR IGNORE INTO assistant_holidays VALUES (?,?,?,?,?,?)",
                    [(h.doc_id, r, h.day.isoformat(), h.name, h.kind, h.locator) for h in holidays for r in h.regions],
                )
    except Exception as exc:
        log_event(logger, "side_tables_not_written", logging.WARNING, error=f"{type(exc).__name__}: {exc}"[:200])
