"""SQLite access. Existing tables are never altered; derived data goes to assistant_* tables."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# One process, one uvicorn worker: this lock serialises leave writes on top of
# SQLite's own BEGIN IMMEDIATE locking.
WRITE_LOCK = threading.Lock()

SIDE_TABLES_DDL = """
CREATE TABLE IF NOT EXISTS assistant_documents (
  doc_id TEXT PRIMARY KEY, file TEXT, sha256 TEXT, ingested_at TEXT,
  n_chunks INTEGER, ocr_used INTEGER, status TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS assistant_chunks (
  chunk_id TEXT PRIMARY KEY, doc_id TEXT, ordinal INTEGER, locator TEXT,
  kind TEXT, tokens INTEGER, text TEXT
);
CREATE TABLE IF NOT EXISTS assistant_holidays (
  doc_id TEXT, region TEXT, day TEXT, name TEXT, kind TEXT, locator TEXT,
  PRIMARY KEY (doc_id, region, day, name)
);
CREATE INDEX IF NOT EXISTS ix_assistant_holidays_region_day ON assistant_holidays(region, day);
CREATE INDEX IF NOT EXISTS ix_assistant_chunks_doc ON assistant_chunks(doc_id);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    # isolation_level=None: autocommit; write paths open explicit BEGIN IMMEDIATE.
    conn = sqlite3.connect(str(db_path), timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connection(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def write_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE ... COMMIT, rolled back on any exception."""
    with WRITE_LOCK:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")


def ensure_side_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(SIDE_TABLES_DDL)
