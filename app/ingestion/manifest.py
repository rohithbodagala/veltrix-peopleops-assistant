"""manifest.json loading and validation."""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from app.ingestion.model import DocMeta
from app.logging_setup import log_event

logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path, corpus_dir: Path) -> dict[str, DocMeta]:
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs: dict[str, DocMeta] = {}
    for entry in entries:
        try:
            regions = entry.get("regions") or ["GLOBAL"]
            if isinstance(regions, str):
                regions = [regions]
            meta = DocMeta(
                doc_id=str(entry["doc_id"]).strip(),
                file=str(entry["file"]).strip(),
                title=str(entry.get("title") or entry["doc_id"]).strip(),
                doc_type=str(entry.get("doc_type") or "").strip().lower(),
                regions=tuple(str(r).strip().upper() for r in regions),
                effective_date=date.fromisoformat(str(entry["effective_date"]).strip()),
                access_level=str(entry.get("access_level") or "").strip().lower(),
                path=corpus_dir / str(entry["file"]).strip(),
            )
        except (KeyError, ValueError, TypeError) as exc:
            log_event(logger, "manifest_entry_invalid", logging.WARNING, entry=str(entry.get("doc_id")), error=str(exc))
            continue
        if meta.doc_id in docs:
            log_event(logger, "manifest_duplicate_doc_id", logging.WARNING, doc_id=meta.doc_id)
            continue
        docs[meta.doc_id] = meta
    return docs
