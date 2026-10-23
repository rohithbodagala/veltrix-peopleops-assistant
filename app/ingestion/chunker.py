"""Metadata-box removal and section-level chunking with citation locators.

Locator rule (spec 4.2): the nearest section heading; for a document without any
heading, the page number ("p. N"). Content before the first section heading in a
document that has a title heading is located by the title.
"""
from __future__ import annotations

import re

from app.ingestion.model import Block, Chunk, DocMeta, Heading, Paragraph, Table
from app.retrieval.tokenizer import count_tokens

MAX_CHUNK_TOKENS = 450
META_KEYS = {"document id", "doc id", "version", "effective date", "applies to", "owner", "classification", "status"}
INSTRUCTION_LIKE = re.compile(
    r"\b(ai assistant|language model|assistant reading|chatbot|ignore (all |the |any )?(previous|prior|above) instructions"
    r"|system prompt|note for the ai|instructions? (for|to) the (ai|assistant|model))\b",
    re.I,
)


def _norm_key(text: str) -> str:
    return re.sub(r"[:\s]+$", "", text.strip().lower())


def is_metadata_table(table: Table) -> bool:
    rows = list(table.rows)
    if table.header:
        rows = [table.header] + rows
    keys = {_norm_key(r[0]) for r in rows if r}
    return len(keys & META_KEYS) >= 3


def split_metadata(blocks: list[Block]) -> tuple[dict[str, str], list[Block]]:
    meta: dict[str, str] = {}
    kept: list[Block] = []
    for block in blocks:
        if isinstance(block, Table) and not meta and is_metadata_table(block):
            rows = ([block.header] if block.header else []) + block.rows
            for row in rows:
                if len(row) >= 2:
                    meta[_norm_key(row[0])] = row[1]
            continue
        kept.append(block)
    return meta, kept


def serialize_table(table: Table) -> list[str]:
    lines: list[str] = []
    for row in table.rows:
        if table.header and len(table.header) == len(row):
            parts = [f"{h}: {v}" if h else v for h, v in zip(table.header, row) if v]
        else:
            parts = [v for v in row if v]
        if parts:
            lines.append(" | ".join(parts))
    return lines


class _Section:
    def __init__(self, heading: str | None, page: int | None) -> None:
        self.heading = heading
        self.page = page
        self.pieces: list[tuple[str, str, int | None]] = []  # (text, kind, page)


def _sections(blocks: list[Block], title_heading: Heading | None, has_headings: bool) -> list[_Section]:
    sections: list[_Section] = []
    current: _Section | None = None
    for block in blocks:
        if block is title_heading:
            continue
        if isinstance(block, Heading):
            current = _Section(block.text, block.page)
            sections.append(current)
            continue
        page = block.page
        if current is None or (not has_headings and current.page != page):
            current = _Section(None, page)
            sections.append(current)
        if isinstance(block, Table):
            for line in serialize_table(block):
                current.pieces.append((line, "table", page))
        elif isinstance(block, Paragraph):
            current.pieces.append((block.text, "ocr" if block.ocr else "text", page))
    return sections


def build_chunks(meta: DocMeta, blocks: list[Block]) -> list[Chunk]:
    first_heading = next((b for b in blocks if isinstance(b, Heading)), None)
    title_heading = first_heading if first_heading is not None and first_heading.level == 1 else None
    title = title_heading.text if title_heading else meta.title
    has_headings = first_heading is not None

    chunks: list[Chunk] = []
    for section in _sections(blocks, title_heading, has_headings):
        if section.heading:
            locator = section.heading
        elif has_headings:
            locator = title
        else:
            locator = f"p. {section.page or 1}"
        heading_line = section.heading or ""
        groups: list[list[tuple[str, str, int | None]]] = [[]]
        budget = MAX_CHUNK_TOKENS - count_tokens(heading_line)
        used = 0
        for piece in section.pieces:
            n = count_tokens(piece[0])
            if groups[-1] and used + n > budget:
                groups.append([])
                used = 0
            groups[-1].append(piece)
            used += n
        for group in groups:
            if not group and not heading_line:
                continue
            if not group:
                continue  # a heading with no content of its own
            body = "\n".join(p[0] for p in group)
            text = f"{heading_line}\n{body}" if heading_line else body
            kinds = {p[1] for p in group}
            kind = "ocr" if "ocr" in kinds else ("table" if "table" in kinds else "text")
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{meta.doc_id}#{ordinal:03d}",
                    doc_id=meta.doc_id,
                    ordinal=ordinal,
                    locator=locator,
                    section=heading_line,
                    text=text,
                    index_text=f"{title}\n{text}",
                    tokens=count_tokens(text),
                    page=group[0][2],
                    kind=kind,
                    instruction_like=bool(INSTRUCTION_LIKE.search(text)),
                )
            )
    return chunks
