"""Markdown parsing: ATX headings, pipe tables, bullet lists and paragraphs."""
from __future__ import annotations

import re
from pathlib import Path

from app.ingestion.model import Block, Heading, Paragraph, ParsedDoc, Table

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _clean(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [_clean(c) for c in line.split("|")]


def parse_md(path: Path) -> ParsedDoc:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    blocks: list[Block] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            blocks.append(Paragraph(text=_clean(" ".join(para))))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush()
            i += 1
            continue
        m = HEADING.match(line)
        if m:
            flush()
            blocks.append(Heading(level=len(m.group(1)), text=_clean(m.group(2))))
            i += 1
            continue
        if "|" in line and i + 1 < len(lines) and TABLE_SEP.match(lines[i + 1]):
            flush()
            header = _cells(line)
            rows: list[list[str]] = []
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(_cells(lines[i]))
                i += 1
            blocks.append(Table(header=header, rows=rows))
            continue
        b = BULLET.match(line)
        if b:
            flush()
            blocks.append(Paragraph(text="• " + _clean(b.group(1))))
            i += 1
            continue
        para.append(line)
        i += 1
    flush()
    return ParsedDoc(blocks=blocks)
