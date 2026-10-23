"""PDF parsing with PyMuPDF.

- Text pages: spans are grouped into visual lines; headings are detected from font
  size/weight; running headers/footers are dropped; tables are located with
  `find_tables()` but their cell text is rebuilt from spans (the library's own cell
  extraction drops spaces for some embedded fonts).
- Pages without a text layer are rendered at 300 dpi and OCR'd with tesseract.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.ingestion.model import Block, Heading, Paragraph, ParsedDoc, Table
from app.ingestion.parsers.ocr import ocr_image_bytes, ocr_text_to_paragraphs

BULLETS = "•▪◦●‣–-*"
NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+\S")
PAGE_MARK = re.compile(r"^(page\s+)?\d+(\s+(of|/)\s+\d+)?$", re.I)
MARGIN_BAND = 0.12  # top/bottom share of the page where running headers and footers live


@dataclass
class _Span:
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool
    text: str


@dataclass
class _Line:
    x0: float
    y0: float
    y1: float
    size: float
    bold: bool
    text: str


def _spans(page: pymupdf.Page) -> list[_Span]:
    out: list[_Span] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for s in line["spans"]:
                if not s["text"].strip():
                    continue
                x0, y0, x1, y1 = s["bbox"]
                bold = bool(s["flags"] & 16) or "bold" in s["font"].lower()
                out.append(_Span(x0, y0, x1, y1, s["size"], bold, s["text"]))
    return out


def _inside(span: _Span, bbox: tuple[float, float, float, float]) -> bool:
    cx, cy = (span.x0 + span.x1) / 2, (span.y0 + span.y1) / 2
    return bbox[0] - 1 <= cx <= bbox[2] + 1 and bbox[1] - 1 <= cy <= bbox[3] + 1


def _join_spans(spans: list[_Span]) -> str:
    spans = sorted(spans, key=lambda s: (round(s.y0 / 3), s.x0))
    text = ""
    prev: _Span | None = None
    for s in spans:
        piece = s.text
        if prev is not None:
            same_line = abs(s.y0 - prev.y0) < 3
            gap = s.x0 - prev.x1
            if not same_line or gap > 1.0:
                text = text.rstrip() + " "
        text += piece
        prev = s
    # NFKC folds ligatures (e.g. U+FB03 'ffi') and compatibility forms into plain text.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _group_lines(spans: list[_Span]) -> list[_Line]:
    spans = sorted(spans, key=lambda s: (s.y0, s.x0))
    groups: list[list[_Span]] = []
    for s in spans:
        if groups and abs(groups[-1][0].y0 - s.y0) < 3:
            groups[-1].append(s)
        else:
            groups.append([s])
    lines: list[_Line] = []
    for g in groups:
        g.sort(key=lambda s: s.x0)
        main = max(g, key=lambda s: len(s.text.strip()))
        lines.append(
            _Line(
                x0=g[0].x0,
                y0=min(s.y0 for s in g),
                y1=max(s.y1 for s in g),
                size=main.size,
                bold=main.bold,
                text=_join_spans(g),
            )
        )
    return lines


def _table_block(table, spans: list[_Span], page_no: int) -> Table | None:
    rows: list[list[str]] = []
    header_bold: list[bool] = []
    for r_index, row in enumerate(table.rows):
        cells: list[str] = []
        bolds: list[bool] = []
        for bbox in row.cells:
            if bbox is None:
                cells.append("")
                continue
            inside = [s for s in spans if _inside(s, bbox)]
            cells.append(_join_spans(inside))
            if inside:
                bolds.append(all(s.bold for s in inside))
        if any(c for c in cells):
            rows.append(cells)
            if r_index == 0:
                header_bold = bolds
    if not rows:
        return None
    header = None
    if len(rows) > 1 and header_bold and all(header_bold) and len(header_bold) == len([c for c in rows[0] if c]):
        header = rows.pop(0)
    return Table(header=header, rows=rows, page=page_no)


def _ocr_page(page: pymupdf.Page, page_no: int) -> list[Block]:
    pix = page.get_pixmap(dpi=300)
    text = ocr_image_bytes(pix.tobytes("png"))
    return list(ocr_text_to_paragraphs(text, page_no))


def parse_pdf(path: Path) -> ParsedDoc:
    doc = pymupdf.open(str(path))
    parsed = ParsedDoc(blocks=[])
    page_items: list[tuple[int, list[tuple[float, object]], float]] = []
    all_sizes: Counter = Counter()

    for page_no, page in enumerate(doc, start=1):
        if len(page.get_text().strip()) < 40 and page.get_images():
            parsed.blocks.extend(_ocr_page(page, page_no))
            parsed.ocr_used = True
            continue
        spans = _spans(page)
        tables = []
        # The default table finder works from ruling lines; a page without vector
        # drawings cannot contain a detectable table, and the finder is expensive.
        if page.get_drawings():
            try:
                tables = page.find_tables().tables
            except Exception:  # pragma: no cover - library edge cases
                tables = []
        items: list[tuple[float, object]] = []
        table_bboxes = []
        for t in tables:
            block = _table_block(t, spans, page_no)
            if block is not None:
                items.append((t.bbox[1], block))
                table_bboxes.append(t.bbox)
        free = [s for s in spans if not any(_inside(s, bb) for bb in table_bboxes)]
        for line in _group_lines(free):
            items.append((line.y0, line))
            all_sizes[round(line.size, 1)] += len(line.text)
        page_items.append((page_no, items, page.rect.height))

    body_size = all_sizes.most_common(1)[0][0] if all_sizes else 10.0
    running = _running_lines(page_items)
    for page_no, items, height in page_items:
        items.sort(key=lambda it: it[0])
        parsed.blocks.extend(_page_blocks(items, page_no, height, body_size, running))
    _merge_across_pages(parsed.blocks)
    return parsed


def _norm_running(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _running_lines(page_items) -> set[str]:
    """Texts that appear in the top/bottom margin band of at least two pages."""
    seen: Counter = Counter()
    for _, items, height in page_items:
        texts = {
            _norm_running(it.text)
            for _, it in items
            if isinstance(it, _Line) and (it.y0 < MARGIN_BAND * height or it.y0 > (1 - MARGIN_BAND) * height)
        }
        seen.update(texts)
    return {t for t, n in seen.items() if n >= 2}


def _page_blocks(items, page_no: int, height: float, body_size: float, running: set[str]) -> list[Block]:
    blocks: list[Block] = []
    para: list[str] = []
    prev_line: _Line | None = None

    def flush() -> None:
        nonlocal para
        if para:
            blocks.append(Paragraph(text=" ".join(para), page=page_no))
            para = []

    for _, item in items:
        if isinstance(item, Table):
            flush()
            blocks.append(item)
            prev_line = None
            continue
        line: _Line = item
        in_margin = line.y0 < MARGIN_BAND * height or line.y0 > (1 - MARGIN_BAND) * height
        if in_margin and (
            line.size < body_size - 0.5 or PAGE_MARK.match(line.text) or _norm_running(line.text) in running
        ):
            continue  # running header / footer / page number
        text = line.text.strip()
        is_heading = line.bold and len(text) < 120 and (
            line.size >= 1.15 * body_size or (NUMBERED_HEADING.match(text) and line.size >= body_size)
        )
        if is_heading:
            flush()
            level = 1 if line.size >= 1.5 * body_size else 2
            blocks.append(Heading(level=level, text=text, page=page_no))
            prev_line = None
            continue
        starts_bullet = text[:1] in BULLETS and (len(text) > 1 and text[1] == " ")
        gap_break = prev_line is not None and (line.y0 - prev_line.y1) > 0.5 * max(1.0, prev_line.y1 - prev_line.y0)
        if starts_bullet or gap_break:
            flush()
        if starts_bullet:
            text = "• " + text[2:].strip()
        para.append(text)
        prev_line = line
    flush()
    return blocks


def _merge_across_pages(blocks: list[Block]) -> None:
    """Join a paragraph split by a page break (previous one ends mid-sentence)."""
    i = 0
    while i < len(blocks) - 1:
        a, b = blocks[i], blocks[i + 1]
        if (
            isinstance(a, Paragraph)
            and isinstance(b, Paragraph)
            and a.page is not None
            and b.page == a.page + 1
            and not a.ocr
            and not re.search(r"[.:;!?]$", a.text)
            and not b.text.startswith("•")
        ):
            a.text = a.text + " " + b.text
            del blocks[i + 1]
            continue
        i += 1
