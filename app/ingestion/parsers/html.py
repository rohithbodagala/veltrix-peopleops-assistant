"""HTML parsing with BeautifulSoup: headings, paragraphs, list items and tables in order."""
from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from app.ingestion.model import Block, Heading, Paragraph, ParsedDoc, Table

DROP_TAGS = ("script", "style", "nav", "footer", "header", "noscript", "template", "form", "button")
HEADINGS = {f"h{i}": i for i in range(1, 7)}
TEXT_BLOCKS = {"p", "pre", "blockquote", "dt", "dd", "figcaption", "address", "summary"}


def _text(node: Tag) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _table(node: Tag) -> Table | None:
    rows: list[list[str]] = []
    header: list[str] | None = None
    for tr in node.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        values = [_text(c) for c in cells]
        if not any(values):
            continue
        is_header = all(c.name == "th" for c in cells) or tr.find_parent("thead") is not None
        if is_header and header is None and not rows:
            header = values
        else:
            rows.append(values)
    if not rows and header is None:
        return None
    return Table(header=header, rows=rows)


def _walk(node: Tag, blocks: list[Block], inline: list[str]) -> None:
    def flush_inline() -> None:
        text = re.sub(r"\s+", " ", " ".join(inline)).strip()
        if text:
            blocks.append(Paragraph(text=text))
        inline.clear()

    for child in node.children:
        if isinstance(child, NavigableString):
            if child.strip() and child.__class__ is NavigableString:
                inline.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name.lower()
        if name in HEADINGS:
            flush_inline()
            text = _text(child)
            if text:
                blocks.append(Heading(level=HEADINGS[name], text=text))
        elif name == "table":
            flush_inline()
            table = _table(child)
            if table is not None:
                blocks.append(table)
        elif name == "li":
            flush_inline()
            nested = child.find(["ul", "ol"])
            if nested is not None:
                nested.extract()
            text = _text(child)
            if text:
                blocks.append(Paragraph(text="• " + text))
            if nested is not None:
                _walk(nested, blocks, inline)
        elif name in TEXT_BLOCKS:
            flush_inline()
            text = _text(child)
            if text:
                blocks.append(Paragraph(text=text))
        elif name == "br":
            flush_inline()
        elif name in ("span", "a", "strong", "b", "em", "i", "u", "code", "small", "sup", "sub", "mark", "abbr", "time"):
            text = _text(child)
            if text:
                inline.append(text)
        else:  # div, section, article, main, ul, ol, dl, ...
            flush_inline()
            _walk(child, blocks, inline)
    flush_inline()


def parse_html(path: Path) -> ParsedDoc:
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    for tag in soup.find_all(DROP_TAGS):
        tag.decompose()
    root = soup.body or soup
    blocks: list[Block] = []
    _walk(root, blocks, [])
    if not any(isinstance(b, Heading) and b.level == 1 for b in blocks) and soup.title and soup.title.string:
        blocks.insert(0, Heading(level=1, text=soup.title.string.strip()))
    return ParsedDoc(blocks=blocks)
