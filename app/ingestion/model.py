"""Shared data types for ingestion and retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    file: str
    title: str
    doc_type: str
    regions: tuple[str, ...]
    effective_date: date
    access_level: str
    path: Path


@dataclass
class Heading:
    level: int
    text: str
    page: int | None = None


@dataclass
class Paragraph:
    text: str
    page: int | None = None
    ocr: bool = False


@dataclass
class Table:
    header: list[str] | None
    rows: list[list[str]]
    page: int | None = None


Block = Union[Heading, Paragraph, Table]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    locator: str
    section: str
    text: str  # what the LLM sees (section heading + body)
    index_text: str  # what retrieval indexes (title + section + body)
    tokens: int
    page: int | None
    kind: str  # "text" | "table" | "ocr"
    instruction_like: bool = False


@dataclass(frozen=True)
class HolidayEntry:
    doc_id: str
    regions: tuple[str, ...]
    day: date
    name: str
    kind: str  # "public" (office closed) | "restricted" (optional, office open)
    locator: str


@dataclass
class ParsedDoc:
    blocks: list[Block]
    ocr_used: bool = False
    warnings: list[str] = field(default_factory=list)
