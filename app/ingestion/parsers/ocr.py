"""OCR through the tesseract CLI (installed in the Docker image)."""
from __future__ import annotations

import re
import shutil
import subprocess

from app.ingestion.model import Paragraph


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def ocr_image_bytes(image: bytes, timeout: float = 120) -> str:
    """Run tesseract on an encoded image (PNG/JPEG) and return plain text."""
    if not tesseract_available():
        raise RuntimeError("tesseract is not installed")
    proc = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", "eng"],
        input=image,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract failed: {proc.stderr.decode(errors='replace')[:200]}")
    return proc.stdout.decode("utf-8", errors="replace")


def _is_text_line(line: str) -> bool:
    """Drop OCR noise such as signatures and stamps: keep lines with real words."""
    words = re.findall(r"[A-Za-z]{2,}", line)
    if len(words) >= 2:
        return True
    return len(words) == 1 and len(words[0]) >= 4 and len(set(words[0].lower())) > 1


def ocr_text_to_paragraphs(text: str, page: int | None) -> list[Paragraph]:
    """Blank-line separated OCR text -> paragraphs, re-joining sentences split across blocks."""
    paragraphs: list[Paragraph] = []
    for raw in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and _is_text_line(ln)]
        if not lines:
            continue
        joined = " ".join(lines)
        if paragraphs and not re.search(r"[.:;!?]$", paragraphs[-1].text) and joined[:1].islower():
            paragraphs[-1].text += " " + joined
        else:
            paragraphs.append(Paragraph(text=joined, page=page, ocr=True))
    return paragraphs
