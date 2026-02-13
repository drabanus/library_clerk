"""
Text extraction from PDFs and ebooks.

Handles PDF (via PyMuPDF), EPUB (via ebooklib), and basic metadata extraction.
"""

import os
import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF
import ebooklib
from ebooklib import epub
from html.parser import HTMLParser


@dataclass
class ExtractedDocument:
    """Represents extracted content from a publication file."""
    file_path: str
    file_hash: str
    file_type: str  # "pdf", "epub"
    raw_text: str
    title: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    num_pages: Optional[int] = None
    file_size_bytes: int = 0

    @property
    def text_preview(self) -> str:
        """First 2000 characters for quick inspection."""
        return self.raw_text[:2000]

    @property
    def word_count(self) -> int:
        return len(self.raw_text.split())


class _HTMLTextExtractor(HTMLParser):
    """Strip HTML tags and extract plain text."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str):
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(html_content: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html_content)
    return parser.get_text()


def _compute_file_hash(path: str) -> str:
    """SHA-256 hash of file contents for change detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_pdf(path: str) -> ExtractedDocument:
    """Extract text and metadata from a PDF file."""
    doc = fitz.open(path)
    metadata = doc.metadata or {}

    pages_text = []
    for page in doc:
        pages_text.append(page.get_text("text"))

    raw_text = _clean_text("\n".join(pages_text))

    title = metadata.get("title") or None
    author_str = metadata.get("author") or ""
    authors = [a.strip() for a in re.split(r"[,;&]", author_str) if a.strip()]

    year = None
    date_str = metadata.get("creationDate") or metadata.get("modDate") or ""
    year_match = re.search(r"(\d{4})", date_str)
    if year_match:
        y = int(year_match.group(1))
        if 1900 <= y <= 2100:
            year = y

    doc.close()

    return ExtractedDocument(
        file_path=str(Path(path).resolve()),
        file_hash=_compute_file_hash(path),
        file_type="pdf",
        raw_text=raw_text,
        title=title if title and title.strip() else None,
        authors=authors,
        year=year,
        num_pages=len(pages_text),
        file_size_bytes=os.path.getsize(path),
    )


def extract_epub(path: str) -> ExtractedDocument:
    """Extract text and metadata from an EPUB file."""
    book = epub.read_epub(path, options={"ignore_ncx": True})

    # Metadata
    title = None
    titles = book.get_metadata("DC", "title")
    if titles:
        title = titles[0][0]

    authors = []
    creators = book.get_metadata("DC", "creator")
    for creator in creators:
        authors.append(creator[0])

    year = None
    dates = book.get_metadata("DC", "date")
    for date_entry in dates:
        year_match = re.search(r"(\d{4})", str(date_entry[0]))
        if year_match:
            y = int(year_match.group(1))
            if 1900 <= y <= 2100:
                year = y
                break

    # Text extraction
    text_parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content().decode("utf-8", errors="ignore")
        text_parts.append(_strip_html(content))

    raw_text = _clean_text("\n".join(text_parts))

    return ExtractedDocument(
        file_path=str(Path(path).resolve()),
        file_hash=_compute_file_hash(path),
        file_type="epub",
        raw_text=raw_text,
        title=title,
        authors=authors,
        year=year,
        num_pages=None,
        file_size_bytes=os.path.getsize(path),
    )


def extract_document(path: str) -> ExtractedDocument:
    """Extract text from any supported document type."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(path)
    elif ext == ".epub":
        return extract_epub(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def scan_library(paths: list[str], extensions: list[str]) -> list[str]:
    """
    Recursively scan directories for publication files.
    Returns sorted list of absolute file paths.
    """
    found = set()
    extensions = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}

    for base_path in paths:
        base = Path(base_path).expanduser().resolve()
        if not base.exists():
            continue
        if base.is_file():
            if base.suffix.lower() in extensions:
                found.add(str(base))
            continue
        for root, _dirs, files in os.walk(base):
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() in extensions:
                    found.add(str(fpath.resolve()))

    return sorted(found)
