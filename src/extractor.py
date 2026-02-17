"""
Text extraction from PDFs and ebooks.

Handles PDF (via PyMuPDF), EPUB (via ebooklib), and basic metadata extraction.
Detects scanned/image-only PDFs, renames them to *_noOCR.pdf, and runs OCR
via ocrmypdf before continuing extraction.
"""

import os
import re
import shutil
import hashlib
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF

# Suppress noisy MuPDF warnings about malformed PDF resources
# (e.g. "cannot find ExtGState resource 'A1'").  These are cosmetic —
# text extraction still works — and they clutter the progress output.
fitz.TOOLS.mupdf_display_errors(False)

import ebooklib
from ebooklib import epub
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

# Minimum average words per page to consider a PDF as having real text content.
# Scanned documents typically yield 0-5 words per page from stray OCR artifacts
# in metadata, while text-based PDFs yield 100+ words per page.
MIN_WORDS_PER_PAGE = 20


@dataclass
class ExtractedDocument:
    """Represents extracted content from a publication file."""
    file_path: str
    file_hash: str
    file_type: str  # "pdf", "epub"
    raw_text: str
    was_ocred: bool = False
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


def compute_file_hash(path: str) -> str:
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


def _is_scanned_pdf(doc: fitz.Document) -> bool:
    """
    Detect whether a PDF is a scanned document (images only, no real text).

    Checks each page for:
      - Text content length (word count)
      - Presence of embedded images

    A PDF is considered scanned if most pages contain images but yield
    fewer than MIN_WORDS_PER_PAGE words of extractable text on average.
    """
    if doc.page_count == 0:
        return False

    total_words = 0
    pages_with_images = 0

    for page in doc:
        text = page.get_text("text").strip()
        words = len(text.split()) if text else 0
        total_words += words

        image_list = page.get_images(full=True)
        if image_list:
            pages_with_images += 1

    avg_words = total_words / doc.page_count

    # Scanned: pages have images but very little extractable text
    image_ratio = pages_with_images / doc.page_count
    is_scanned = avg_words < MIN_WORDS_PER_PAGE and image_ratio > 0.5

    if is_scanned:
        logger.info(
            f"  Scanned PDF detected: {avg_words:.0f} avg words/page, "
            f"{pages_with_images}/{doc.page_count} pages with images"
        )
    else:
        logger.debug(
            f"  Text PDF: {avg_words:.0f} avg words/page, "
            f"{pages_with_images}/{doc.page_count} pages with images"
        )

    return is_scanned


def _has_unpaper() -> bool:
    """Check whether the 'unpaper' binary is on PATH."""
    return shutil.which("unpaper") is not None


def _ocr_available() -> bool:
    """Check whether ocrmypdf and tesseract are both available."""
    if not shutil.which("ocrmypdf"):
        logger.warning("  ocrmypdf not found on PATH — OCR disabled")
        return False
    if not shutil.which("tesseract"):
        logger.warning(
            "  tesseract not found on PATH — OCR disabled. "
            "Install with: sudo apt install tesseract-ocr"
        )
        return False
    return True


def _run_ocr(pdf_path: str) -> str:
    """
    Run OCR on a scanned PDF.

    1. Renames the original file from *.pdf to *_noOCR.pdf
    2. Runs ocrmypdf to produce a text-layer PDF at the original path
    3. Returns the path to the OCR'd PDF (= original path)

    Raises RuntimeError if OCR fails.
    """
    original = Path(pdf_path)
    stem = original.stem
    no_ocr_name = f"{stem}_noOCR.pdf"
    no_ocr_path = original.parent / no_ocr_name

    # If the _noOCR version already exists, another run already renamed it.
    # The current file at pdf_path is then already the OCR'd version.
    if no_ocr_path.exists():
        logger.info(f"  OCR backup already exists: {no_ocr_path}")
        return str(original)

    # Rename original → _noOCR.pdf
    logger.info(f"  Renaming: {original.name} -> {no_ocr_name}")
    shutil.move(str(original), str(no_ocr_path))

    # Build ocrmypdf command.  --clean requires the 'unpaper' system
    # package; only use it when unpaper is installed.
    cmd = [
        "ocrmypdf",
        "--skip-text",       # Don't re-OCR pages that already have text
        "--optimize", "1",   # Light optimization
        "--deskew",          # Fix skewed scans
    ]
    if _has_unpaper():
        cmd.append("--clean")   # Clean up scan artifacts (needs unpaper)
    else:
        logger.info("  unpaper not found — skipping --clean flag")
    cmd += ["--quiet", str(no_ocr_path), str(original)]

    # Run ocrmypdf:  _noOCR.pdf  →  original.pdf
    logger.info(f"  Running OCR: {no_ocr_name} -> {original.name}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per file
        )

        if result.returncode != 0:
            # OCR failed — restore original file
            stderr = result.stderr.strip()
            logger.error(f"  OCR failed (exit {result.returncode}): {stderr}")
            shutil.move(str(no_ocr_path), str(original))
            raise RuntimeError(
                f"ocrmypdf failed on {original.name}: {stderr}"
            )

        logger.info(f"  OCR complete: {original.name}")
        return str(original)

    except FileNotFoundError:
        # ocrmypdf not installed — restore and raise
        logger.error(
            "  ocrmypdf not found. Install with: "
            "pip install ocrmypdf && sudo apt install tesseract-ocr"
        )
        shutil.move(str(no_ocr_path), str(original))
        raise RuntimeError(
            "ocrmypdf is not installed. "
            "Install: pip install ocrmypdf && sudo apt install tesseract-ocr"
        )

    except subprocess.TimeoutExpired:
        logger.error(f"  OCR timed out for {original.name}")
        # Restore if OCR'd file wasn't produced
        if not original.exists() and no_ocr_path.exists():
            shutil.move(str(no_ocr_path), str(original))
        raise RuntimeError(f"OCR timed out for {original.name}")


def extract_pdf(path: str) -> ExtractedDocument:
    """
    Extract text and metadata from a PDF file.

    If the PDF is detected as a scanned document (images with no text layer):
      1. Original is renamed to *_noOCR.pdf
      2. OCR is run via ocrmypdf, producing a text-layer PDF at the original path
      3. Text is extracted from the new OCR'd PDF
    """
    doc = fitz.open(path)
    metadata = doc.metadata or {}
    num_pages = doc.page_count
    was_ocred = False

    # Check if this is a scanned document
    if _is_scanned_pdf(doc):
        if _ocr_available():
            doc.close()

            # OCR pipeline: rename original, produce OCR'd version
            path = _run_ocr(path)
            was_ocred = True

            # Re-open the OCR'd PDF
            doc = fitz.open(path)
            metadata = doc.metadata or {}
        else:
            logger.warning(
                f"  Scanned PDF but OCR tools not available — "
                f"extracting with limited text: {Path(path).name}"
            )

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
        file_hash=compute_file_hash(path),
        file_type="pdf",
        raw_text=raw_text,
        was_ocred=was_ocred,
        title=title if title and title.strip() else None,
        authors=authors,
        year=year,
        num_pages=num_pages,
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
        file_hash=compute_file_hash(path),
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
    Skips *_noOCR.pdf files (these are the pre-OCR originals).
    Returns sorted list of absolute file paths.
    """
    found = set()
    extensions = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}

    for base_path in paths:
        base = Path(base_path).expanduser().resolve()
        if not base.exists():
            continue
        if base.is_file():
            if base.suffix.lower() in extensions and not _is_noocr_backup(base):
                found.add(str(base))
            continue
        for root, _dirs, files in os.walk(base):
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() in extensions and not _is_noocr_backup(fpath):
                    found.add(str(fpath.resolve()))

    return sorted(found, key=lambda p: os.path.getsize(p))


def _is_noocr_backup(path: Path) -> bool:
    """Check if a file is a *_noOCR.pdf backup of a scanned original."""
    return path.stem.endswith("_noOCR")
