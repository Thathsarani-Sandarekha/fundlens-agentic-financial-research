"""
PDF page thumbnail generation using PyMuPDF.

Generates a PNG image for each page, used in the UI's
"source viewer" tab so analysts can click through to see
the original filing context.
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

# DPI controls quality vs file size
# 72 = web-quality, ~30 KB per page (good for thumbnails)
# 150 = readable, ~150 KB per page
# 300 = print-quality, ~600 KB per page
DEFAULT_DPI = 72


# ============================================================
# THUMBNAIL GENERATION
# ============================================================

def generate_thumbnails(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
    max_pages: int = None,
) -> list[Path]:
    """
    Generate one PNG thumbnail per page of a PDF.

    Args:
        pdf_path: Local path to PDF file
        output_dir: Where to write PNG files
        dpi: Resolution (default 72 — low for thumbnails)
        max_pages: Optionally limit how many pages to render

    Returns:
        List of local paths to generated PNGs, in page order
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    pages_to_render = min(total_pages, max_pages) if max_pages else total_pages

    logger.info(
        f"  Generating {pages_to_render} thumbnails at {dpi} DPI from {pdf_path.name}"
    )

    # PyMuPDF uses a matrix to scale pages — 72 DPI is default
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    generated_paths = []
    for page_idx in range(pages_to_render):
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        # Use 1-indexed page numbers in filenames
        out_path = output_dir / f"page_{page_idx + 1:03d}.png"
        pix.save(str(out_path))
        generated_paths.append(out_path)

    doc.close()
    logger.info(f"  ✅ Generated {len(generated_paths)} thumbnails in {output_dir}")
    return generated_paths


def count_pdf_pages(pdf_path: Path) -> int:
    """Return the total page count of a PDF."""
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count