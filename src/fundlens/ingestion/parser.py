"""
Main parsing service: PDF (in S3) → ParsedDocument

Workflow:
  1. Download PDF from S3 to a temp file
  2. Parse with Docling → structured pages/tables/text
  3. Generate page thumbnails with PyMuPDF
  4. Upload thumbnails back to S3
  5. Return a ParsedDocument Pydantic model
  6. Cleanup temp files

Usage:
    from ingestion.parser import parse_document
    doc = parse_document(
        bucket="fundlens-filings",
        key="filings/AAPL/AAPL_2024_10K.pdf",
        ticker="AAPL",
        filing_type="10-K",
    )
    print(doc.summary())
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

# ⚠️ CRITICAL: must be BEFORE docling imports
# Force CPU mode (avoids GTX 1650 4GB VRAM OOM crashes)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fundlens.models.parsed import ParsedDocument, ParsedPage, TextBlock, ExtractedTable
from fundlens.config import settings
from fundlens.storage.s3_client import download_to_local, upload_file, ensure_bucket
from fundlens.ingestion.thumbnails import generate_thumbnails, count_pdf_pages
from fundlens.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

THUMBNAIL_BUCKET = "fundlens-thumbnails"
THUMBNAIL_DPI = 100  # readable but small file size


# ============================================================
# DOCLING TYPE → BLOCK TYPE MAPPING
# ============================================================

def classify_block_type(docling_type_name: str) -> str:
    """Map Docling element types to our block_type enum."""
    t = docling_type_name.lower()

    if "title" in t or "sectionheader" in t or "heading" in t:
        return "header"
    if "footnote" in t:
        return "footnote"
    if "listitem" in t or "list" in t:
        return "list_item"
    if "caption" in t:
        return "caption"
    if "text" in t or "paragraph" in t:
        return "narrative"
    return "other"


def extract_bbox(item) -> Optional[tuple[float, float, float, float]]:
    """Extract bounding box from a Docling item if available."""
    if not hasattr(item, "prov") or not item.prov:
        return None
    for p in item.prov:
        if hasattr(p, "bbox") and p.bbox:
            bbox = p.bbox
            return (
                float(bbox.l),
                float(bbox.t),
                float(bbox.r),
                float(bbox.b),
            )
    return None


def extract_page_number(item) -> Optional[int]:
    """Extract page number from a Docling item if available."""
    if not hasattr(item, "prov") or not item.prov:
        return None
    for p in item.prov:
        if hasattr(p, "page_no"):
            return int(p.page_no)
    return None


# ============================================================
# TABLE EXTRACTION
# ============================================================

def parse_docling_table(table_item, doc=None) -> ExtractedTable:
    """Convert a Docling TableItem to our ExtractedTable model."""
    headers: list[str] = []
    rows: list[list[str]] = []
    raw_html = ""

    # Try to export as markdown for raw_html field
    try:
        if doc is not None:
            raw_html = table_item.export_to_markdown(doc=doc)
        elif hasattr(table_item, "export_to_markdown"):
            raw_html = table_item.export_to_markdown()
    except Exception as e:
        logger.debug(f"    Could not export table to markdown: {e}")

    # Try to get structured rows/columns
    try:
        if hasattr(table_item, "data") and table_item.data:
            table_data = table_item.data
            # table_data.table_cells is a 2D list
            if hasattr(table_data, "grid") and table_data.grid:
                grid = table_data.grid
                if grid:
                    # First row is usually headers
                    headers = [cell.text if hasattr(cell, "text") else str(cell) for cell in grid[0]]
                    rows = [
                        [cell.text if hasattr(cell, "text") else str(cell) for cell in row]
                        for row in grid[1:]
                    ]
    except Exception as e:
        logger.debug(f"    Could not extract table structure: {e}")

    return ExtractedTable(
        raw_html=raw_html,
        headers=headers,
        rows=rows,
        caption=None,
        bbox=extract_bbox(table_item),
    )


# ============================================================
# MAIN PARSING FUNCTION
# ============================================================

def parse_document(
    bucket: str,
    key: str,
    ticker: str,
    filing_type: str,
    filing_id: Optional[str] = None,
    upload_thumbnails: bool = True,
    max_pages: Optional[int] = None,
) -> ParsedDocument:
    """
    Parse a PDF from S3 into a structured ParsedDocument.

    Args:
        bucket: S3 bucket name (e.g., 'fundlens-filings')
        key: S3 key/path (e.g., 'filings/AAPL/AAPL_2024_10K.pdf')
        ticker: Stock ticker for the company
        filing_type: '10-K', '10-Q', 'earnings', etc.
        filing_id: Optional unique ID. Defaults to ticker+key.
        upload_thumbnails: Whether to upload thumbnails to S3
        max_pages: Optionally limit thumbnail generation

    Returns:
        ParsedDocument with pages, tables, text blocks, and thumbnail paths
    """
    start_time = time.time()
    filing_id = filing_id or f"{ticker}_{Path(key).stem}"
    warnings_collected: list[str] = []

    logger.info(f"📄 Parsing {ticker} {filing_type} | s3://{bucket}/{key}")

    # ── Ensure thumbnail bucket exists ───────────────────────
    if upload_thumbnails:
        ensure_bucket(THUMBNAIL_BUCKET)

    # ── Use temp dir for downloaded PDF + thumbnails ─────────
    with tempfile.TemporaryDirectory(prefix="docmind_") as tmp:
        tmp_path = Path(tmp)
        local_pdf = tmp_path / Path(key).name
        thumb_dir = tmp_path / "thumbnails"

        # ── 1. Download PDF from S3 ──────────────────────────
        logger.info(f"  ⬇️  Downloading from S3...")
        download_to_local(bucket, key, local_pdf)
        logger.info(f"  ✅ Downloaded {local_pdf.stat().st_size // 1024} KB")

        # ── 2. Get true page count from PDF ──────────────────
        total_pages = count_pdf_pages(local_pdf)
        logger.info(f"  📊 PDF has {total_pages} pages")

        # ── 3. Generate thumbnails ───────────────────────────
        logger.info(f"  🖼️  Generating thumbnails at {THUMBNAIL_DPI} DPI...")
        thumb_paths = generate_thumbnails(
            local_pdf,
            thumb_dir,
            dpi=THUMBNAIL_DPI,
            max_pages=max_pages,
        )

        # ── 4. Upload thumbnails to S3 ───────────────────────
        thumbnail_s3_paths: dict[int, str] = {}
        if upload_thumbnails:
            logger.info(f"  ⬆️  Uploading {len(thumb_paths)} thumbnails to S3...")
            pdf_stem = Path(key).stem  # AAPL_2024_10K (no extension)
            for thumb_path in thumb_paths:
                # Parse "page_001.png" → 1
                page_num = int(thumb_path.stem.replace("page_", ""))
                thumb_key = f"thumbnails/{ticker}/{pdf_stem}/{thumb_path.name}"
                upload_file(
                    thumb_path,
                    THUMBNAIL_BUCKET,
                    thumb_key,
                    content_type="image/png",
                )
                thumbnail_s3_paths[page_num] = thumb_key
            logger.info(f"  ✅ Uploaded {len(thumbnail_s3_paths)} thumbnails")

        # ── 5. Parse with Docling ────────────────────────────
        logger.info(f"  🧠 Running Docling parser (CPU mode)...")
        pages_dict = parse_with_docling(local_pdf, warnings_collected)
        logger.info(f"  ✅ Parsed {len(pages_dict)} pages")

    # ── 6. Build ParsedPage objects ──────────────────────────
    parsed_pages: list[ParsedPage] = []
    for page_num in sorted(pages_dict.keys()):
        page_data = pages_dict[page_num]
        parsed_pages.append(
            ParsedPage(
                page_number=page_num,
                text_blocks=page_data["text_blocks"],
                tables=page_data["tables"],
                thumbnail_s3_path=thumbnail_s3_paths.get(page_num),
            )
        )

    # ── 7. Build final ParsedDocument ────────────────────────
    duration = time.time() - start_time
    parsed_doc = ParsedDocument(
        filing_id=filing_id,
        company_ticker=ticker,
        filing_type=filing_type,
        s3_bucket=bucket,
        s3_key=key,
        pages=parsed_pages,
        total_pages_in_pdf=total_pages,
        pages_parsed_successfully=len(parsed_pages),
        parser_used="docling",
        parse_duration_seconds=duration,
        parser_warnings=warnings_collected,
    )

    logger.info(f"✅ {parsed_doc.summary()}")
    return parsed_doc


# ============================================================
# DOCLING PARSING (isolated for clarity)
# ============================================================

def parse_with_docling(local_pdf: Path, warnings: list[str]) -> dict[int, dict]:
    """
    Run Docling on a PDF, processing in small batches to isolate crashes.
    
    When Docling hits a problematic page, std::bad_alloc cascades to all
    subsequent pages. We work around this by processing in batches and
    catching failures per batch.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        AcceleratorOptions,
        AcceleratorDevice,
        TableFormerMode,
    )
    import fitz
    import tempfile
    
    # Setup Docling options (same as before)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=2,
        device=AcceleratorDevice.CPU,
    )
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    # Get total page count
    src_doc = fitz.open(str(local_pdf))
    total_pages = len(src_doc)
    
    pages_dict: dict[int, dict] = {}
    BATCH_SIZE = 5  # process 5 pages at a time
    
    # Process in batches
    for batch_start in range(0, total_pages, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_pages)
        page_range = list(range(batch_start, batch_end))
        page_labels = f"pages {batch_start + 1}-{batch_end}"
        
        # Create a temp PDF with just this batch of pages
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_pdf_path = Path(tmp.name)
        
        try:
            batch_doc = fitz.open()
            batch_doc.insert_pdf(src_doc, from_page=batch_start, to_page=batch_end - 1)
            batch_doc.save(str(tmp_pdf_path))
            batch_doc.close()
            
            # Try to parse this batch
            logger.info(f"    Processing batch: {page_labels}")
            
            try:
                result = converter.convert(str(tmp_pdf_path))
                batch_pages = _extract_pages_from_result(result, page_offset=batch_start)
                pages_dict.update(batch_pages)
                logger.info(f"    ✅ Batch {page_labels} OK ({len(batch_pages)} pages extracted)")
            except Exception as e:
                msg = f"Batch {page_labels} failed: {type(e).__name__}: {e}"
                warnings.append(msg)
                logger.warning(f"    ⚠️  {msg}")
                # Try to extract pages one-by-one within this batch
                logger.info(f"    Falling back to per-page processing for {page_labels}")
                for page_idx in page_range:
                    try:
                        single_pdf = Path(tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name)
                        single_doc = fitz.open()
                        single_doc.insert_pdf(src_doc, from_page=page_idx, to_page=page_idx)
                        single_doc.save(str(single_pdf))
                        single_doc.close()
                        
                        result = converter.convert(str(single_pdf))
                        single_pages = _extract_pages_from_result(result, page_offset=page_idx)
                        pages_dict.update(single_pages)
                        logger.info(f"      ✅ Page {page_idx + 1} OK (single-page mode)")
                        single_pdf.unlink(missing_ok=True)
                    except Exception as e2:
                        warnings.append(f"Page {page_idx + 1} unrecoverable: {e2}")
                        logger.warning(f"      ❌ Page {page_idx + 1} unrecoverable")
        finally:
            tmp_pdf_path.unlink(missing_ok=True)
    
    src_doc.close()
    return pages_dict


def _extract_pages_from_result(result, page_offset: int = 0) -> dict[int, dict]:
    """Extract pages from a Docling conversion result.
    page_offset is added to page numbers to map back to the original PDF."""
    pages: dict[int, dict] = {}
    document = result.document
    
    for item, _level in document.iterate_items():
        page_no = extract_page_number(item)
        if page_no is None:
            continue
        
        # Map batch-local page number to original PDF page number
        original_page_no = page_no + page_offset
        
        if original_page_no not in pages:
            pages[original_page_no] = {"text_blocks": [], "tables": []}
        
        item_type = type(item).__name__
        
        if "Table" in item_type:
            try:
                table = parse_docling_table(item, document)
                pages[original_page_no]["tables"].append(table)
            except Exception as e:
                logger.debug(f"Table parse error on page {original_page_no}: {e}")
        elif hasattr(item, "text") and item.text:
            text = item.text.strip()
            if not text:
                continue
            pages[original_page_no]["text_blocks"].append(
                TextBlock(
                    content=text,
                    block_type=classify_block_type(item_type),
                    bbox=extract_bbox(item),
                )
            )
    
    return pages