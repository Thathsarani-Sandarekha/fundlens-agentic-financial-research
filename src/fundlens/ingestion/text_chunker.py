"""
Narrative text chunking with section awareness and contextual enrichment.

Strategy:
  1. Walk pages in order, identifying section boundaries via heading detection
  2. Group text blocks by detected section
  3. Within each section, chunk text into ~512-token windows with 50-token overlap
  4. Enrich each chunk with metadata + a contextual prefix for better retrieval

Usage:
    from ingestion.text_chunker import chunk_document
    chunks = chunk_document(
        parsed_doc,
        ticker="GOOGL",
        fiscal_year=2025,
        fiscal_quarter="Q1",
        filing_type="10-Q",
        filing_id="GOOGL_2025_Q1_10Q",
    )
"""

import hashlib
import logging
import re
from typing import Optional

import tiktoken

from fundlens.models.parsed import ParsedDocument, ParsedPage, TextBlock
from fundlens.models.chunk import TextChunk

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

# Target chunk size — fits comfortably in most embedding model contexts
CHUNK_TARGET_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50
CHUNK_MIN_TOKENS = 50  # discard tiny fragments

# Encoding for token counting (works for OpenAI + most modern models)
TOKEN_ENCODING = "cl100k_base"
_encoder = tiktoken.get_encoding(TOKEN_ENCODING)


# ============================================================
# SECTION DETECTION
# ============================================================

# Common SEC filing sections — patterns matched against header text
# Order matters: more specific patterns first.
SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    # 10-K / 10-Q standard sections (Item-based)
    ("Risk Factors",          re.compile(r"item\s*1a[\.\s]*risk\s*factors", re.IGNORECASE)),
    ("MD&A",                  re.compile(r"item\s*[27][\.\s]*management.{0,40}discussion", re.IGNORECASE)),
    ("Financial Statements",  re.compile(r"item\s*[18][\.\s]*(financial\s*statements|condensed)", re.IGNORECASE)),
    ("Quantitative Risk",     re.compile(r"item\s*[37a][\.\s]*quantitative.{0,40}market\s*risk", re.IGNORECASE)),
    ("Controls and Procedures", re.compile(r"item\s*4[\.\s]*controls\s*and\s*procedures", re.IGNORECASE)),
    ("Legal Proceedings",     re.compile(r"item\s*1[\.\s]*legal\s*proceedings", re.IGNORECASE)),
    ("Properties",            re.compile(r"item\s*2[\.\s]*properties", re.IGNORECASE)),
    ("Business",              re.compile(r"item\s*1[\.\s]*business", re.IGNORECASE)),
    ("Executive Compensation", re.compile(r"item\s*11[\.\s]*executive\s*compensation", re.IGNORECASE)),

    # Loose patterns (catch headings without "Item N")
    ("MD&A",                  re.compile(r"management.{0,30}discussion.{0,30}analysis", re.IGNORECASE)),
    ("Risk Factors",          re.compile(r"^risk\s*factors", re.IGNORECASE)),
    ("Forward-Looking Statements", re.compile(r"forward[-\s]*looking\s*statements?", re.IGNORECASE)),
    ("Notes to Financial Statements", re.compile(r"notes\s*to.{0,30}financial\s*statements", re.IGNORECASE)),

    # Earnings call / shareholder letter patterns
    ("CEO Letter",            re.compile(r"(letter\s*(to|from).*(shareholders|ceo)|chairman.{0,20}letter)", re.IGNORECASE)),

    # Cover / preamble
    ("Cover Page",            re.compile(r"(form\s*10[-\s]*[kq]|united\s*states\s*securities)", re.IGNORECASE)),
]

DEFAULT_SECTION = "Other"


def detect_section(text: str) -> Optional[str]:
    """Return the section name if text matches any known section pattern."""
    text_clean = text.strip()
    if not text_clean or len(text_clean) > 200:
        return None
    for section_name, pattern in SECTION_PATTERNS:
        if pattern.search(text_clean):
            return section_name
    return None

# ============================================================
# CHUNK BUILDING
# ============================================================

def count_tokens(text: str) -> int:
    """Count tokens using the configured encoder."""
    return len(_encoder.encode(text))


def encode_to_tokens(text: str) -> list[int]:
    return _encoder.encode(text)


def decode_from_tokens(tokens: list[int]) -> str:
    return _encoder.decode(tokens)


def build_chunk_id(filing_id: str, section: str, page_number: int, content: str) -> str:
    """Deterministic ID — same chunk text → same ID. Useful for idempotent re-runs."""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    safe_section = re.sub(r"[^a-z0-9]", "_", section.lower())[:20]
    return f"{filing_id}__{safe_section}__p{page_number}__{content_hash}"


def build_context_prefix(
    *,
    ticker: str,
    fiscal_year: int,
    fiscal_quarter: str,
    filing_type: str,
    section: str,
    page_number: int,
) -> str:
    """
    Generate a brief static context line to prepend to each chunk.

    This is "Contextual Retrieval" (Anthropic 2024) — adding context
    significantly improves retrieval recall on the embedded side.

    We use static metadata-derived context (no LLM call) to keep cost
    at zero per chunk.
    """
    period_label = f"{fiscal_quarter} {fiscal_year}"
    return (
        f"This excerpt is from {ticker}'s {period_label} {filing_type} filing, "
        f"in the {section} section (page {page_number}). Content: "
    )


# ============================================================
# CORE SLIDING-WINDOW CHUNKER
# ============================================================

def sliding_window_chunks(
    text: str,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    min_tokens: int = CHUNK_MIN_TOKENS,
) -> list[tuple[str, bool]]:
    """
    Split text into overlapping token windows.
    Returns: list of (chunk_text, is_overlap_chunk) tuples.

    Strategy: token-aware sliding window. We do NOT try to split on
    sentence boundaries inside a single text block — that's expensive
    and the overlap mitigates mid-sentence cuts.
    """
    tokens = encode_to_tokens(text)
    if len(tokens) < min_tokens:
        return []  # too small to be useful

    if len(tokens) <= target_tokens:
        return [(text, False)]  # whole thing fits — one chunk, no overlap

    chunks: list[tuple[str, bool]] = []
    stride = target_tokens - overlap_tokens

    start = 0
    while start < len(tokens):
        end = min(start + target_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = decode_from_tokens(chunk_tokens).strip()
        is_overlap = start > 0
        if len(chunk_tokens) >= min_tokens:
            chunks.append((chunk_text, is_overlap))
        if end >= len(tokens):
            break
        start += stride

    return chunks


# ============================================================
# SECTION-AWARE GROUPING
# ============================================================

def group_blocks_by_section(pages: list[ParsedPage]) -> list[dict]:
    """
    Walk all pages in order, partition text blocks into section groups.

    Returns a list of dicts:
        [
          {
            "section": "MD&A",
            "blocks": [
              {"page": 23, "text": "..."},
              {"page": 24, "text": "..."},
              ...
            ]
          },
          ...
        ]
    """
    groups: list[dict] = []
    current_section = DEFAULT_SECTION
    current_group = {"section": current_section, "blocks": []}

    for page in sorted(pages, key=lambda p: p.page_number):
        for block in page.text_blocks:
            text = block.content.strip()
            if not text:
                continue

            # If this looks like a section header, switch sections
            if block.block_type == "header":
                detected = detect_section(text)
                if detected and detected != current_section:
                    # Close previous group if it has content
                    if current_group["blocks"]:
                        groups.append(current_group)
                    current_section = detected
                    current_group = {"section": current_section, "blocks": []}
                    logger.debug(f"  📑 New section: {detected} (page {page.page_number})")
                    # Include the header itself as a block
                    current_group["blocks"].append({"page": page.page_number, "text": text})
                    continue

            current_group["blocks"].append({"page": page.page_number, "text": text})

    # Don't forget the final group
    if current_group["blocks"]:
        groups.append(current_group)

    return groups


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def chunk_document(
    parsed_doc: ParsedDocument,
    *,
    ticker: str,
    fiscal_year: int,
    fiscal_quarter: str,
    filing_type: str,
    filing_id: str,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[TextChunk]:
    """
    Convert a ParsedDocument into a list of enriched TextChunks
    ready for embedding + storage.
    """
    logger.info(
        f"📝 Chunking {ticker} {fiscal_quarter} {fiscal_year} {filing_type} "
        f"({len(parsed_doc.pages)} pages parsed)"
    )

    # 1. Group blocks by section
    section_groups = group_blocks_by_section(parsed_doc.pages)
    logger.info(f"  Found {len(section_groups)} sections")
    for g in section_groups:
        logger.debug(f"    - {g['section']:30s} ({len(g['blocks'])} blocks)")

    # 2. Within each section, chunk the text
    all_chunks: list[TextChunk] = []
    chunk_index = 0

    for group in section_groups:
        section = group["section"]
        blocks = group["blocks"]
        if not blocks:
            continue

        # Concatenate blocks within the section. Track page mapping so we
        # know which page each character came from.
        # Strategy: paragraph blocks → join with double newline.
        section_text_parts: list[str] = []
        page_markers: list[tuple[int, int]] = []  # (char_offset, page_number)
        offset = 0
        for block in blocks:
            page_markers.append((offset, block["page"]))
            section_text_parts.append(block["text"])
            offset += len(block["text"]) + 2  # for the "\n\n"

        section_text = "\n\n".join(section_text_parts)

        # Now chunk this section's combined text
        windows = sliding_window_chunks(
            section_text,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )

        for raw_chunk, is_overlap in windows:
            # Find which page this chunk starts on (approximate but good enough)
            chunk_start = section_text.find(raw_chunk[:50])
            page_for_chunk = blocks[0]["page"]  # fallback
            for marker_offset, marker_page in page_markers:
                if marker_offset <= max(chunk_start, 0):
                    page_for_chunk = marker_page
                else:
                    break

            # Build contextual prefix
            prefix = build_context_prefix(
                ticker=ticker,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
                filing_type=filing_type,
                section=section,
                page_number=page_for_chunk,
            )

            enriched_content = prefix + raw_chunk

            chunk = TextChunk(
                chunk_id=build_chunk_id(filing_id, section, page_for_chunk, raw_chunk),
                content=enriched_content,
                raw_content=raw_chunk,
                filing_id=filing_id,
                company_ticker=ticker,
                filing_type=filing_type,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
                section=section,
                page_number=page_for_chunk,
                token_count=count_tokens(enriched_content),
                char_count=len(enriched_content),
                chunk_index=chunk_index,
                overlap_with_prev=is_overlap,
            )

            all_chunks.append(chunk)
            chunk_index += 1

    logger.info(f"✅ Produced {len(all_chunks)} chunks across {len(section_groups)} sections")
    return all_chunks


# ============================================================
# SUMMARY / DEBUG UTILITY
# ============================================================

def summarize_chunks(chunks: list[TextChunk]) -> dict:
    """Return summary stats useful for testing and reporting."""
    if not chunks:
        return {"total_chunks": 0}

    by_section: dict[str, int] = {}
    total_tokens = 0
    for c in chunks:
        by_section[c.section] = by_section.get(c.section, 0) + 1
        total_tokens += c.token_count

    return {
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "avg_tokens_per_chunk": round(total_tokens / len(chunks), 1),
        "min_tokens": min(c.token_count for c in chunks),
        "max_tokens": max(c.token_count for c in chunks),
        "chunks_with_overlap": sum(1 for c in chunks if c.overlap_with_prev),
        "by_section": dict(sorted(by_section.items(), key=lambda x: -x[1])),
        "pages_covered": len(set(c.page_number for c in chunks)),
    }