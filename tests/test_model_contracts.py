"""Small, offline validation tests for the reviewed ingestion contracts."""
import pytest
from pydantic import ValidationError
from fundlens.models.verified import TableRow, VerifiedTable
from fundlens.models.parsed import ParsedDocument, ParsedPage, TextBlock
from fundlens.models.chunk import TextChunk


def verified_sample(**overrides):
    data = dict(table_type="income_statement", headers=["Q1 2025"],
                rows=[TableRow(label="Revenue", values={"Q1 2025": 95_359.0})],
                units="millions", currency="usd", confidence_score=0.95)
    data.update(overrides)
    return data


def chunk_sample(**overrides):
    data = dict(chunk_id="a1", content="Revenue: 95,359", raw_content="Revenue: 95,359",
                filing_id="AAPL_2025_Q1", company_ticker="AAPL", filing_type="10-Q",
                fiscal_year=2025, fiscal_quarter="Q1", section="MD&A", page_number=1,
                token_count=5, char_count=15, chunk_index=0)
    data.update(overrides)
    return data


def test_explicit_table_units_and_currency_normalization():
    result = VerifiedTable(**verified_sample())
    assert result.units == "millions"
    assert result.currency == "USD"
    assert result.is_high_confidence


def test_missing_units_rejected_instead_of_assuming_millions():
    data = verified_sample()
    del data["units"]
    with pytest.raises(ValidationError):
        VerifiedTable(**data)


def test_nonfinite_financial_values_rejected():
    with pytest.raises(ValidationError):
        TableRow(label="Revenue", values={"Q1 2025": float("nan")})


def test_confidence_bounds_enforced():
    with pytest.raises(ValidationError):
        VerifiedTable(**verified_sample(confidence_score=1.5))


def test_parsed_document_summary_and_counts():
    page = ParsedPage(page_number=1, text_blocks=[TextBlock(content="a", block_type="narrative")])
    doc = ParsedDocument(filing_id="a", company_ticker="AAPL", filing_type="10-Q",
                         s3_bucket="filings", s3_key="a.pdf", pages=[page],
                         total_pages_in_pdf=3, pages_parsed_successfully=1)
    assert doc.parse_success_rate == pytest.approx(1/3)
    assert doc.total_text_blocks == 1


def test_impossible_page_totals_rejected():
    with pytest.raises(ValidationError):
        ParsedDocument(filing_id="a", company_ticker="AAPL", filing_type="10-Q",
                       s3_bucket="filings", s3_key="a.pdf", total_pages_in_pdf=2,
                       pages_parsed_successfully=3)


@pytest.mark.parametrize("field,value", [("page_number", 0), ("token_count", -1),
                                         ("char_count", -1), ("chunk_index", -1)])
def test_invalid_chunk_values_rejected(field, value):
    with pytest.raises(ValidationError):
        TextChunk(**chunk_sample(**{field: value}))


def test_valid_chunk_accepted():
    assert TextChunk(**chunk_sample()).chunk_index == 0
