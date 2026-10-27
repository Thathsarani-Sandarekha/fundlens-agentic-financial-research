"""Financial mapping helpers should never silently assume a currency scale."""
import pytest
from fundlens.models.verified import VerifiedTable, TableRow


def sample(units="thousands", currency="USD"):
    return VerifiedTable(table_type="income_statement", headers=["2025"],
      rows=[TableRow(label="Revenue",values={"2025":100000.0}),
            TableRow(label="Earnings per share",values={"2025":1.5})],
      units=units,currency=currency,confidence_score=.98)


def test_units_contract():
    obj=sample()
    assert obj.units=="thousands"
    assert obj.rows[0].values["2025"]==100000.0


def test_scale_monetary_but_not_eps():
    pytest.importorskip("sqlalchemy")
    # Financial mapper depends on the uninstalled Groq API in some offline environments.
    pytest.importorskip("groq")
    pytest.importorskip("instructor")
    from fundlens.ingestion.financial_mapper import normalize_verified_financial_table
    original=sample()
    scaled=normalize_verified_financial_table(original)
    assert scaled.rows[0].values["2025"]==pytest.approx(100.0)
    assert scaled.rows[1].values["2025"]==1.5
    assert original.rows[0].values["2025"]==100000.0


def test_non_usd_rejected():
    pytest.importorskip("groq")
    pytest.importorskip("instructor")
    from fundlens.ingestion.financial_mapper import normalize_verified_financial_table
    with pytest.raises(ValueError):
        normalize_verified_financial_table(sample(currency="EUR"))
