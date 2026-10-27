"""Regression tests for fail-closed LLM-generated SQL validation."""
import pytest
from fundlens.utils.sql_safety import validate_read_only_sql


@pytest.mark.parametrize("sql", [
    "SELECT c.ticker FROM fundlens.companies c LIMIT 5",
    "SELECT 'DROP' AS label FROM fundlens.companies LIMIT 1",
    "WITH c AS (SELECT ticker FROM fundlens.companies) SELECT * FROM c LIMIT 5",
])
def test_safe_sql(sql):
    pytest.importorskip("sqlglot")
    assert validate_read_only_sql(sql)


@pytest.mark.parametrize("sql", [
    "DELETE FROM fundlens.companies",
    "SELECT * FROM fundlens.companies; DELETE FROM fundlens.companies",
    "WITH x AS (DELETE FROM fundlens.companies RETURNING *) SELECT * FROM x",
    "SELECT * FROM pg_catalog.pg_users",
    "SELECT pg_sleep(5) FROM fundlens.companies",
    "SELECT * INTO backup FROM fundlens.companies",
])
def test_unsafe_sql_rejected(sql):
    pytest.importorskip("sqlglot")
    with pytest.raises(ValueError):
        validate_read_only_sql(sql)
