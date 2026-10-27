"""Conservative PostgreSQL read-only SQL validation.

A syntax-tree check is defense in depth, never a substitute for least-privilege
PostgreSQL credentials and a READ ONLY transaction.
"""

ALLOWED_RELATIONS = {
    "companies", "filing_periods", "filings", "financial_metrics",
    "segment_metrics", "executive_commentary", "v_financials_wide"
}
ALLOWED_FUNCTIONS = {
    "abs", "avg", "count", "sum", "min", "max", "round", "coalesce",
    "nullif", "lower", "upper", "trim", "date_trunc", "to_char",
    "greatest", "least", "string_agg", "concat", "concat_ws"
}


def validate_read_only_sql(sql: str) -> str:
    """Return validated SQL or raise ValueError; refuses execution without parser."""
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL query must not be empty")
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError as exc:
        raise ValueError("SQL safety parser unavailable; install project dependencies") from exc
    try:
        statements=sqlglot.parse(sql, read="postgres")
    except Exception as exc:
        raise ValueError("Invalid SQL syntax") from exc
    if len(statements)!=1 or statements[0] is None:
        raise ValueError("Only one SELECT statement is allowed")
    tree=statements[0]
    if not isinstance(tree,(exp.Select,exp.Union,exp.Except,exp.Intersect)):
        raise ValueError("Only SELECT queries are allowed")
    blocked_types=[getattr(exp,name) for name in (
        "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter",
        "Command", "Transaction", "Grant", "Revoke", "Copy", "Into",
        "Lock", "Set", "Use", "TruncateTable"
    ) if hasattr(exp,name)]
    cte_names={cte.alias.lower() for cte in tree.find_all(exp.CTE) if cte.alias}
    for node in tree.walk():
        if isinstance(node,tuple(blocked_types)):
            raise ValueError("Write operations and unsafe SQL commands are not allowed")
        if isinstance(node,exp.Table):
            schema=(node.db or "").lower()
            name=node.name.lower()
            if name in cte_names and not schema:
                continue
            if name not in ALLOWED_RELATIONS or schema not in ("", "fundlens"):
                raise ValueError("Query references an unauthorized table")
        if isinstance(node,exp.Anonymous) and node.name.lower() not in ALLOWED_FUNCTIONS:
            raise ValueError("SQL function is not on the approved list")
    return sql.strip().rstrip(";")
