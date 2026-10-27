"""
SQL Executor: runs validated SQL against Postgres and returns structured results.

Handles:
  - Read-only execution (raises on any non-SELECT)
  - Structured error info (so the self-correction loop knows what went wrong)
  - Query timeout (5 sec default — prevents runaway queries)
  - Row limit safety (caps at 1000 rows even if SQL forgot LIMIT)
"""

import logging
import time
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from fundlens.storage.db import get_session
from fundlens.models.sql import SQLResult, SQLExecutionError
from fundlens.utils.sql_safety import validate_read_only_sql

logger = logging.getLogger(__name__)


# ============================================================
# SAFETY CONSTANTS
# ============================================================

QUERY_TIMEOUT_MS = 5000          # 5-second timeout
MAX_ROWS = 1000                  # hard cap, even if SQL has no LIMIT


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def execute_sql(sql: str) -> SQLResult:
    """
    Execute a SQL query against Postgres and return structured results.
    
    Args:
        sql: A validated SELECT query.
    
    Returns:
        SQLResult with rows, column names, timing, or error.
    """
    start = time.time()
    logger.info(f"🗄️  Executing SQL...")
    
    try:
        sql = validate_read_only_sql(sql)
        with get_session() as session:
            # Independent PostgreSQL enforcement: even parser bypasses cannot write.
            session.execute(text("SET TRANSACTION READ ONLY"))
            session.execute(text("SELECT set_config('statement_timeout', :ms, true)"), {"ms": f"{QUERY_TIMEOUT_MS}ms"})
            result = session.execute(text(sql))
            column_names = list(result.keys())
            
            # Fetch up to MAX_ROWS
            raw_rows = result.fetchmany(MAX_ROWS)
            rows = [dict(zip(column_names, row)) for row in raw_rows]
            
            # Convert non-JSON-serializable types
            rows = [_serialize_row(row) for row in rows]
            
            duration_ms = (time.time() - start) * 1000
            logger.info(f"  ✅ {len(rows)} rows in {duration_ms:.0f}ms")
            
            return SQLResult(
                success=True,
                rows=rows,
                row_count=len(rows),
                column_names=column_names,
                execution_time_ms=duration_ms,
            )
    
    except SQLAlchemyError as e:
        duration_ms = (time.time() - start) * 1000
        error_type = type(e.orig).__name__ if hasattr(e, 'orig') and e.orig else type(e).__name__
        error_msg = str(e.orig) if hasattr(e, 'orig') and e.orig else str(e)
        
        logger.error(f"  ❌ {error_type}: {error_msg[:200]}")
        
        return SQLResult(
            success=False,
            execution_time_ms=duration_ms,
            error=SQLExecutionError(
                error_type=error_type,
                error_message=error_msg,
                failed_sql=sql,
            ),
        )
    
    except Exception as e:
        duration_ms = (time.time() - start) * 1000
        logger.exception(f"  ❌ Unexpected error: {e}")
        
        return SQLResult(
            success=False,
            execution_time_ms=duration_ms,
            error=SQLExecutionError(
                error_type=type(e).__name__,
                error_message=str(e),
                failed_sql=sql,
            ),
        )


# ============================================================
# SERIALIZATION HELPERS
# ============================================================

def _serialize_row(row: dict) -> dict:
    """Convert Decimal, datetime, etc. to JSON-serializable types."""
    from decimal import Decimal
    from datetime import date, datetime
    
    result = {}
    for key, value in row.items():
        if isinstance(value, Decimal):
            result[key] = float(value)
        elif isinstance(value, (date, datetime)):
            result[key] = value.isoformat()
        else:
            result[key] = value
    return result