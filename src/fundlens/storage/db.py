"""
SQLAlchemy database connection + session management.

This is the single entry point for all database access in fundlens.
Use the get_session() context manager in all data-access code.

Usage:
    from storage.db import get_session
    from storage.models import Company

    with get_session() as session:
        company = Company(ticker="AAPL", name="Apple Inc.")
        session.add(company)
        session.commit()
"""

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker, declarative_base

from fundlens.config import settings


# ============================================================
# ENGINE — singleton, created once per app process
# ============================================================

def _build_engine() -> Engine:
    """
    Build the SQLAlchemy engine with sensible production defaults.

    Connection pooling means we reuse connections instead of
    opening a new one for every query (huge perf win).
    """
    return create_engine(
        settings.postgres_url,
        pool_size=10,           # base pool of 10 connections
        max_overflow=20,        # burst up to 30 total
        pool_pre_ping=True,     # verify connection before using (handles DB restarts)
        pool_recycle=3600,      # recycle connections after 1 hour
        echo=settings.app_debug,  # log all SQL in dev mode
        future=True,            # use 2.0-style API
        connect_args={"options": "-csearch_path=fundlens,public"},
    )


engine: Engine = _build_engine()


# ============================================================
# SESSION FACTORY
# ============================================================

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,    # objects stay usable after commit
)


# ============================================================
# BASE CLASS — all models inherit from this
# ============================================================

Base = declarative_base()


# ============================================================
# CONTEXT MANAGER — preferred API for getting a session
# ============================================================

@contextmanager
def get_session() -> Iterator[Session]:
    """
    Yield a database session with automatic commit/rollback.

    Example:
        with get_session() as session:
            session.add(some_object)
            # auto-commits on exit, auto-rollbacks on exception
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ============================================================
# UTILITY — quick health check
# ============================================================

def ping() -> bool:
    """Return True if database is reachable."""
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False