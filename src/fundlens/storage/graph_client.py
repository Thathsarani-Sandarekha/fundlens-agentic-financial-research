"""
Neo4j connection wrapper — singleton driver, session helpers.
"""

import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from neo4j import GraphDatabase, Driver, Session

from fundlens.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_driver() -> Driver:
    """Return a singleton Neo4j driver."""
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )
    # Smoke test connection
    driver.verify_connectivity()
    return driver


@contextmanager
def neo4j_session() -> Iterator[Session]:
    """Context manager for a Neo4j session."""
    driver = get_driver()
    session = driver.session()
    try:
        yield session
    finally:
        session.close()


def close_driver():
    """Call at application shutdown."""
    if get_driver.cache_info().currsize > 0:
        get_driver().close()
        get_driver.cache_clear()


def ping() -> bool:
    """Health check."""
    try:
        get_driver().verify_connectivity()
        return True
    except Exception:
        return False