"""
Centralized Groq + Instructor client with retry/backoff.

Replaces the duplicated `_client` / `get_client()` singletons that lived in
table_extractor.py, financial_mapper.py, and graph_builder.py.

Usage:
    from fundlens.llm.groq_client import get_instructor_client, call_with_backoff
    
    client = get_instructor_client()
    result = client.chat.completions.create(
        model="...",
        response_model=MyPydanticModel,
        messages=[...],
    )
"""

from functools import lru_cache
import logging

import instructor
from groq import Groq, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from fundlens.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_instructor_client():
    """
    Singleton Instructor-wrapped Groq client (TOOLS mode).
    
    TOOLS mode is more token-efficient than JSON mode because the
    Pydantic schema is sent via the API's function-definition mechanism
    rather than inlined in the prompt text. Saves ~1500 tokens per call.
    """
    settings.require_groq()
    raw = Groq(api_key=settings.groq_api_key.get_secret_value(), timeout=settings.llm_timeout_seconds)
    return instructor.from_groq(raw, mode=instructor.Mode.TOOLS)


@lru_cache(maxsize=1)
def get_raw_client() -> Groq:
    """
    Raw Groq client for non-structured calls.
    
    Use this when you don't need Pydantic schema enforcement.
    """
    settings.require_groq()
    return Groq(api_key=settings.groq_api_key.get_secret_value(), timeout=settings.llm_timeout_seconds)


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def call_with_backoff(client, **kwargs):
    """
    Call Groq with exponential backoff on rate-limit errors.
    
    Retries only on rate-limit errors.
    Only sleeps when actually rate-limited, otherwise calls immediately.
    
    Usage:
        result = call_with_backoff(
            client,
            model="...",
            response_model=MyModel,
            messages=[...],
        )
    """
    return client.chat.completions.create(**kwargs)