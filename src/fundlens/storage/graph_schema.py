"""
Neo4j schema bootstrap — constraints, indexes, controlled vocabulary.

Run once on a fresh database (or anytime — operations are idempotent).
"""

import logging
from fundlens.storage.graph_client import neo4j_session

logger = logging.getLogger(__name__)


# ============================================================
# CONTROLLED VOCABULARY — used for Topic node creation
# Keeping topics in a fixed set makes graph queries consistent.
# ============================================================

CANONICAL_TOPICS = [
    "AI",                       # Artificial intelligence, ML, LLMs
    "Cloud",                    # Cloud computing platforms
    "Cybersecurity",
    "Regulation",               # Antitrust, privacy, compliance
    "Macroeconomic",            # Inflation, recession, geopolitics
    "Supply Chain",
    "ESG",                      # Environmental, social, governance
    "Hardware",                 # Chips, devices, semiconductors
    "Subscription Services",
    "Advertising",
    "Competition",              # Competitive landscape mentions
    "M&A",                      # Mergers & acquisitions
    "Capital Returns",          # Buybacks, dividends
    "International Expansion",
    "Litigation",
]


# ============================================================
# CONSTRAINTS & INDEXES
# ============================================================

CONSTRAINTS = [
    # Uniqueness — these are the natural keys for each node type
    "CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
    "CREATE CONSTRAINT executive_id IF NOT EXISTS FOR (e:Executive) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT filing_id IF NOT EXISTS FOR (f:Filing) REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX company_sector IF NOT EXISTS FOR (c:Company) ON (c.sector)",
    "CREATE INDEX executive_name IF NOT EXISTS FOR (e:Executive) ON (e.name)",
    "CREATE INDEX segment_name IF NOT EXISTS FOR (s:Segment) ON (s.name)",
    "CREATE INDEX filing_year IF NOT EXISTS FOR (f:Filing) ON (f.fiscal_year)",
]


def bootstrap_schema():
    """Create all constraints and indexes. Idempotent."""
    logger.info("🗄️  Bootstrapping Neo4j schema...")
    with neo4j_session() as session:
        for stmt in CONSTRAINTS:
            session.run(stmt)
        for stmt in INDEXES:
            session.run(stmt)

        # Pre-create canonical topic nodes
        for topic in CANONICAL_TOPICS:
            session.run(
                "MERGE (t:Topic {name: $name}) ON CREATE SET t.created_at = datetime()",
                name=topic,
            )

    logger.info(f"✅ Schema ready: {len(CONSTRAINTS)} constraints, "
                f"{len(INDEXES)} indexes, {len(CANONICAL_TOPICS)} topics")