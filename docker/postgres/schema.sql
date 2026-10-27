-- FundLens fresh database initialization (non-destructive).
-- NOT an in-place schema migration solution.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS fundlens;
SET search_path TO fundlens, public;

-- ============================================================
-- fundlens / FundLens Research — Postgres Schema
-- 
-- PURPOSE:
--   Target schema for NL→SQL pipeline. The LLM will write
--   queries against THIS schema, so every table/column is
--   designed to be self-documenting and query-friendly.
--
-- DESIGN PRINCIPLES:
--   1. Snake_case everywhere (Postgres convention)
--   2. Singular concept tables, plural fact tables
--   3. Pre-compute YoY/QoQ growth (don't make LLM write windows)
--   4. Foreign keys cascade DELETE for clean re-ingestion
--   5. Comments on every column (shown to LLM via schema RAG)
--   6. Standardized enum values (segment names, sectors)
--   7. Source provenance tracked everywhere (page_number, source)
--
-- USAGE:
--   psql -U fundlens_user -d fundlens -f storage/schema.sql
-- ============================================================

-- ============================================================
-- ENUMS — constrain values for clean NL→SQL filtering
-- ============================================================

DO $fundlens_enum$ BEGIN
 IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
                WHERE t.typname='filing_type_enum' AND n.nspname='fundlens') THEN
  CREATE TYPE fundlens.filing_type_enum AS ENUM ('10-K', '10-Q', '8-K');
 END IF;
END $fundlens_enum$;
DO $fundlens_enum$ BEGIN
 IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
                WHERE t.typname='fiscal_quarter_enum' AND n.nspname='fundlens') THEN
  CREATE TYPE fundlens.fiscal_quarter_enum AS ENUM ('Q1', 'Q2', 'Q3', 'Q4', 'FY');
 END IF;
END $fundlens_enum$;
DO $fundlens_enum$ BEGIN
 IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
                WHERE t.typname='market_cap_bucket_enum' AND n.nspname='fundlens') THEN
  CREATE TYPE fundlens.market_cap_bucket_enum AS ENUM (
    'mega_cap',     -- > $200B
    'large_cap',    -- $10B - $200B
    'mid_cap',      -- $2B - $10B
    'small_cap'     -- < $2B
);
 END IF;
END $fundlens_enum$;
DO $fundlens_enum$ BEGIN
 IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
                WHERE t.typname='sentiment_enum' AND n.nspname='fundlens') THEN
  CREATE TYPE fundlens.sentiment_enum AS ENUM ('positive', 'neutral', 'negative', 'mixed');
 END IF;
END $fundlens_enum$;
DO $fundlens_enum$ BEGIN
 IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
                WHERE t.typname='commentary_topic_enum' AND n.nspname='fundlens') THEN
  CREATE TYPE fundlens.commentary_topic_enum AS ENUM (
    'revenue',
    'earnings',
    'profitability',
    'guidance',
    'segments',
    'macroeconomic',
    'product',
    'competition',
    'ai_strategy',
    'cost_management',
    'capital_allocation',
    'risk_factor',
    'other'
);
 END IF;
END $fundlens_enum$;


-- ============================================================
-- TABLE: companies
-- 
-- One row per public company tracked by the platform.
-- This is the master dimension that everything joins on.
-- ============================================================

CREATE TABLE IF NOT EXISTS companies (
    id                  SERIAL PRIMARY KEY,
    ticker              VARCHAR(10) UNIQUE NOT NULL,
    name                VARCHAR(255) NOT NULL,
    sector              VARCHAR(100),
    industry            VARCHAR(100),
    market_cap_bucket   market_cap_bucket_enum,
    cik                 VARCHAR(20),                          -- SEC Central Index Key
    fiscal_year_end_month SMALLINT CHECK (fiscal_year_end_month BETWEEN 1 AND 12),
    headquarters_country VARCHAR(50) DEFAULT 'USA',
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE companies IS
    'Master list of public companies tracked. Joins to all other tables via id.';
COMMENT ON COLUMN companies.ticker IS
    'Stock ticker symbol, e.g., AAPL, GOOGL. Use this in WHERE clauses for company lookup.';
COMMENT ON COLUMN companies.sector IS
    'GICS sector, e.g., Technology, Financials, Healthcare';
COMMENT ON COLUMN companies.market_cap_bucket IS
    'Categorical: mega_cap (>$200B), large_cap, mid_cap, small_cap';
COMMENT ON COLUMN companies.cik IS
    'SEC Central Index Key — unique SEC identifier';

CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies(ticker);
CREATE INDEX IF NOT EXISTS idx_companies_sector ON companies(sector);


-- ============================================================
-- TABLE: filing_periods
--
-- Time dimension. Each row is one (fiscal_year, fiscal_quarter)
-- combination and actual end date. Companies may have different
-- fiscal year-end calendars.
-- ============================================================

CREATE TABLE IF NOT EXISTS filing_periods (
    id                  SERIAL PRIMARY KEY,
    fiscal_year         SMALLINT NOT NULL,
    fiscal_quarter      fiscal_quarter_enum NOT NULL,         -- Q1, Q2, Q3, Q4, FY (annual)
    period_end_date     DATE NOT NULL,                        -- last day of the period
    filing_type         filing_type_enum NOT NULL,            -- 10-K (annual) or 10-Q (quarterly)
    period_label        VARCHAR(20),                          -- "Q3 2024" - auto-set by trigger
    created_at          TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (fiscal_year, fiscal_quarter, filing_type, period_end_date)
);

-- Trigger to auto-populate period_label on INSERT/UPDATE
-- (Postgres doesn't allow generated columns from enum types)
CREATE OR REPLACE FUNCTION fn_set_period_label()
RETURNS TRIGGER AS $$
BEGIN
    NEW.period_label = NEW.fiscal_quarter::TEXT || ' ' || NEW.fiscal_year::TEXT;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_filing_periods_set_label ON filing_periods;
CREATE TRIGGER trg_filing_periods_set_label
    BEFORE INSERT OR UPDATE OF fiscal_year, fiscal_quarter ON filing_periods
    FOR EACH ROW EXECUTE FUNCTION fn_set_period_label();

COMMENT ON TABLE filing_periods IS
    'Time dimension. (Fiscal year, quarter, filing type, actual period-end date).';
COMMENT ON COLUMN filing_periods.fiscal_quarter IS
    'Q1, Q2, Q3, Q4 for 10-Q filings. FY for annual 10-K filings.';
COMMENT ON COLUMN filing_periods.period_label IS
    'Auto-generated readable label, e.g., "Q3 2024". Useful for display.';

CREATE INDEX IF NOT EXISTS idx_filing_periods_year ON filing_periods(fiscal_year DESC);
CREATE INDEX IF NOT EXISTS idx_filing_periods_quarter ON filing_periods(fiscal_quarter);
CREATE INDEX IF NOT EXISTS idx_filing_periods_end_date ON filing_periods(period_end_date DESC);


-- ============================================================
-- TABLE: filings
--
-- One row per actual SEC document. The bridge between
-- (company, period) and the physical PDF in S3.
-- ============================================================

CREATE TABLE IF NOT EXISTS filings (
    id                  SERIAL PRIMARY KEY,
    company_id          INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    period_id           INTEGER NOT NULL REFERENCES filing_periods(id) ON DELETE CASCADE,
    filing_date         DATE NOT NULL,                        -- date filed with SEC
    accession_number    VARCHAR(30),                          -- SEC's unique filing ID
    s3_bucket           VARCHAR(100) NOT NULL,
    s3_key              VARCHAR(500) NOT NULL,                -- path inside the bucket
    page_count          INTEGER,
    file_size_bytes     BIGINT,
    sha256_hash         VARCHAR(64),                          -- file integrity
    processed_at        TIMESTAMPTZ,                          -- when our pipeline finished it
    created_at          TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (company_id, period_id),                           -- one filing per company per period
    UNIQUE (s3_bucket, s3_key)
);

COMMENT ON TABLE filings IS
    'Physical SEC filing documents (PDFs in S3). Bridge between company+period and the PDF.';
COMMENT ON COLUMN filings.accession_number IS
    'SEC accession number, e.g., 0000320193-24-000123';
COMMENT ON COLUMN filings.s3_key IS
    'Path in S3 bucket, e.g., filings/AAPL/AAPL_2024_10K.pdf';
COMMENT ON COLUMN filings.processed_at IS
    'NULL = not yet processed by ingestion pipeline';

CREATE INDEX IF NOT EXISTS idx_filings_company ON filings(company_id);
CREATE INDEX IF NOT EXISTS idx_filings_period ON filings(period_id);
CREATE INDEX IF NOT EXISTS idx_filings_date ON filings(filing_date DESC);
CREATE INDEX IF NOT EXISTS idx_filings_unprocessed ON filings(id) WHERE processed_at IS NULL;


-- ============================================================
-- TABLE: financial_metrics
--
-- 🎯 THE PRIMARY NL→SQL QUERY TARGET 🎯
--
-- One row per (company, period) — the core fact table.
-- All monetary values in MILLIONS USD unless noted.
-- All per-share values in actual dollars.
-- 
-- YoY growth columns are PRE-COMPUTED to keep LLM-generated
-- SQL simple (no window functions needed for common queries).
-- ============================================================

CREATE TABLE IF NOT EXISTS financial_metrics (
    id                          SERIAL PRIMARY KEY,
    company_id                  INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    period_id                   INTEGER NOT NULL REFERENCES filing_periods(id) ON DELETE CASCADE,
    filing_id                   INTEGER NOT NULL REFERENCES filings(id) ON DELETE CASCADE,

    -- ── Income Statement (millions USD) ──
    revenue                     NUMERIC(18, 2),               -- top-line
    cost_of_revenue             NUMERIC(18, 2),
    gross_profit                NUMERIC(18, 2),
    gross_margin                NUMERIC(6, 4),                -- 0.4523 = 45.23%
    operating_expenses          NUMERIC(18, 2),
    rd_expense                  NUMERIC(18, 2),               -- research & development
    sga_expense                 NUMERIC(18, 2),               -- selling, general, admin
    operating_income            NUMERIC(18, 2),
    operating_margin            NUMERIC(6, 4),
    net_income                  NUMERIC(18, 2),
    net_margin                  NUMERIC(6, 4),

    -- ── Per Share (dollars) ──
    eps_basic                   NUMERIC(10, 4),
    eps_diluted                 NUMERIC(10, 4),
    shares_outstanding_basic    NUMERIC(18, 2),               -- in millions
    shares_outstanding_diluted  NUMERIC(18, 2),

    -- ── Balance Sheet (millions USD) ──
    total_assets                NUMERIC(18, 2),
    total_liabilities           NUMERIC(18, 2),
    total_equity                NUMERIC(18, 2),
    cash_and_equivalents        NUMERIC(18, 2),
    total_debt                  NUMERIC(18, 2),

    -- ── Cash Flow (millions USD) ──
    operating_cash_flow         NUMERIC(18, 2),
    capex                       NUMERIC(18, 2),               -- capital expenditure
    free_cash_flow              NUMERIC(18, 2),               -- operating_cash_flow - capex

    -- ── PRE-COMPUTED YoY GROWTH (%) ──
    -- These exist so NL→SQL doesn't need window functions for
    -- common queries like "top N by growth". Values are decimal
    -- form: 0.15 = 15% growth, -0.10 = -10% decline.
    revenue_yoy_growth          NUMERIC(8, 4),
    net_income_yoy_growth       NUMERIC(8, 4),
    eps_yoy_growth              NUMERIC(8, 4),
    operating_income_yoy_growth NUMERIC(8, 4),
    free_cash_flow_yoy_growth   NUMERIC(8, 4),

    -- ── PRE-COMPUTED QoQ GROWTH (%) ──
    revenue_qoq_growth          NUMERIC(8, 4),
    eps_qoq_growth              NUMERIC(8, 4),

    -- ── Metadata ──
    currency                    VARCHAR(3) DEFAULT 'USD',
    is_restated                 BOOLEAN DEFAULT FALSE,
    source_page_number          INTEGER,                      -- which page in the PDF
    extracted_at                TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (company_id, period_id)
);

COMMENT ON TABLE financial_metrics IS
    'PRIMARY NL→SQL TARGET. One row per (company, period). All monetary values in millions USD.';

COMMENT ON COLUMN financial_metrics.revenue IS
    'Total revenue / net sales for the period. Millions USD.';
COMMENT ON COLUMN financial_metrics.gross_margin IS
    'Decimal form: 0.4523 = 45.23%. Use *100 in SELECT for display.';
COMMENT ON COLUMN financial_metrics.eps_diluted IS
    'Diluted earnings per share, in actual dollars (not millions).';
COMMENT ON COLUMN financial_metrics.revenue_yoy_growth IS
    'Pre-computed year-over-year revenue growth. Decimal form: 0.15 = 15% growth, -0.10 = -10% decline. Use this for "top by growth" queries.';
COMMENT ON COLUMN financial_metrics.eps_yoy_growth IS
    'Pre-computed YoY EPS growth. Decimal form. Use < 0 to filter "negative EPS growth" queries.';
COMMENT ON COLUMN financial_metrics.free_cash_flow IS
    'Free cash flow = operating_cash_flow - capex. Millions USD.';
COMMENT ON COLUMN financial_metrics.source_page_number IS
    'Which page of the PDF this data was extracted from. For citation.';
COMMENT ON COLUMN financial_metrics.is_restated IS
    'TRUE if these numbers were later corrected. Filter WHERE is_restated = FALSE for clean data.';

-- Indexes for common NL→SQL access patterns
CREATE INDEX IF NOT EXISTS idx_fm_company_period ON financial_metrics(company_id, period_id);
CREATE INDEX IF NOT EXISTS idx_fm_revenue ON financial_metrics(revenue DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_fm_revenue_growth ON financial_metrics(revenue_yoy_growth DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_fm_eps_growth ON financial_metrics(eps_yoy_growth);
CREATE INDEX IF NOT EXISTS idx_fm_filing ON financial_metrics(filing_id);


-- ============================================================
-- TABLE: segment_metrics
--
-- One row per (filing, segment). Critical for queries like
-- "Compare AWS, Azure, GCP" — these are SEGMENTS within
-- AMZN, MSFT, GOOGL respectively.
--
-- IMPORTANT: segment_name must be STANDARDIZED during
-- ingestion. Do not relabel Microsoft "Intelligent Cloud" as "Azure".
-- They refer to different scopes of financial reporting.
-- ============================================================

CREATE TABLE IF NOT EXISTS segment_metrics (
    id                          SERIAL PRIMARY KEY,
    company_id                  INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    period_id                   INTEGER NOT NULL REFERENCES filing_periods(id) ON DELETE CASCADE,
    filing_id                   INTEGER NOT NULL REFERENCES filings(id) ON DELETE CASCADE,

    segment_name                VARCHAR(100) NOT NULL,        -- STANDARDIZED name
    segment_name_raw            VARCHAR(255),                 -- as written in filing
    segment_type                VARCHAR(50),                  -- 'business_unit', 'geographic', 'product'

    -- ── Segment Financials (millions USD) ──
    segment_revenue             NUMERIC(18, 2),
    segment_operating_income    NUMERIC(18, 2),
    segment_operating_margin    NUMERIC(6, 4),

    -- ── Pre-computed Growth ──
    segment_revenue_yoy_growth  NUMERIC(8, 4),
    segment_revenue_qoq_growth  NUMERIC(8, 4),

    -- ── Metadata ──
    source_page_number          INTEGER,
    extracted_at                TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (company_id, period_id, segment_name)
);

COMMENT ON TABLE segment_metrics IS
    'Segment-level financials within a company. Used for "Compare AWS/Azure/GCP" style queries. segment_name is STANDARDIZED.';

COMMENT ON COLUMN segment_metrics.segment_name IS
    'Reported segment name. Examples: AWS, Intelligent Cloud, Google Cloud, iPhone. Use this in WHERE clauses.';
COMMENT ON COLUMN segment_metrics.segment_name_raw IS
    'Original name from the filing, e.g., "Intelligent Cloud" is preserved as a reported business segment.';
COMMENT ON COLUMN segment_metrics.segment_type IS
    'business_unit, geographic, or product';
COMMENT ON COLUMN segment_metrics.segment_revenue_yoy_growth IS
    'Pre-computed YoY growth for this segment. Decimal form.';

CREATE INDEX IF NOT EXISTS idx_sm_company_period ON segment_metrics(company_id, period_id);
CREATE INDEX IF NOT EXISTS idx_sm_segment_name ON segment_metrics(segment_name);
CREATE INDEX IF NOT EXISTS idx_sm_segment_growth ON segment_metrics(segment_revenue_yoy_growth DESC NULLS LAST);


-- ============================================================
-- TABLE: executive_commentary
--
-- 🌉 BRIDGE BETWEEN STRUCTURED AND NARRATIVE 🌉
--
-- Pre-extracted, summarized executive quotes/commentary that
-- can be JOINED to financial_metrics for "filter + narrative"
-- queries like "negative EPS growth + why".
--
-- The chunk_id links back to the vector store for full context.
-- ============================================================

CREATE TABLE IF NOT EXISTS executive_commentary (
    id                  SERIAL PRIMARY KEY,
    company_id          INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    period_id           INTEGER NOT NULL REFERENCES filing_periods(id) ON DELETE CASCADE,
    filing_id           INTEGER NOT NULL REFERENCES filings(id) ON DELETE CASCADE,

    executive_name      VARCHAR(100),                         -- "Tim Cook"
    executive_title     VARCHAR(100),                         -- "CEO"
    topic               commentary_topic_enum NOT NULL,
    summary             TEXT NOT NULL,                        -- 1-3 sentence summary
    full_quote          TEXT,                                 -- exact quote if available
    sentiment           sentiment_enum,
    
    -- ── Provenance ──
    source_page_number  INTEGER,
    chunk_id            VARCHAR(64),                          -- links to vector store chunk
    
    extracted_at        TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE executive_commentary IS
    'Pre-extracted executive quotes and commentary. JOIN to financial_metrics on (company_id, period_id) for combo queries like "negative EPS + why".';

COMMENT ON COLUMN executive_commentary.topic IS
    'Categorical topic. Use in WHERE clauses, e.g., WHERE topic = ''earnings''.';
COMMENT ON COLUMN executive_commentary.summary IS
    '1-3 sentence summary of what was said. Used in result snippets.';
COMMENT ON COLUMN executive_commentary.full_quote IS
    'Exact verbatim quote when available. Use in WHERE ... ILIKE for keyword search.';
COMMENT ON COLUMN executive_commentary.sentiment IS
    'positive, neutral, negative, or mixed. Use to filter "pessimistic outlook" queries.';
COMMENT ON COLUMN executive_commentary.chunk_id IS
    'ID linking to the full chunk in pgvector. Use to retrieve full narrative context.';

CREATE INDEX IF NOT EXISTS idx_ec_company_period ON executive_commentary(company_id, period_id);
CREATE INDEX IF NOT EXISTS idx_ec_topic ON executive_commentary(topic);
CREATE INDEX IF NOT EXISTS idx_ec_sentiment ON executive_commentary(sentiment);
CREATE INDEX IF NOT EXISTS idx_ec_summary_fts ON executive_commentary
    USING gin (to_tsvector('english', summary));


-- ============================================================
-- VIEWS — Convenience views the NL→SQL agent may use
-- ============================================================

-- Wide view: one row per (company, period) with all common joins pre-done.
-- Makes simple lookup queries much easier for the LLM to write.
CREATE OR REPLACE VIEW v_financials_wide AS
SELECT
    c.ticker,
    c.name AS company_name,
    c.sector,
    fp.fiscal_year,
    fp.fiscal_quarter,
    fp.period_label,
    fp.filing_type,
    fp.period_end_date,
    f.filing_date,
    fm.revenue,
    fm.gross_profit,
    fm.gross_margin,
    fm.operating_income,
    fm.operating_margin,
    fm.net_income,
    fm.net_margin,
    fm.eps_diluted,
    fm.free_cash_flow,
    fm.revenue_yoy_growth,
    fm.net_income_yoy_growth,
    fm.eps_yoy_growth,
    fm.operating_income_yoy_growth,
    fm.source_page_number,
    f.s3_key
FROM financial_metrics fm
JOIN companies c       ON c.id  = fm.company_id
JOIN filing_periods fp ON fp.id = fm.period_id
JOIN filings f         ON f.id  = fm.filing_id;

COMMENT ON VIEW v_financials_wide IS
    'Denormalized view of financial metrics. Use this for simple lookups to avoid manual joins.';


-- ============================================================
-- TRIGGERS — Auto-update updated_at timestamps
-- ============================================================

CREATE OR REPLACE FUNCTION fn_update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_companies_updated_at ON companies;
CREATE TRIGGER trg_companies_updated_at
    BEFORE UPDATE ON companies
    FOR EACH ROW EXECUTE FUNCTION fn_update_updated_at();


-- ============================================================
-- SCHEMA VERSION TRACKING
-- ============================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version     VARCHAR(20) PRIMARY KEY,
    applied_at  TIMESTAMPTZ DEFAULT NOW(),
    description TEXT
);

INSERT INTO schema_version (version, description)
VALUES ('1.0.0', 'Initial schema: companies, filings, financial_metrics, segment_metrics, executive_commentary')
ON CONFLICT DO NOTHING;



-- Narrative chunks used by BM25 + pgvector retrieval (768-dimensional BGE model)
CREATE TABLE IF NOT EXISTS document_chunks (
 chunk_id TEXT PRIMARY KEY,
 content TEXT NOT NULL,
 raw_content TEXT NOT NULL,
 embedding vector(768) NOT NULL,
 filing_id TEXT NOT NULL,
 company_ticker TEXT NOT NULL,
 filing_type TEXT NOT NULL,
 fiscal_year SMALLINT NOT NULL,
 fiscal_quarter TEXT NOT NULL,
 section TEXT NOT NULL,
 page_number INTEGER NOT NULL CHECK (page_number >= 1),
 chunk_index INTEGER NOT NULL,
 token_count INTEGER NOT NULL,
 char_count INTEGER NOT NULL,
 overlap_with_prev BOOLEAN DEFAULT FALSE,
 created_at TIMESTAMPTZ DEFAULT NOW(),
 updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chunks_company ON document_chunks(company_ticker);
CREATE INDEX IF NOT EXISTS idx_chunks_filing ON document_chunks(filing_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON document_chunks(section);
CREATE INDEX IF NOT EXISTS idx_chunks_company_period ON document_chunks(company_ticker, fiscal_year, fiscal_quarter);
-- For large collections; cosine distance matches the query operator <=>.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- ============================================================
-- VERIFICATION
-- ============================================================

DO $$
DECLARE
    table_count INT;
    schema_list TEXT;
BEGIN
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = ANY(current_schemas(false))
      AND table_name IN (
          'companies', 'filing_periods', 'filings',
          'financial_metrics', 'segment_metrics', 'executive_commentary'
      );

    SELECT array_to_string(current_schemas(false), ', ') INTO schema_list;

    IF table_count = 6 THEN
        RAISE NOTICE '✅ Schema deployed successfully. 6/6 core tables in: %', schema_list;
    ELSE
        RAISE EXCEPTION '❌ Only % of 6 expected tables exist in: %', table_count, schema_list;
    END IF;
END $$;