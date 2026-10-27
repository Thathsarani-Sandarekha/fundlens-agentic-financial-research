-- Executed on an empty PostgreSQL data directory.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS fundlens;
-- docker-entrypoint-initdb.d/20_schema.sql creates FundLens objects.
