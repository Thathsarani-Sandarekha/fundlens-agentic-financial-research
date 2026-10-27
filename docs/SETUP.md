# Local setup (Windows / Conda)

These instructions match the **inspected public engineering-project layout**, not a fully verified clean install of the latest local working copy. Review `.env.example`, `pyproject.toml`, `docker-compose.yml`, and the ingestion script in your actual repository before publishing.

## 1. Prerequisites

- Windows Terminal / CMD and Conda with Python 3.10+ (the evaluated working environment uses `docmind_env`).
- Docker Desktop with Docker Compose.
- A Groq API key, internet for first model downloads, and legal access to public SEC filings.

## 2. Python environment and dependencies

```cmd
cd /d F:\Github\Fundlens
conda activate docmind_env
python -m pip install -e ".[ingestion,dev]"
set "PYTHONPATH=%CD%\src"
```

If you are starting fresh rather than reusing an existing environment:

```cmd
conda create -n fundlens python=3.11 -y
conda activate fundlens
python -m pip install -e ".[ingestion,dev]"
```

The editable install command assumes `pyproject.toml` exists with `ingestion` and `dev` extras. If your working repo has a different dependency manifest, follow its own dependency definitions.

## 3. Environment variables

```cmd
copy .env.example .env
```

Supply your own passwords and API key. The inspected example expects:

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq generation and synthesis |
| `POSTGRES_URL` | SQLAlchemy database URL |
| `POSTGRES_PASSWORD` | Docker PostgreSQL password |
| `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET` | MinIO connectivity |
| `NEO4J_URI`, `NEO4J_PASSWORD` | Optional graph connectivity |

Default Compose ports in the inspected engineering preview: **PostgreSQL 5433**, **Redis 6380**, **Neo4j Bolt 7688 / UI 7475**, **MinIO API 9000 / Console 9001** (bound to localhost). Your local configuration may differ.

**Never commit `.env` or `.streamlit/secrets.toml`.** Do not post screenshots of secrets when troubleshooting.

## 4. Start services

```cmd
docker compose up -d
docker compose ps
```

Docker initialisation scripts only run when an empty database volume is created. **Do not delete, prune, recreate, or reset your existing Postgres volume to follow this guide.** A new checkout won't contain the benchmark dataset unless filings are ingested.

## 5. Ingest your own public filings (optional for already-ingested local systems)

For the inspected ingestion script, the file convention is:

```text
data/raw/sec_filings/GOOGL/GOOGL_2024_10K.pdf
data/raw/sec_filings/GOOGL/GOOGL_2025_Q1_10Q.pdf
```

The script requires both the **period end** and the **actual SEC filing date** (verify the filing date from the source's official metadata):

```cmd
python scripts\run_ingestion_pipeline.py GOOGL 2024 FY 10-K --period-end 2024-12-31 --filing-date YYYY-MM-DD
python scripts\run_ingestion_pipeline.py GOOGL 2025 Q1 10-Q --period-end 2025-03-31 --filing-date YYYY-MM-DD
```

Replace `YYYY-MM-DD` with the verified SEC filing date before running. These commands are examples, not a request to re-ingest your working local database. Ingestion may require Docling and OCR/table-recognition dependencies beyond the core Python package.

## 6. Launch the UI

```cmd
set "PYTHONPATH=%CD%\src"
python -m streamlit run app.py --server.port 8502
```

Open http://localhost:8502 and try a sample SQL, RAG, or Hybrid query from [DEMO.md](DEMO.md).

## 7. Tests and evaluations

```cmd
python -m pytest -q tests
```

If `scripts/evaluate_metrics.py` is present in your local project and your existing data are available:

```cmd
python scripts\evaluate_metrics.py run --suite core
```

This is an integration benchmark using live dependencies; it can be affected by model/API availability. It should not replace the successful historical repeatability results without a like-for-like rerun.

## Troubleshooting

- **Model hub warning:** Downloading SentenceTransformers weights without an HF token can show a warning; that warning alone does not prove failure.
- **Groq SQL retries:** Inspect terminal/provider error logs. A failed SQL request may fall back to RAG, which doesn't count as an SQL pass.
- **Database connection failure:** Check `docker compose ps`, port mappings, `POSTGRES_URL`, and passwords. Avoid resetting persistent volumes.
- **Incorrect filing evidence:** Check ticker, 10-K vs 10-Q, FY vs quarter and `Risk Factors` metadata.
- **Slow Hybrid answers:** Hybrid often combines SQL generation, retrieval, and synthesis and was the slowest evaluated path.
