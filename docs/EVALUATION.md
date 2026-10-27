# FundLens — Evaluation Results

**Evaluation date:** 27 October 2026  
**System:** LangGraph-powered financial research assistant with SQL, RAG, and Hybrid query paths  
**Dataset scope:** Curated questions about Alphabet (GOOGL), using FY2024 10-K and Q1 2025 10-Q data  
**Evaluation approach:** Deterministic functional tests over three repeated runs, plus a separate custom Groq LLM-as-a-judge evaluation on the final run.

## Executive summary

**All 30 of 30 automatic benchmark executions passed** across 10 questions repeated three times. Routes were consistent, all 21 checked numeric SQL results matched reference values, and no final answer was empty. On the five RAG/Hybrid questions from the third run, an LLM judge assessed claim support as **47/47** and numbered citation support as **11/11**, but strict retrieval **Precision@5 was 44% (11 relevant chunks out of 25)**. These findings reflect a successful small functional benchmark and a promising answer-grounding assessment; they do **not** establish general real-world accuracy or verified original-PDF citation page accuracy.

## 1. Automated functional and repeatability evaluation

| Metric | Result | Denominator / definition |
|---|---:|---|
| Automatic benchmark pass rate | **100%** | 30 / 30 executions passed defined checks |
| Intent routing accuracy | **100%** | 30 / 30 matched expected SQL/RAG/Hybrid route |
| Routing consistency | **100%** | 10 / 10 questions assigned the same route in all three runs |
| Exact SQL numeric accuracy | **100%** | 21 / 21 checked SQL/Hybrid numerical results matched benchmark reference values |
| Nonempty final response rate | **100%** | 30 / 30 final answers nonempty |
| Valid numbered citation indices | **100%** | 15 / 15 RAG or Hybrid cases had syntactically valid source-card indices |
| Retrieval metadata mismatch | **0** | No cross-filing/section mismatches recorded by this check |

### Latency (30 executions)

| Measure | Seconds |
|---|---:|
| Mean | 21.79 |
| Median (p50) | 20.661 |
| p95 | 47.117 |
| Minimum | 2.174 |
| Maximum | 59.265 |

| Route | Executions | Mean latency |
|---|---:|---:|
| SQL | 15 | 18.571 s |
| RAG | 9 | 16.279 s |
| Hybrid | 6 | 38.104 s |

Hybrid is the slowest path on average in this small benchmark. Latency varied between repeat runs and may reflect LLM/API response-time variability.

## 2. LLM-as-a-judge source-grounding evaluation

**Judge model:** `openai/gpt-oss-20b` via Groq.  
**Evaluated subset:** 5 RAG/Hybrid questions from **Run 3**, with 5 retrieved chunks per question (25 chunks).  
**Judge basis:** The generated answers, stored SQL results, and indexed retrieval passages; not an independent human or original-PDF audit.

| Metric | LLM-judged result | Interpretation |
|---|---:|---|
| Claim support / faithfulness | **100% (47/47)** | Judge found every extracted answer claim supported by supplied SQL rows or retrieved text |
| Retrieval Precision@5 | **44% (11/25)** | Only chunks graded **2 or 3** on a 0–3 relevance scale counted as relevant |
| Retrieval MRR@5 | **0.800** | First strongly relevant result usually appeared near the top |
| Retrieval nDCG@5 | **0.7979** | Graded ranking quality across the five retrieval cases |
| Numbered citation support | **100% (11/11)** | Judge considered each assessed numbered citation supported by its referenced passage |
| Answer relevance | **5.0 / 5** | Average judge score |
| Answer completeness | **4.8 / 5** | Average judge score |
| Exact original-PDF page citation accuracy | **Not verified** | Requires matching evidence against the published PDFs |

### Interpreting the 44% Precision@5 result

This is the most important improvement area. Although the judge found that final answers could be grounded in retrieved evidence, **14 of the 25 top-five passages were not judged strongly relevant** under the selected threshold. Useful follow-up work would include relevance-aware reranking, deduplicating adjacent chunks, adjusting hybrid retrieval weights, or adding a minimum reranker relevance threshold. These are future optimization opportunities, not a reason to discard the successful functional benchmark.

### Quality-assessment caveats

- Claim support is **LLM-judged**, not a human-verified 100% factual accuracy score.
- Citation support (11/11) was judged against **indexed passages**, not independent PDF pages.
- Valid citation indices (15/15) check that referenced source-card numbers exist; they are **not** equivalent to citation precision.
- Metadata matching is **not** equivalent to semantic retrieval relevance.
- The qualitative evaluation covers **5 questions**, not all 15 RAG/Hybrid executions.
- Prior runs occasionally produced truncated *intermediate* Hybrid RAG text, although the final answers in this repeatability benchmark were nonempty.
- The 10 benchmark questions are curated and focused on a single company; generalization to unseen companies, filings, and adversarial inputs remains untested.
- Original-PDF page accuracy, human-reviewed citation accuracy, token counts, and inference costs are not measured.

## 3. Representative financial facts and demo queries

The benchmark checked these reference values (source figures in USD millions):

| Example question | Route | Reference value / evidence |
|---|---|---|
| What was Alphabet's FY2024 revenue? | SQL | USD **350,018 million** (about $350.0B) |
| What was Alphabet's FY2024 net income? | SQL | USD **100,118 million** (about $100.1B) |
| What was Alphabet's Q1 2025 consolidated revenue? | SQL | USD **90,234 million** (about $90.2B) |
| What AI-related risks did Alphabet discuss in its 2024 annual report? | RAG | FY2024 10-K narrative on AI competition, investments, regulatory uncertainty and potential harms |
| What was Alphabet's FY2024 net income, and what AI-related risks did its annual report discuss? | Hybrid | SQL metric + retrieved FY2024 risk narrative |

**Published references:**
- Alphabet FY2024 10-K: https://s206.q4cdn.com/479360582/files/doc_financials/2024/q4/goog-10-k-2024.pdf
- Alphabet Q1 2025 10-Q: https://s206.q4cdn.com/479360582/files/doc_financials/2025/q1/goog-10-q-q1-2025.pdf

The PDF references provide source context; the automated citation audit did **not** verify exact PDF page mapping.

## 4. Reproducibility and reporting

For a precise evaluation claim, use:

> On a curated 10-question Alphabet benchmark repeated three times, FundLens passed 30/30 automated functional checks, including 30/30 routing decisions and 21/21 exact SQL numeric checks. In a separate five-question, LLM-judged RAG/Hybrid assessment, claim support was 47/47 and retrieval Precision@5 was 44%. The latter scores are model-generated assessments, not human-verified factual accuracy; original-PDF citation page accuracy remains unverified.

**Artifacts to retain in the repository:** raw outputs and summaries for Runs 1–3; `quality_summary.json`; `quality_report.md`; this evaluation overview. Remove or redact any keys, tokens, personal paths, or other secrets before publishing.

## 5. Retrieval experiment and post-rollback verification

A read-only A/B comparison tested a cleaner Hybrid narrative query against the original retrieval. A fair LLM rejudgment of unique passages gave original **72% (18/25)** versus focused-query **76% (19/25)** Precision@5. These are a **different judging run**, not directly comparable to the earlier 44%. Per-question results were mixed, and the offline result did not establish a safe deployed improvement.

A subsequent experimental Hybrid-only code patch produced **5/10** automatic passes in an expanded regression. At least one FY2024 Hybrid question retrieved a Q1 2025 10-Q passage; SQL/RAG failures also occurred, but their precise cause was not isolated to the patch. The experiment was rejected and rolled back. The restored system then passed **5/5** core checks (100% routing and numerical accuracy; RAG and Hybrid successful, with FY2024 10-K metadata used). Retrieval precision was **not** re-scored after the rollback. The stable-version 30/30 benchmark remains the main published reference.

Full record: [Retrieval experiment](RETRIEVAL_EXPERIMENT.md).

## 6. Status and recommended follow-up

**Evaluation milestone: completed**, within the stated curated-benchmark and automated-judge scope. The existing metrics can be reported honestly now. Highest-value future work is improving the **44% retrieval Precision@5** and validating citations against original filings on a small independent sample. No new ingestion, database repair, or evaluation rerun is required to publish the current results with these caveats.
