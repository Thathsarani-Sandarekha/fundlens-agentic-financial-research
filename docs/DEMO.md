# FundLens demo walkthrough

This is a repeatable **manual demo script**, not a substitute for running integration tests. Start the Streamlit application with an ingested local database.

| Step | Ask FundLens | Expected path | Check |
|---|---|---|---|
| 1 | What was Alphabet's FY2024 revenue? | SQL | SQL rows contain `350018` USD million; answer may round to **$350.0 billion**. |
| 2 | What AI-related risks did Alphabet discuss in its 2024 annual report? | RAG | Uses 2024 10-K Risk Factors passages; discusses AI liability, competition, investment, privacy or regulation; source indices visible. |
| 3 | What was Alphabet's FY2024 revenue, and what AI-related risks did its annual report discuss? | Hybrid | Contains both the SQL metric and appropriately sourced AI-risk explanation. |

**Additional reference:** Alphabet FY2024 net income was `100118` USD million (~$100.1B), and Q1 2025 consolidated revenue was `90234` USD million (~$90.2B), as checked against benchmark reference rows.

## Presenting the demo

Show the question, route taken, answer, citation cards and relevant evidence. Do not claim that page references are independently verified against the original PDF. If you capture screenshots, ensure API keys, paths and terminal logs are not visible.

## Screenshots and videos

Place your **own, reviewed** demo screenshots in an appropriate `docs/images/` folder and link them from the README. No screenshots have been fabricated in this publication pack.
