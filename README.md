# Autonomous Loan Underwriting Agent

**Built by Sujit** · [GitHub](https://github.com/sujit428755) · MIT License

A multi-agent underwriting workflow built around a credit-risk model. A loan file (application form, salary slip,
bank statement, credit bureau report) goes in. A cited credit memo and a recommendation come out, and the graph
**pauses for a human underwriter** before anything is final.

```
documents -> extract -> [incomplete? -> request_documents -> END]
          -> finance -> fraud checks -> credit risk (ML + SHAP) -> policy RAG checks
          -> recommend + memo -> HUMAN REVIEW (LangGraph interrupt) -> finalize + audit log
```

## Design decisions

| Decision | Why |
|---|---|
| **LLM never does math.** Income, EMI, DTI, PD come from code and the ML model. | Hallucinated numbers are the main risk in finance. The optional Groq commentary is rejected if it contains any number not in the supplied facts. |
| **Verified income = min(salary slip, avg bank salary credit).** | Conservative, and a gap above 10% triggers a fraud referral (P-5.1). |
| **Every policy check cites a clause** (`P-3.1`, ...) retrieved from `data/policy/lending_policy.md`. | Auditable. `/policy/search` exposes the TF-IDF clause retriever. |
| **No credit decision on an incomplete file.** | Returns `REQUEST_DOCUMENTS` (P-6.1) instead of guessing. |
| **Fraud referral outranks decline.** | A declined file built on forged documents is still a fraud case. |
| **Human gate + override rule.** | Approving a declined file (or rejecting an approved one) requires a written reason (P-7.2). |
| **Protected attributes are never model features.** | Asserted in the graph, with parity monitoring in `evaluate.py`. |
| **Append-only audit log** (`logs/audit.jsonl`). | Every agent step is stored, so a decision can be reconstructed. |

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=src                      # Windows: set PYTHONPATH=src
python scripts/train_model.py              # trains PD model, logs to MLflow if installed
python scripts/generate_samples.py         # 7 sample files in data/samples/
python scripts/run_demo.py data/samples/clean --action approve --reviewer "S. Rao"
streamlit run app/streamlit_app.py         # underwriter workbench (6 tabs, themed via .streamlit/config.toml)
uvicorn app.api:app --reload               # REST API (docs at /docs)
python scripts/evaluate.py                 # evaluation + fairness report
pytest -q
docker compose up --build                  # API :8000, UI :8501
```

Optional: copy `.env.example` to `.env` and set `GROQ_API_KEY` for LLM-written commentary.

## UI (Streamlit)

`app/streamlit_app.py` is a six-tab workbench: **Underwrite** (upload or pick a sample, KPI cards, gauges, SHAP chart,
cited policy checks, memo, agent trace, human decision gate), **Batch** (run many synthetic applicants and see a risk map),
**Evaluation** (accuracy gauges, confusion matrix, fairness parity), **Review Queue** (recorded human decisions and overrides),
**Audit Log** and **Architecture & Policy** (clause search over the lending policy). Chart helpers live in `app/visuals.py`.

## API

- `POST /applications` with `{"documents": {"application_form": "...", "salary_slip": "...", "bank_statement": "...", "credit_report": "..."}}` returns the memo and a `thread_id`.
- `POST /applications/{thread_id}/decision` with `{"action": "approve|reject|request_more_info", "reviewer": "...", "note": "..."}`.
- `GET /applications/{thread_id}/trace` returns the agent trace.
- `GET /policy/search?q=...` retrieves policy clauses.

## Evaluation (synthetic, 42 cases + 500-applicant parity check)

Run `python scripts/evaluate.py`. Latest result is in `reports/eval_report.json`:
decision accuracy 1.00, extraction field accuracy 1.00, missing-document refusal 1.00, disparate impact ratio 0.84 (passes four-fifths).

**Read these honestly.** The documents and the parsers come from the same synthetic generator, so near-perfect
extraction is expected. It proves the pipeline logic, not real-world robustness. The parity check shows the
monitoring works, not that a production model is fair. Real data would need separate fairness testing.

## Limitations and next steps

1. Replace the synthetic PD model with your real credit-risk model from the MLOps project (`credit_model.py`: `train`/`predict`).
2. Real documents: add OCR (Azure Document Intelligence or pytesseract) before `extraction.py`, plus an LLM fallback for fields the regex parsers miss, validated against the same rules.
3. Swap TF-IDF for embeddings in Chroma/FAISS (`PolicyIndex` is the only class to change).
4. Replace `MemorySaver` with a Postgres/SQLite checkpointer so pending reviews survive restarts.
5. Add authentication and role checks to the decision endpoint.

All data in this repo is synthetic. This is a portfolio project, not a lending system.

---
Built by [Sujit](https://github.com/sujit428755). Licensed under MIT.
