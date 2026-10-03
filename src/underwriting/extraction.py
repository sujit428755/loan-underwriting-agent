"""Document -> structured fields. Deterministic parsers (txt/PDF text); no LLM needed for the numbers."""
import re
from pathlib import Path
from statistics import mean

from .config import REQUIRED_DOCS


def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # scanned PDFs would need OCR (e.g. pytesseract) first
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


def load_folder(folder) -> dict:
    return {p.stem: load_text(p) for p in sorted(Path(folder).iterdir())
            if p.suffix.lower() in {".txt", ".pdf"}}


def _kv(text: str, key: str):
    m = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", text or "", re.M | re.I)
    return m.group(1).strip() if m else None


def _num(v):
    if v is None or not v.strip():
        return None
    try:
        return float(v.replace(",", "").strip())
    except ValueError:
        return None


def _parse_bank(text: str) -> dict:
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 5 and re.match(r"\d{4}-\d{2}-\d{2}", parts[0]):
            rows.append({"date": parts[0], "desc": parts[1].upper(),
                         "debit": _num(parts[2]) or 0.0, "credit": _num(parts[3]) or 0.0,
                         "balance": _num(parts[4])})
    if not rows:
        return {}
    months = sorted({r["date"][:7] for r in rows})
    salary = [r["credit"] for r in rows if r["desc"].startswith("SALARY")]
    spend = sum(r["debit"] for r in rows if "EMI" not in r["desc"] and "BOUNCE" not in r["desc"])
    return {
        "bank_name": _kv(text, "Account Holder"),
        "avg_salary_credit": mean(salary) if salary else None,
        "avg_balance": mean(r["balance"] for r in rows),
        "avg_monthly_spend": spend / len(months),
        "bounce_count": sum("BOUNCE" in r["desc"] for r in rows),
        "months_covered": len(months),
    }


def extract(documents: dict) -> dict:
    missing_docs = [d for d in REQUIRED_DOCS if not (documents.get(d) or "").strip()]
    f = {}
    form, slip, cr = (documents.get(k, "") for k in ("application_form", "salary_slip", "credit_report"))
    if form:
        f.update(application_id=_kv(form, "Application ID"), name=_kv(form, "Applicant Name"),
                 age=_num(_kv(form, "Age")), gender=_kv(form, "Gender"), employer=_kv(form, "Employer"),
                 years_employed=_num(_kv(form, "Years at Current Employer")),
                 declared_income=_num(_kv(form, "Monthly Net Income (declared)")),
                 purpose=_kv(form, "Loan Purpose"), loan_amount=_num(_kv(form, "Loan Amount Requested")),
                 tenure_months=int(_num(_kv(form, "Tenure (months)")) or 0) or None,
                 interest_rate=_num(_kv(form, "Interest Rate (% p.a.)")))
    if slip:
        f.update(slip_name=_kv(slip, "Employee"), slip_employer=_kv(slip, "Employer"),
                 slip_net_salary=_num(_kv(slip, "Net Salary")))
    if cr:
        f.update(credit_name=_kv(cr, "Applicant"), credit_score=_num(_kv(cr, "Credit Score")),
                 existing_emi=_num(_kv(cr, "Total Monthly Obligations")))
    if documents.get("bank_statement"):
        f.update(_parse_bank(documents["bank_statement"]))
    needed = ["name", "age", "loan_amount", "tenure_months", "interest_rate", "years_employed",
              "slip_net_salary", "credit_score", "existing_emi", "avg_salary_credit", "avg_balance",
              "avg_monthly_spend", "bounce_count"]
    missing_fields = [k for k in needed if f.get(k) is None] if not missing_docs else []
    return {"fields": f, "missing_docs": missing_docs, "missing_fields": missing_fields}
