"""Rule-based document consistency / fraud checks (policy P-5.1, P-5.2)."""
from .config import INCOME_MISMATCH_TOL


def _norm(s):
    return (s or "").strip().lower()


def check_documents(x: dict) -> dict:
    flags = []
    gap = (x["slip_net_salary"] - x["avg_salary_credit"]) / x["avg_salary_credit"]
    if abs(gap) > INCOME_MISMATCH_TOL:
        flags.append({"clause": "P-5.1", "type": "income_mismatch",
                      "detail": f"Salary slip net pay differs from average bank salary credit by {gap:+.1%}"})
    decl = (x["declared_income"] - x["slip_net_salary"]) / x["slip_net_salary"] if x.get("declared_income") else 0
    if abs(decl) > INCOME_MISMATCH_TOL:
        flags.append({"clause": "P-5.1", "type": "declared_vs_slip",
                      "detail": f"Declared income differs from salary slip by {decl:+.1%}"})
    names = {_norm(x.get(k)) for k in ("name", "slip_name", "credit_name", "bank_name") if x.get(k)}
    if len(names) > 1:
        flags.append({"clause": "P-5.2", "type": "name_mismatch", "detail": f"Names differ across documents: {sorted(names)}"})
    if x.get("slip_employer") and _norm(x["slip_employer"]) != _norm(x.get("employer")):
        flags.append({"clause": "P-5.2", "type": "employer_mismatch",
                      "detail": f"Employer on form '{x.get('employer')}' vs salary slip '{x['slip_employer']}'"})
    return {"flagged": bool(flags), "flags": flags}
