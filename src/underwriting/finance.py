"""Deterministic financial calculations. The LLM never does arithmetic."""
from statistics import mean


def emi(principal: float, annual_rate_pct: float, months: int) -> float:
    r = annual_rate_pct / 1200.0
    if r == 0:
        return principal / months
    f = (1 + r) ** months
    return principal * r * f / (f - 1)


def compute_metrics(x: dict) -> dict:
    """x = extracted fields. Verified income = min(salary slip, avg bank salary credit)."""
    slip = x["slip_net_salary"]
    bank = x["avg_salary_credit"]
    verified = min(slip, bank)
    new_emi = emi(x["loan_amount"], x["interest_rate"], x["tenure_months"])
    existing = x["existing_emi"]
    return {
        "verified_income": round(verified, 2),
        "new_emi": round(new_emi, 2),
        "existing_emi": existing,
        "dti_current": round(existing / verified, 4),
        "dti_post": round((existing + new_emi) / verified, 4),
        "loan_to_monthly_income": round(x["loan_amount"] / verified, 2),
        "avg_balance": round(x["avg_balance"], 2),
        "avg_monthly_spend": round(x["avg_monthly_spend"], 2),
        "monthly_surplus_after_emi": round(verified - existing - new_emi - x["avg_monthly_spend"], 2),
        "bounce_count": x["bounce_count"],
        "income_gap_pct": round((slip - bank) / bank, 4) if bank else None,
    }


def avg(values):
    return mean(values) if values else 0.0
