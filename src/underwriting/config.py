"""Central thresholds. These mirror data/policy/lending_policy.md - change both together."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "data" / "policy" / "lending_policy.md"
MODEL_PATH = ROOT / "models" / "credit_model.joblib"
AUDIT_LOG = ROOT / "logs" / "audit.jsonl"

REQUIRED_DOCS = ["application_form", "salary_slip", "bank_statement", "credit_report"]

AGE_MIN, AGE_MAX, AGE_PLUS_TENURE_MAX = 21, 60, 65
TENURE_YEARS_REVIEW, TENURE_YEARS_DECLINE = 1.0, 0.5
MIN_INCOME = 25_000
LTI_REFER, LTI_DECLINE = 12, 15
DTI_REFER, DTI_DECLINE = 0.40, 0.50
SCORE_PASS, SCORE_REFER = 700, 650
BOUNCE_REFER, BOUNCE_DECLINE = 2, 3
INCOME_MISMATCH_TOL = 0.10
PD_REFER, PD_DECLINE = 0.10, 0.25

# Protected attributes are NEVER model features (policy P-7.3).
MODEL_FEATURES = [
    "credit_score", "dti_post", "loan_to_monthly_income",
    "years_employed", "age", "bounce_count", "tenure_months",
]
PROTECTED_ATTRIBUTES = ["gender"]
