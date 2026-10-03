"""Synthetic loan applications: text documents + ground truth for evaluation."""
import random
from .finance import emi

FIRST = ["Aarav", "Priya", "Rohan", "Sneha", "Vikram", "Ananya", "Karan", "Meera", "Arjun", "Isha",
         "Rahul", "Neha", "Amit", "Pooja", "Siddharth", "Kavya", "Nikhil", "Divya", "Manish", "Riya"]
LAST = ["Sharma", "Patil", "Deshmukh", "Iyer", "Kulkarni", "Reddy", "Joshi", "Mehta", "Nair", "Gupta"]
EMPLOYERS = ["Infosys", "TCS", "Persistent Systems", "Bajaj Auto", "Tata Motors", "Wipro", "KPIT", "Cognizant"]
PURPOSES = ["Home renovation", "Medical expenses", "Education", "Wedding", "Travel", "Debt consolidation"]
SCENARIOS = ["clean", "high_dti", "low_score", "fraud_mismatch", "missing_doc", "bounces", "short_tenure_job"]
EXPECTED = {  # scenario -> expected recommendation
    "clean": "APPROVE_RECOMMENDED",
    "high_dti": "DECLINE_RECOMMENDED",
    "low_score": "DECLINE_RECOMMENDED",
    "fraud_mismatch": "REFER_FRAUD_REVIEW",
    "missing_doc": "REQUEST_DOCUMENTS",
    "bounces": "DECLINE_RECOMMENDED",
    "short_tenure_job": "DECLINE_RECOMMENDED",
}
MONTHS = [(2026, m) for m in range(4, 10)]  # Apr-Sep 2026


def make_applicant(rng: random.Random, scenario: str = "clean", idx: int = 0) -> dict:
    age = rng.randint(25, 50)
    income = round(rng.uniform(40_000, 200_000), -2)
    a = {
        "application_id": f"APP-{idx:04d}",
        "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        "age": age,
        "gender": rng.choice(["F", "M"]),
        "employer": rng.choice(EMPLOYERS),
        "years_employed": round(rng.uniform(3, 12), 1),
        "monthly_income": income,
        "existing_emi": round(income * rng.uniform(0.0, 0.12), -2),
        "credit_score": rng.randint(740, 850),
        "bounce_count": 0,
        "loan_amount": round(income * rng.uniform(2, 6), -3),
        "tenure_months": rng.choice([36, 48, 60]),
        "interest_rate": rng.choice([11.5, 12.0, 12.5, 13.0]),
        "purpose": rng.choice(PURPOSES),
        "slip_inflation": 1.0,
        "scenario": scenario,
    }
    if scenario == "high_dti":
        a["existing_emi"] = round(income * 0.47, -2)
    elif scenario == "low_score":
        a["credit_score"] = rng.randint(480, 620)
    elif scenario == "fraud_mismatch":
        a["slip_inflation"] = rng.uniform(1.25, 1.5)
    elif scenario == "bounces":
        a["bounce_count"] = rng.randint(3, 5)
    elif scenario == "short_tenure_job":
        a["years_employed"] = round(rng.uniform(0.1, 0.4), 1)
    return a


def render_documents(a: dict, rng: random.Random) -> dict:
    inc = a["monthly_income"]
    slip_net = round(inc * a["slip_inflation"])
    gross = round(slip_net * 1.18)
    docs = {}
    docs["application_form"] = (
        "LOAN APPLICATION FORM\n"
        f"Application ID: {a['application_id']}\nApplicant Name: {a['name']}\nAge: {a['age']}\n"
        f"Gender: {a['gender']}\nEmployment Type: Salaried\nEmployer: {a['employer']}\n"
        f"Years at Current Employer: {a['years_employed']}\n"
        f"Monthly Net Income (declared): {int(slip_net)}\n"
        f"Loan Purpose: {a['purpose']}\nLoan Amount Requested: {int(a['loan_amount'])}\n"
        f"Tenure (months): {a['tenure_months']}\nInterest Rate (% p.a.): {a['interest_rate']}\n"
    )
    docs["salary_slip"] = (
        "SALARY SLIP - September 2026\n"
        f"Employer: {a['employer']}\nEmployee: {a['name']}\n"
        f"Gross Salary: {gross}\nTotal Deductions: {gross - slip_net}\nNet Salary: {slip_net}\n"
    )
    docs["credit_report"] = (
        "CREDIT BUREAU REPORT\n"
        f"Applicant: {a['name']}\nCredit Score: {a['credit_score']}\n"
        f"Total Monthly Obligations: {int(a['existing_emi'])}\nDelinquencies (24m): {a['bounce_count']}\n"
    )
    # bank statement: 6 months
    lines = ["BANK STATEMENT", f"Account Holder: {a['name']}", "Period: 2026-04-01 to 2026-09-30",
             "DATE | DESCRIPTION | DEBIT | CREDIT | BALANCE"]
    bal = round(inc * rng.uniform(0.8, 1.5))
    bounce_months = set(rng.sample(range(6), min(a["bounce_count"], 6)))
    for i, (y, m) in enumerate(MONTHS):
        bal += inc
        lines.append(f"{y}-{m:02d}-01 | SALARY CREDIT - {a['employer'].upper()} | | {int(inc)} | {int(bal)}")
        if a["existing_emi"] > 0:
            bal -= a["existing_emi"]
            lines.append(f"{y}-{m:02d}-05 | EMI AUTO DEBIT | {int(a['existing_emi'])} | | {int(bal)}")
        if i in bounce_months:
            bal -= 590
            lines.append(f"{y}-{m:02d}-06 | CHEQUE BOUNCE CHARGE | 590 | | {int(bal)}")
        budget = max(inc - a["existing_emi"], 0) * rng.uniform(0.35, 0.55)
        for d, label in zip((9, 14, 20, 26), ("UPI GROCERY", "RENT TRANSFER", "UTILITIES", "UPI DINING")):
            amt = round(budget * rng.uniform(0.15, 0.35))
            bal -= amt
            lines.append(f"{y}-{m:02d}-{d:02d} | {label} | {amt} | | {int(bal)}")
    docs["bank_statement"] = "\n".join(lines) + "\n"
    if a["scenario"] == "missing_doc":
        docs.pop(rng.choice(["salary_slip", "bank_statement"]))
    return docs


def make_case(seed: int, scenario: str, idx: int = 0):
    rng = random.Random(seed)
    a = make_applicant(rng, scenario, idx)
    return a, render_documents(a, rng)
