"""Evaluation harness: extraction accuracy, decision accuracy, missing-document refusal, fairness monitoring.
Run: python -m scripts.evaluate   (or python scripts/evaluate.py)"""
import json
import random
from collections import Counter, defaultdict

from . import graph, synth
from .config import ROOT


def _truth(a):
    return {"slip_net_salary": round(a["monthly_income"] * a["slip_inflation"]), "credit_score": a["credit_score"],
            "loan_amount": a["loan_amount"], "tenure_months": a["tenure_months"], "age": a["age"],
            "existing_emi": a["existing_emi"], "avg_salary_credit": a["monthly_income"],
            "bounce_count": a["bounce_count"]}


def run_suite(per_scenario=6, seed=100):
    rows, field_hits, field_total = [], 0, 0
    miss_ok, miss_n = 0, 0
    per = defaultdict(lambda: [0, 0])
    for si, sc in enumerate(synth.SCENARIOS):
        for k in range(per_scenario):
            a, docs = synth.make_case(seed + si * 100 + k, sc, si * 100 + k)
            s = graph.analyze(docs, a["application_id"])
            got, exp = s["recommendation"], synth.EXPECTED[sc]
            per[sc][0] += got == exp
            per[sc][1] += 1
            rows.append({"id": a["application_id"], "scenario": sc, "expected": exp, "got": got})
            if sc == "missing_doc":
                miss_n += 1
                miss_ok += got == "REQUEST_DOCUMENTS" and "credit" not in s
            else:
                f = s["extracted"]["fields"]
                for key, val in _truth(a).items():
                    field_total += 1
                    field_hits += abs((f.get(key) or -1) - val) < 0.5 if key != "bounce_count" else f.get(key) == val
    n = len(rows)
    return {
        "cases": n,
        "decision_accuracy": round(sum(r["expected"] == r["got"] for r in rows) / n, 3),
        "decision_accuracy_by_scenario": {k: f"{v[0]}/{v[1]}" for k, v in per.items()},
        "extraction_field_accuracy": round(field_hits / field_total, 3),
        "missing_doc_refusal_rate": round(miss_ok / miss_n, 3) if miss_n else None,
        "failures": [r for r in rows if r["expected"] != r["got"]],
        "rows": rows,
    }


def fairness(n=500, seed=2026):
    """Recommendation-rate parity across a protected attribute (never a model input)."""
    rng = random.Random(seed)
    weights = [0.55, 0.1, 0.1, 0.05, 0.0, 0.1, 0.1]  # scenario mix, independent of gender by construction
    counts = defaultdict(Counter)
    for i in range(n):
        sc = rng.choices(synth.SCENARIOS, weights)[0]
        a, docs = synth.make_case(seed + i, sc, i)
        s = graph.analyze(docs, a["application_id"])
        counts[a["gender"]][s["recommendation"]] += 1
    rates = {g: c["APPROVE_RECOMMENDED"] / sum(c.values()) for g, c in counts.items()}
    hi, lo = max(rates.values()), min(rates.values())
    return {"n": n, "approve_rate_by_gender": {g: round(r, 3) for g, r in rates.items()},
            "demographic_parity_difference": round(hi - lo, 3),
            "disparate_impact_ratio": round(lo / hi, 3) if hi else None,
            "four_fifths_rule_pass": bool(hi and lo / hi >= 0.8)}


def main():
    report = {"suite": run_suite(), "fairness": fairness()}
    out = ROOT / "reports" / "eval_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report
