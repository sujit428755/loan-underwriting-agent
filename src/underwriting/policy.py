"""Author: Sujit (github.com/sujit428755)
Policy RAG: clause-level index over the lending policy + rule checks that cite clauses.
Retrieval is TF-IDF (offline, no API). Swap `PolicyIndex` for Chroma/FAISS + embeddings if you prefer."""
import re
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config as C


class PolicyIndex:
    def __init__(self, path=C.POLICY_PATH):
        text = path.read_text(encoding="utf-8")
        self.clauses = {}
        for m in re.finditer(r"###\s+(P-\d+\.\d+)\s+(.+?)\n(.*?)(?=\n###|\n## |\Z)", text, re.S):
            cid, title, body = m.group(1), m.group(2).strip(), " ".join(m.group(3).split())
            self.clauses[cid] = {"id": cid, "title": title, "text": body}
        self.ids = list(self.clauses)
        docs = [f"{c['title']}. {c['text']}" for c in self.clauses.values()]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(docs)
        self.mat = self.vec.transform(docs)

    def retrieve(self, query: str, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.mat)[0]
        top = sims.argsort()[::-1][:k]
        return [{**self.clauses[self.ids[i]], "score": round(float(sims[i]), 3)} for i in top if sims[i] > 0]

    def get(self, cid):
        return self.clauses[cid]


@lru_cache(maxsize=1)
def index() -> PolicyIndex:
    return PolicyIndex()


def _row(check, status, value, threshold, cid):
    c = index().get(cid)
    return {"check": check, "status": status, "value": value, "threshold": threshold,
            "clause": cid, "clause_title": c["title"], "evidence": c["text"]}


def _tier(v, refer, decline, higher_is_worse=True):
    if higher_is_worse:
        return "FAIL" if v > decline else "WARN" if v > refer else "PASS"
    return "FAIL" if v < decline else "WARN" if v < refer else "PASS"


def run_checks(x: dict, m: dict, pd_: float, fraud: dict) -> list:
    rows = []
    age_end = x["age"] + x["tenure_months"] / 12
    ok = C.AGE_MIN <= x["age"] <= C.AGE_MAX and age_end <= C.AGE_PLUS_TENURE_MAX
    rows.append(_row("Age & maturity age", "PASS" if ok else "FAIL",
                     f"age {x['age']:.0f}, {age_end:.1f} at maturity", "21-60; maturity <= 65", "P-1.1"))
    rows.append(_row("Employment stability", _tier(x["years_employed"], C.TENURE_YEARS_REVIEW,
                     C.TENURE_YEARS_DECLINE, higher_is_worse=False),
                     f"{x['years_employed']} yrs", ">= 1 yr (review 0.5-1)", "P-1.2"))
    rows.append(_row("Minimum income", "PASS" if m["verified_income"] >= C.MIN_INCOME else "FAIL",
                     f"INR {m['verified_income']:,.0f}", f">= INR {C.MIN_INCOME:,}", "P-2.1"))
    rows.append(_row("Loan to monthly income", _tier(m["loan_to_monthly_income"], C.LTI_REFER, C.LTI_DECLINE),
                     f"{m['loan_to_monthly_income']:.1f}x", f"<= {C.LTI_REFER}x (decline > {C.LTI_DECLINE}x)", "P-2.2"))
    rows.append(_row("Debt to income (post-loan)", _tier(m["dti_post"], C.DTI_REFER, C.DTI_DECLINE),
                     f"{m['dti_post']:.1%}", f"<= {C.DTI_REFER:.0%} (decline > {C.DTI_DECLINE:.0%})", "P-3.1"))
    rows.append(_row("Credit score", _tier(x["credit_score"], C.SCORE_PASS, C.SCORE_REFER, higher_is_worse=False),
                     f"{x['credit_score']:.0f}", f">= {C.SCORE_PASS} (decline < {C.SCORE_REFER})", "P-4.1"))
    b = x["bounce_count"]
    rows.append(_row("Repayment conduct (bounces, 6m)", "FAIL" if b >= C.BOUNCE_DECLINE else "WARN" if b >= C.BOUNCE_REFER else "PASS",
                     f"{b:.0f}", f"<= 1 (decline >= {C.BOUNCE_DECLINE})", "P-4.2"))
    inc = [f for f in fraud["flags"] if f["clause"] == "P-5.1"]
    rows.append(_row("Income verification", "WARN" if inc else "PASS",
                     inc[0]["detail"] if inc else "slip matches bank credits", "within 10%", "P-5.1"))
    idn = [f for f in fraud["flags"] if f["clause"] == "P-5.2"]
    rows.append(_row("Identity consistency", "WARN" if idn else "PASS",
                     idn[0]["detail"] if idn else "consistent", "no mismatch", "P-5.2"))
    rows.append(_row("Probability of default", _tier(pd_, C.PD_REFER, C.PD_DECLINE),
                     f"{pd_:.1%}", f"< {C.PD_REFER:.0%} (decline > {C.PD_DECLINE:.0%})", "P-7.1"))
    return rows


def decide(checks: list, fraud: dict, missing_docs: list) -> str:
    if missing_docs:
        return "REQUEST_DOCUMENTS"
    if fraud["flagged"]:
        return "REFER_FRAUD_REVIEW"
    statuses = {c["status"] for c in checks}
    if "FAIL" in statuses:
        return "DECLINE_RECOMMENDED"
    if "WARN" in statuses:
        return "REFER_MANUAL_REVIEW"
    return "APPROVE_RECOMMENDED"
