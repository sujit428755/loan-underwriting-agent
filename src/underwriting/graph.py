"""Author: Sujit (github.com/sujit428755)

LangGraph workflow:
extract -> (missing docs? request_documents -> END)
        -> finance -> fraud -> credit_risk -> policy -> recommend -> HUMAN REVIEW (interrupt) -> finalize
"""
import operator
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import audit, credit_model, extraction, finance, fraud, memo, policy
from .config import MODEL_FEATURES, PROTECTED_ATTRIBUTES


class UState(TypedDict, total=False):
    application_id: str
    documents: dict
    extracted: dict
    metrics: dict
    fraud: dict
    credit: dict
    checks: list
    recommendation: str
    memo: str
    human_decision: dict
    final_status: str
    override: bool
    trace: Annotated[list, operator.add]


def _t(s, node, payload):
    return [audit.log(s.get("application_id", "?"), node, payload)]


def extract_node(s):
    e = extraction.extract(s["documents"])
    aid = s.get("application_id") or e["fields"].get("application_id") or "UNKNOWN"
    return {"application_id": aid, "extracted": e,
            "trace": _t({"application_id": aid}, "extract",
                        {"missing_docs": e["missing_docs"], "missing_fields": e["missing_fields"],
                         "fields": e["fields"]})}


def route_after_extract(s):
    e = s["extracted"]
    return "request_documents" if e["missing_docs"] or e["missing_fields"] else "finance"


def request_documents_node(s):
    e = s["extracted"]
    need = e["missing_docs"] or e["missing_fields"]
    text = (f"# Credit Memo - {s['application_id']}\n\n## Recommendation: REQUEST_DOCUMENTS\n\n"
            f"The file is incomplete (policy P-6.1), so no credit recommendation was produced.\n"
            f"Missing: {', '.join(need)}")
    return {"recommendation": "REQUEST_DOCUMENTS", "memo": text, "final_status": "PENDING_DOCUMENTS",
            "trace": _t(s, "request_documents", {"missing": need})}


def finance_node(s):
    m = finance.compute_metrics(s["extracted"]["fields"])
    return {"metrics": m, "trace": _t(s, "finance", m)}


def fraud_node(s):
    f = fraud.check_documents(s["extracted"]["fields"])
    return {"fraud": f, "trace": _t(s, "fraud", f)}


def credit_node(s):
    x, m = s["extracted"]["fields"], s["metrics"]
    feats = {**x, **m}
    assert not set(MODEL_FEATURES) & set(PROTECTED_ATTRIBUTES), "protected attribute used as feature"
    r = credit_model.predict(feats)
    return {"credit": r, "trace": _t(s, "credit_risk", {**r, "features": MODEL_FEATURES})}


def policy_node(s):
    checks = policy.run_checks(s["extracted"]["fields"], s["metrics"], s["credit"]["pd"], s["fraud"])
    return {"checks": checks, "trace": _t(s, "policy", {"checks": [(c["clause"], c["status"]) for c in checks]})}


def recommend_node(s):
    rec = policy.decide(s["checks"], s["fraud"], s["extracted"]["missing_docs"])
    state = {**s, "recommendation": rec}
    return {"recommendation": rec, "memo": memo.build_memo(state),
            "trace": _t(s, "recommend", {"recommendation": rec})}


def human_review_node(s):
    decision = interrupt({"application_id": s["application_id"], "recommendation": s["recommendation"],
                          "memo": s["memo"], "allowed_actions": ["approve", "reject", "request_more_info"]})
    return {"human_decision": decision}


NO_NOTE_OK = {"approve": {"APPROVE_RECOMMENDED", "REFER_MANUAL_REVIEW"},
              "reject": {"DECLINE_RECOMMENDED", "REFER_MANUAL_REVIEW", "REFER_FRAUD_REVIEW"},
              "request_more_info": None}


def finalize_node(s):
    h, rec = s["human_decision"], s["recommendation"]
    action = h.get("action")
    status = {"approve": "APPROVED", "reject": "REJECTED", "request_more_info": "MORE_INFO_REQUESTED"}.get(action, "INVALID_ACTION")
    ok_set = NO_NOTE_OK.get(action)
    override = ok_set is not None and rec not in ok_set
    if override and not (h.get("note") or "").strip():
        status = "OVERRIDE_BLOCKED_REASON_REQUIRED"  # policy P-7.2
    payload = {"recommendation": rec, "human_action": action, "reviewer": h.get("reviewer"),
               "note": h.get("note"), "override": override, "final_status": status}
    return {"final_status": status, "override": override, "trace": _t(s, "human_decision", payload)}


def build_graph(checkpointer=None):
    g = StateGraph(UState)
    for name, fn in [("extract", extract_node), ("request_documents", request_documents_node),
                     ("finance", finance_node), ("fraud", fraud_node), ("credit_risk", credit_node),
                     ("policy", policy_node), ("recommend", recommend_node),
                     ("human_review", human_review_node), ("finalize", finalize_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "extract")
    g.add_conditional_edges("extract", route_after_extract, ["request_documents", "finance"])
    g.add_edge("request_documents", END)
    for a, b in [("finance", "fraud"), ("fraud", "credit_risk"), ("credit_risk", "policy"),
                 ("policy", "recommend"), ("recommend", "human_review"), ("human_review", "finalize"),
                 ("finalize", END)]:
        g.add_edge(a, b)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def analyze(documents: dict, application_id: str | None = None) -> dict:
    """Run everything up to the recommendation without the human gate (used by evaluation/fairness)."""
    s: dict[str, Any] = {"documents": documents, "trace": []}
    if application_id:
        s["application_id"] = application_id

    def step(fn):
        u = fn(s)
        tr = u.pop("trace", [])
        s.update(u)
        s["trace"] = s["trace"] + tr

    step(extract_node)
    if route_after_extract(s) == "request_documents":
        step(request_documents_node)
        return s
    for fn in (finance_node, fraud_node, credit_node, policy_node, recommend_node):
        step(fn)
    return s
