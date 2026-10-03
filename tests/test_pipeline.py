import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from langgraph.types import Command
from underwriting import graph as G, synth
from underwriting.finance import emi
from underwriting.policy import index


def test_emi_matches_known_value():
    assert emi(100_000, 12, 12) == pytest.approx(8884.88, abs=0.01)


@pytest.mark.parametrize("scenario", synth.SCENARIOS)
def test_scenarios_get_expected_recommendation(scenario):
    for seed in (1, 2, 3):
        a, docs = synth.make_case(seed, scenario, seed)
        assert G.analyze(docs, a["application_id"])["recommendation"] == synth.EXPECTED[scenario]


def test_missing_docs_produces_no_credit_decision():
    a, docs = synth.make_case(5, "missing_doc", 5)
    s = G.analyze(docs)
    assert s["recommendation"] == "REQUEST_DOCUMENTS" and "credit" not in s


def test_policy_retrieval_finds_dti_clause():
    assert index().retrieve("maximum debt to income ratio")[0]["id"] == "P-3.1"


def _run(scenario, resume):
    a, docs = synth.make_case(9, scenario, 9)
    g = G.build_graph()
    cfg = {"configurable": {"thread_id": "t"}}
    r = g.invoke({"documents": docs, "trace": []}, cfg)
    assert "__interrupt__" in r  # graph pauses for the human underwriter
    return g.invoke(Command(resume=resume), cfg)


def test_human_approval_gate_and_aligned_decision():
    f = _run("clean", {"action": "approve", "reviewer": "x", "note": ""})
    assert f["final_status"] == "APPROVED" and f["override"] is False


def test_override_without_reason_is_blocked():
    f = _run("low_score", {"action": "approve", "reviewer": "x", "note": ""})
    assert f["final_status"] == "OVERRIDE_BLOCKED_REASON_REQUIRED"
    f2 = _run("low_score", {"action": "approve", "reviewer": "x", "note": "Guarantor added, verified by branch"})
    assert f2["final_status"] == "APPROVED" and f2["override"] is True
