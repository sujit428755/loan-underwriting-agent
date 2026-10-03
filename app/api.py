"""FastAPI service. Run: uvicorn app.api:app --reload   (from project root, PYTHONPATH=src)"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel
from underwriting.graph import build_graph
from underwriting.policy import index

app = FastAPI(title="Autonomous Loan Underwriting Agent",
              description="Multi-agent underwriting with policy RAG, credit ML and human approval. Built by Sujit.",
              contact={"name": "Sujit", "url": "https://github.com/sujit428755"},
              license_info={"name": "MIT"})
graph = build_graph()


class Application(BaseModel):
    documents: dict[str, str]  # keys: application_form, salary_slip, bank_statement, credit_report


class Decision(BaseModel):
    action: str  # approve | reject | request_more_info
    reviewer: str
    note: str = ""


def _cfg(tid):
    return {"configurable": {"thread_id": tid}}


@app.post("/applications")
def submit(app_in: Application):
    tid = str(uuid.uuid4())
    r = graph.invoke({"documents": app_in.documents, "trace": []}, _cfg(tid))
    return {"thread_id": tid, "awaiting_human": "__interrupt__" in r, "application_id": r.get("application_id"),
            "recommendation": r.get("recommendation"), "memo": r.get("memo"), "final_status": r.get("final_status")}


@app.post("/applications/{tid}/decision")
def decide(tid: str, d: Decision):
    snap = graph.get_state(_cfg(tid))
    if not snap.next:
        raise HTTPException(409, "No pending human review for this application")
    if d.action not in {"approve", "reject", "request_more_info"}:
        raise HTTPException(422, "action must be approve, reject or request_more_info")
    r = graph.invoke(Command(resume=d.model_dump()), _cfg(tid))
    return {"final_status": r["final_status"], "override": r["override"], "trace_steps": len(r["trace"])}


@app.get("/applications/{tid}/trace")
def trace(tid: str):
    snap = graph.get_state(_cfg(tid))
    if not snap.values:
        raise HTTPException(404, "unknown application")
    return snap.values.get("trace", [])


@app.get("/policy/search")
def policy_search(q: str, k: int = 3):
    return index().retrieve(q, k)


@app.get("/health")
def health():
    return {"status": "ok"}
