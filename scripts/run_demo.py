"""CLI demo: run one application through the graph, pause for the human underwriter, then finalize.
  python scripts/run_demo.py data/samples/clean --action approve --reviewer "S. Rao" """
import _path  # noqa
import argparse
import uuid

from langgraph.types import Command
from underwriting.extraction import load_folder
from underwriting.graph import build_graph

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--action", choices=["approve", "reject", "request_more_info"], default=None)
ap.add_argument("--reviewer", default="demo-underwriter")
ap.add_argument("--note", default="")
args = ap.parse_args()
print("=" * 60)
print(" Autonomous Loan Underwriting Agent  |  built by Sujit")
print(" github.com/sujit428755")
print("=" * 60)

graph = build_graph()
cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
result = graph.invoke({"documents": load_folder(args.folder), "trace": []}, cfg)

if "__interrupt__" not in result:
    print(result["memo"])
    print("\nFinal status:", result["final_status"])
    raise SystemExit

payload = result["__interrupt__"][0].value
print(payload["memo"])
action = args.action or input("\nUnderwriter action [approve/reject/request_more_info]: ").strip()
note = args.note if args.action else input("Note (required for overrides): ").strip()
final = graph.invoke(Command(resume={"action": action, "reviewer": args.reviewer, "note": note}), cfg)
print(f"\nFinal status: {final['final_status']} (override={final['override']})")
print("Audit trail written to logs/audit.jsonl")
