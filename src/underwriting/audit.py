"""Append-only audit trail. Every agent step is stored so a decision can be reconstructed."""
import json
import time

from .config import AUDIT_LOG


def log(application_id: str, node: str, payload: dict, path=AUDIT_LOG) -> dict:
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "application_id": application_id,
             "node": node, "payload": payload}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass  # never fail an underwriting run because of logging
    return entry
