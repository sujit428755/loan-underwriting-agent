"""Write sample application folders to data/samples/<scenario>/ as .txt documents."""
import _path  # noqa
from underwriting import synth
from underwriting.config import ROOT

for i, sc in enumerate(synth.SCENARIOS):
    a, docs = synth.make_case(500 + i, sc, i + 1)
    d = ROOT / "data" / "samples" / sc
    d.mkdir(parents=True, exist_ok=True)
    for name, text in docs.items():
        (d / f"{name}.txt").write_text(text, encoding="utf-8")
    print("wrote", d)
