"""Underwriter workbench - run with:  streamlit run app/streamlit_app.py"""
import io
import json
import random
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os

import pandas as pd
import streamlit as st
import visuals as vz
from langgraph.types import Command
from underwriting import config as C
from underwriting import evaluate as EV
from underwriting import graph as G
from underwriting import synth
from underwriting.policy import index as policy_index

st.set_page_config(page_title="Loan Underwriting Studio", page_icon="🏦", layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={"About": "Autonomous Loan Underwriting Agent, built by Sujit. https://github.com/sujit428755"})

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 3.2rem; max-width: 1350px; }
.hero { background: linear-gradient(120deg,#3a0ca3 0%,#6c5ce7 45%,#00b4d8 100%);
  border-radius: 22px; padding: 30px 38px; color: white; margin-bottom: 18px;
  box-shadow: 0 12px 34px rgba(108,92,231,.35); position: relative; overflow: hidden; }
.hero:after { content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px;
  border-radius:50%; background:rgba(255,255,255,.10); }
.hero h1 { margin:0; font-size: 2.15rem; font-weight: 800; letter-spacing:-.5px; color:white; }
.hero p { margin:.4rem 0 0 0; opacity:.92; font-size:1.02rem; max-width: 900px; }
.pill { display:inline-block; padding:4px 12px; margin:10px 6px 0 0; border-radius:999px;
  background:rgba(255,255,255,.18); font-size:.78rem; font-weight:600; backdrop-filter: blur(4px); }
.kpi-row { display:grid; grid-template-columns: repeat(auto-fit,minmax(140px,1fr)); gap:14px; margin: 8px 0 18px 0; }
.kpi { background:white; border-radius:16px; padding:16px 18px; border-left:6px solid var(--c);
  box-shadow:0 4px 16px rgba(30,30,80,.07); transition: transform .15s; }
.kpi:hover { transform: translateY(-3px); }
.kpi-icon { font-size:1.3rem; }
.kpi-label { font-size:.74rem; text-transform:uppercase; letter-spacing:.8px; color:#7a7a9a; font-weight:600; }
.kpi-value { font-size:1.75rem; font-weight:800; color:var(--c); line-height:1.15; }
.kpi-sub { font-size:.78rem; color:#8b8ba7; }
.note-box { background:white; border:1px solid #e7e8f5; border-radius:14px; padding:16px 18px;
  line-height:1.8; font-size:.95rem; box-shadow:0 2px 10px rgba(30,30,80,.05); min-height:90px; }
.rec-banner { display:flex; align-items:center; gap:18px; flex-wrap:wrap; background:white; border-radius:16px;
  padding:16px 22px; border-left:8px solid var(--c); box-shadow:0 4px 16px rgba(30,30,80,.07); margin-bottom:14px; }
.rec-icon { font-size:2rem; }
.rec-text { font-size:1.5rem; font-weight:800; color:var(--c); letter-spacing:-.3px; }
.rec-note { margin-left:auto; color:#8b8ba7; font-size:.82rem; max-width:340px; }
.badge { color:white; padding:3px 12px; border-radius:999px; font-weight:700; font-size:.8rem; }
.flow { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:10px 0 18px 0; }
.step { background:white; border-radius:14px; padding:10px 16px; font-weight:600; font-size:.85rem;
  box-shadow:0 3px 12px rgba(30,30,80,.08); border-top:4px solid var(--c); }
.arrow { color:#a0a0c0; font-size:1.2rem; font-weight:700; }
.section-title { font-size:1.15rem; font-weight:700; margin:22px 0 6px 0; color:#1b1b3a; }
.stTabs [data-baseweb="tab-list"] { gap:6px; background:white; padding:6px; border-radius:14px;
  box-shadow:0 3px 12px rgba(30,30,80,.06); flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { border-radius:10px; padding:8px 16px; font-weight:600; }
.stTabs [aria-selected="true"] { background: linear-gradient(120deg,#6c5ce7,#00b4d8); color:white !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#1b1b3a 0%,#2d1b69 100%); }
[data-testid="stSidebar"] * { color:#f1f1ff !important; }
[data-testid="stSidebar"] input { color:#1b1b3a !important; }
div[data-testid="stMetric"] { background:white; border-radius:14px; padding:10px 14px;
  box-shadow:0 3px 12px rgba(30,30,80,.06); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading credit model and policy index...")
def load_graph():
    from underwriting import credit_model
    credit_model._model()  # train once if the model file is missing
    policy_index()
    return G.build_graph()


graph = load_graph()


def show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def html(x):
    st.markdown(x, unsafe_allow_html=True)


def section(t):
    html(f'<div class="section-title">{t}</div>')


def tint(x):
    return {"PASS": "background-color:#c9f5e5;font-weight:700", "WARN": "background-color:#fff1c9;font-weight:700",
            "FAIL": "background-color:#ffd6dd;font-weight:700"}.get(x, "")


SAMPLES = {
    "clean": "Clean file - should be approved",
    "high_dti": "Over-leveraged - high debt to income",
    "low_score": "Weak credit score",
    "fraud_mismatch": "Inflated salary slip (fraud signal)",
    "missing_doc": "Incomplete file - a document is missing",
    "bounces": "Repeated cheque bounces",
    "short_tenure_job": "New job - under 6 months",
}


@st.cache_data
def sample_docs(scenario):
    i = synth.SCENARIOS.index(scenario)
    return synth.make_case(500 + i, scenario, i + 1)[1]


def bank_rows(text):
    rows = []
    for line in (text or "").splitlines():
        p = [x.strip() for x in line.split("|")]
        if len(p) == 5 and p[0][:4].isdigit():
            try:
                rows.append({"date": p[0], "desc": p[1], "balance": float(p[4])})
            except ValueError:
                pass
    return pd.DataFrame(rows)


def read_audit():
    if not C.AUDIT_LOG.exists():
        return pd.DataFrame()
    recs = []
    for line in C.AUDIT_LOG.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
            recs.append({"ts": e["ts"], "application_id": e["application_id"], "node": e["node"],
                         "payload": json.dumps(e["payload"], default=str)[:300], "_raw": e["payload"]})
        except Exception:
            continue
    return pd.DataFrame(recs)


for k, v in {"result": None, "tid": None, "source": None, "docs": None, "batch": None,
             "suite": None, "fair": None, "last_error": None}.items():
    st.session_state.setdefault(k, v)


def run_one(docs, source):
    tid = str(uuid.uuid4())
    with st.status("Running the multi-agent underwriting graph...", expanded=False) as stt:
        r = graph.invoke({"documents": docs, "trace": []}, {"configurable": {"thread_id": tid}})
        stt.update(label="Graph paused for human review" if "__interrupt__" in r else "Graph complete",
                   state="complete")
    st.session_state.update(result=r, tid=tid, source=source, docs=docs)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🏦 Underwriting Studio")
    st.caption("Agentic credit decisioning · human-in-the-loop")
    st.header("Underwriter")
    reviewer = st.text_input("Your name", placeholder="e.g. S. Rao",
                             help="Recorded in the audit log with every decision (policy P-7.2).")
    st.header("System status")
    st.write(f"🧠 Commentary: **{'LLM (Groq)' if os.getenv('GROQ_API_KEY') else 'Template'}**")
    st.write(f"🛡️ Protected attributes in model: **None**")
    st.write(f"📜 Policy clauses indexed: **{len(policy_index().clauses)}**")
    st.divider()
    st.caption("Status colour legend")
    html("".join(vz.badge(k, c) + " " for k, c in vz.STATUS_COLORS.items()))
    st.caption("")
    html("".join(f'<div style="margin:4px 0">{vz.REC_ICONS[k]} <span style="font-size:.8rem">'
                 f'{k.replace("_", " ").title()}</span></div>' for k in vz.REC_COLORS))
    st.divider()
    st.caption("Synthetic data only. Portfolio project, not a lending system.")
    st.divider()
    st.markdown("**Built by Sujit**")
    st.markdown("[GitHub](https://github.com/sujit428755)")

# ------------------------------------------------------------------ hero
html("""
<div class="hero">
  <h1>🏦 Autonomous Loan Underwriting Studio</h1>
  <p>Upload a loan file and a team of agents extracts the documents, checks them for fraud, scores default risk,
  tests every lending-policy clause and drafts a cited credit memo. Nothing is final until a human underwriter decides.</p>
  <span class="pill">LangGraph</span><span class="pill">Policy RAG</span><span class="pill">Credit ML + SHAP</span>
  <span class="pill">Fraud checks</span><span class="pill">LLM never does math</span><span class="pill">Human approval</span>
  <span class="pill">Audit-ready</span><span class="pill">Built by Sujit</span>
</div>
""")

tab_uw, tab_batch, tab_eval, tab_review, tab_audit, tab_arch = st.tabs([
    "🏦 Underwrite", "📦 Batch", "📊 Evaluation", "👩‍💼 Review Queue", "📋 Audit Log", "🏗️ Architecture & Policy"])

# ======================================================================
# TAB 1 - UNDERWRITE
# ======================================================================
with tab_uw:
    c1, c2 = st.columns([3, 2])
    with c1:
        uploads = st.file_uploader("Upload the loan file (.txt): application_form, salary_slip, bank_statement, credit_report",
                                   type=["txt"], accept_multiple_files=True)
        if st.button("Run uploaded files", type="primary", disabled=not uploads):
            run_one({Path(u.name).stem: u.getvalue().decode("utf-8", "ignore") for u in uploads}, "uploaded files")
    with c2:
        scen = st.selectbox("...or pick a sample application", list(SAMPLES), format_func=lambda k: SAMPLES[k])
        if st.button("Run sample application"):
            run_one(sample_docs(scen), f"sample: {scen}")

    r = st.session_state.result
    if r is None:
        html('<div class="note-box" style="text-align:center;color:#8b8ba7">'
             '⬆️ Upload a loan file or pick a sample to watch the agents work.</div>')
    else:
        rec = r["recommendation"]
        awaiting = "__interrupt__" in r
        html(vz.rec_banner(rec))

        if not r.get("metrics"):  # incomplete file path
            e = r["extracted"]
            st.warning("The file is incomplete, so no credit recommendation was produced (policy P-6.1). "
                       "Missing: " + ", ".join(e["missing_docs"] or e["missing_fields"]))
            st.markdown(r["memo"])
        else:
            x, m, cr = r["extracted"]["fields"], r["metrics"], r["credit"]
            n_pass = sum(c["status"] == "PASS" for c in r["checks"])
            pd_col = "#06d6a0" if cr["pd"] < C.PD_REFER else "#ffb703" if cr["pd"] <= C.PD_DECLINE else "#ef476f"
            dti_col = "#06d6a0" if m["dti_post"] <= C.DTI_REFER else "#ffb703" if m["dti_post"] <= C.DTI_DECLINE else "#ef476f"
            nfl = len(r["fraud"]["flags"])
            html(vz.kpi_row([
                ("Default probability", f"{cr['pd']:.1%}", "gradient boosting model", pd_col, "🎯"),
                ("DTI after loan", f"{m['dti_post']:.1%}", f"was {m['dti_current']:.1%}", dti_col, "⚖️"),
                ("Verified income", f"{m['verified_income']:,.0f}", "INR / month (lower of slip and bank)", "#6c5ce7", "💼"),
                ("New EMI", f"{m['new_emi']:,.0f}", f"{x['tenure_months']} months @ {x['interest_rate']}%", "#00b4d8", "🧮"),
                ("Credit score", f"{x['credit_score']:.0f}", "bureau", "#8338ec", "📊"),
                ("Policy checks", f"{n_pass}/{len(r['checks'])}", "passed", "#06d6a0" if n_pass == len(r['checks']) else "#ffb703", "📜"),
                ("Fraud flags", nfl, "document consistency", "#ef476f" if nfl else "#06d6a0", "🕵️"),
            ]))

            t_over, t_pol, t_docs, t_memo, t_trace = st.tabs(
                ["Overview", "Policy checks", "Documents & extraction", "Credit memo", "Agent trace"])

            with t_over:
                g1, g2, g3 = st.columns(3)
                with g1:
                    show(vz.gauge(100 * cr["pd"], "Probability of default", 100 * C.PD_REFER, 100 * C.PD_DECLINE))
                with g2:
                    show(vz.gauge(100 * min(m["dti_post"], 1), "Debt to income (post-loan)", 100 * C.DTI_REFER, 100 * C.DTI_DECLINE))
                with g3:
                    show(vz.gauge(100 * n_pass / len(r["checks"]), "Policy checks passed", 60, 90, invert=True))
                a, b, c = st.columns(3)
                with a:
                    show(vz.income_compare(x, m))
                with b:
                    show(vz.dti_waterfall(m))
                with c:
                    show(vz.cashflow_stack(m))
                d, e, f = st.columns([2, 1, 2])
                with d:
                    if cr["top_factors"]:
                        show(vz.shap_bar(cr["top_factors"]))
                with e:
                    show(vz.checks_donut(r["checks"]))
                with f:
                    bdf = bank_rows((st.session_state.docs or {}).get("bank_statement"))
                    if not bdf.empty:
                        show(vz.balance_line(bdf))

            with t_pol:
                section("Policy checks - every result cites a clause from the lending policy")
                chk = pd.DataFrame(r["checks"])[["status", "check", "value", "threshold", "clause"]]
                table(chk.style.map(tint, subset=["status"]))
                section("Fraud and consistency flags")
                if r["fraud"]["flags"]:
                    for fl in r["fraud"]["flags"]:
                        st.error(f"**{fl['clause']}** · {fl['detail']}")
                else:
                    st.success("No document inconsistencies found.")
                section("Retrieved policy evidence")
                for ck in r["checks"]:
                    icon = {"PASS": "🟢", "WARN": "🟡", "FAIL": "🔴"}[ck["status"]]
                    with st.expander(f"{icon} {ck['clause']} · {ck['clause_title']} · {ck['value']}"):
                        st.write(ck["evidence"])

            with t_docs:
                section("Source documents")
                names = [n for n in C.REQUIRED_DOCS if n in (st.session_state.docs or {})]
                for tab, n in zip(st.tabs([n.replace("_", " ").title() for n in names]), names):
                    with tab:
                        st.code(st.session_state.docs[n], language="text")
                section("Extracted fields (deterministic parsers)")
                ex = pd.DataFrame([{"field": k, "value": str(v)} for k, v in x.items()])
                table(ex)

            with t_memo:
                st.markdown(r["memo"])

            with t_trace:
                section("Agent trace - what each node saw and produced")
                nodes = [t["node"] for t in r["trace"]]
                html(vz.flow([n.replace("_", " ") for n in nodes]))
                for i, t in enumerate(r["trace"], 1):
                    with st.expander(f"{i}. {t['node']} · {t['ts']}"):
                        st.json(t["payload"], expanded=False)

        # ---------------- human review gate
        section("Human review")
        if r.get("final_status") and not awaiting:
            ok = r["final_status"] in {"APPROVED", "REJECTED", "MORE_INFO_REQUESTED", "PENDING_DOCUMENTS"}
            (st.success if ok else st.error)(
                f"Final status: **{r['final_status']}**" + (" · override recorded" if r.get("override") else ""))
        elif awaiting:
            st.caption("The graph is paused at a LangGraph interrupt. The decision below resumes it and is written to the audit log.")
            note = st.text_area("Note (required if you disagree with the recommendation)", key="review_note")
            if not reviewer.strip():
                st.info("Enter your name in the sidebar to enable the decision buttons.")
            cols = st.columns(3)
            for col, action in zip(cols, ["approve", "reject", "request_more_info"]):
                if col.button(action.replace("_", " ").title(), key=f"btn_{action}", disabled=not reviewer.strip()):
                    ok_set = G.NO_NOTE_OK.get(action)
                    is_override = ok_set is not None and rec not in ok_set
                    if is_override and not note.strip():
                        st.error(f"'{action}' conflicts with the recommendation ({rec}). "
                                 "A written reason is required (policy P-7.2).")
                    else:
                        final = graph.invoke(Command(resume={"action": action, "reviewer": reviewer.strip(), "note": note}),
                                             {"configurable": {"thread_id": st.session_state.tid}})
                        st.session_state.result = final
                        st.rerun()

        if r.get("memo"):
            section("Download")
            d1, d2, d3 = st.columns(3)
            d1.download_button("⬇️ Credit memo (.md)", r["memo"], f"{r.get('application_id', 'memo')}_memo.md", "text/markdown")
            report = {k: r.get(k) for k in ("application_id", "recommendation", "metrics", "credit", "fraud", "final_status")}
            d2.download_button("⬇️ JSON report", json.dumps(report, indent=2, default=str), "underwriting_report.json", "application/json")
            if r.get("checks"):
                d3.download_button("⬇️ Policy checks (CSV)", pd.DataFrame(r["checks"]).to_csv(index=False),
                                   "policy_checks.csv", "text/csv")

# ======================================================================
# TAB 2 - BATCH
# ======================================================================
with tab_batch:
    section("Batch underwriting on synthetic applicants")
    st.caption("Generates applicants from the chosen scenarios and runs them through every agent (no human gate). "
               "Useful for seeing how the policy behaves across a portfolio.")
    b1, b2, b3 = st.columns([2, 1, 1])
    chosen = b1.multiselect("Scenarios", synth.SCENARIOS, default=synth.SCENARIOS, format_func=lambda k: SAMPLES[k])
    n = b2.slider("Applications", 10, 300, 60, step=10)
    seed = b3.number_input("Seed", value=7, step=1)
    if st.button("🚀 Run batch", type="primary", disabled=not chosen):
        rng, rows = random.Random(int(seed)), []
        bar = st.progress(0, text="Processing applications...")
        for i in range(n):
            sc = rng.choice(chosen)
            a, docs = synth.make_case(int(seed) * 1000 + i, sc, i)
            s = G.analyze(docs, a["application_id"])
            m, cr = s.get("metrics", {}), s.get("credit", {})
            rows.append({"application_id": a["application_id"], "scenario": sc, "gender": a["gender"],
                         "recommendation": s["recommendation"], "pd": cr.get("pd"), "dti_post": m.get("dti_post"),
                         "verified_income": m.get("verified_income"), "credit_score": a["credit_score"],
                         "fraud_flags": len(s.get("fraud", {}).get("flags", [])),
                         "expected": synth.EXPECTED[sc]})
            bar.progress((i + 1) / n, text=f"Processing application {i + 1}/{n}")
        bar.empty()
        st.session_state.batch = pd.DataFrame(rows)

    B = st.session_state.batch
    if B is None:
        st.info("Choose scenarios and press **Run batch**.")
    else:
        scored = B.dropna(subset=["pd"])
        appr = int((B["recommendation"] == "APPROVE_RECOMMENDED").sum())
        html(vz.kpi_row([
            ("Applications", len(B), "processed", "#6c5ce7", "📄"),
            ("Approve rate", f"{100 * appr / len(B):.0f}%", f"{appr} recommended", "#06d6a0", "✅"),
            ("Declined", int((B["recommendation"] == "DECLINE_RECOMMENDED").sum()), "recommended", "#ef476f", "⛔"),
            ("Fraud referrals", int((B["recommendation"] == "REFER_FRAUD_REVIEW").sum()), "", "#ff7b00", "🕵️"),
            ("Docs requested", int((B["recommendation"] == "REQUEST_DOCUMENTS").sum()), "incomplete files", "#8338ec", "📎"),
            ("Avg default prob.", f"{100 * scored['pd'].mean():.1f}%", "scored files", "#00b4d8", "🎯"),
        ]))
        c1, c2 = st.columns([1, 2])
        with c1:
            show(vz.rec_pie(B["recommendation"]))
        with c2:
            import plotly.express as px
            fig = px.scatter(scored, x="dti_post", y="pd", color="recommendation", size="verified_income",
                             hover_name="application_id", color_discrete_map={k: v for k, v in vz.REC_COLORS.items()})
            fig.add_vline(x=C.DTI_REFER, line_dash="dash", line_color="#ffb703")
            fig.add_vline(x=C.DTI_DECLINE, line_dash="dash", line_color="#ef476f")
            show(vz._layout(fig, "Risk map: post-loan DTI vs. default probability (dashed = policy limits)", 320))
        section("Results")
        f = st.multiselect("Filter by recommendation", sorted(B["recommendation"].unique()),
                           default=sorted(B["recommendation"].unique()))
        table(B[B["recommendation"].isin(f)].drop(columns=["expected"]))
        st.download_button("⬇️ Batch results (CSV)", B.to_csv(index=False), "batch_results.csv", "text/csv")

# ======================================================================
# TAB 3 - EVALUATION
# ======================================================================
with tab_eval:
    section("Evaluation dashboard")
    st.caption("Labelled synthetic cases test decision accuracy, extraction accuracy and the refusal to decide on "
               "incomplete files. A separate parity check compares approval rates across a protected attribute.")
    e1, e2, e3 = st.columns([1, 1, 1])
    per = e1.slider("Cases per scenario", 3, 15, 6)
    nfair = e2.slider("Fairness sample size", 100, 1000, 400, step=100)
    if e3.button("▶️ Run evaluation", type="primary"):
        with st.spinner("Scoring labelled cases and the fairness sample..."):
            st.session_state.suite = EV.run_suite(per_scenario=per)
            st.session_state.fair = EV.fairness(n=nfair)
    S, F = st.session_state.suite, st.session_state.fair
    if S is None:
        st.info("Press **Run evaluation**.")
    else:
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            show(vz.gauge(100 * S["decision_accuracy"], "Decision accuracy", 60, 90, invert=True))
        with g2:
            show(vz.gauge(100 * S["extraction_field_accuracy"], "Extraction accuracy", 60, 90, invert=True))
        with g3:
            show(vz.gauge(100 * (S["missing_doc_refusal_rate"] or 0), "Missing-doc refusal", 60, 90, invert=True))
        with g4:
            show(vz.gauge(100 * (F["disparate_impact_ratio"] or 0), "Disparate impact ratio", 60, 80, invert=True))
        (st.success if F["four_fifths_rule_pass"] else st.error)(
            f"Four-fifths rule {'passed' if F['four_fifths_rule_pass'] else 'FAILED'}: "
            f"ratio {F['disparate_impact_ratio']}, parity difference {F['demographic_parity_difference']}.")
        x1, x2 = st.columns([3, 2])
        with x1:
            show(vz.confusion(pd.DataFrame(S["rows"])))
        with x2:
            show(vz.parity_bar(F["approve_rate_by_gender"]))
        section("Accuracy by scenario")
        table(pd.DataFrame([{"scenario": k, "correct": v} for k, v in S["decision_accuracy_by_scenario"].items()]))
        if S["failures"]:
            st.warning("Mismatched cases"); table(pd.DataFrame(S["failures"]))
        st.caption("Honest reading: documents and parsers share one synthetic generator, so near-perfect extraction is "
                   "expected. These numbers prove the pipeline logic, not real-world robustness.")
        st.download_button("⬇️ Evaluation report (JSON)",
                           json.dumps({"suite": {k: v for k, v in S.items() if k != "rows"}, "fairness": F}, indent=2),
                           "eval_report.json", "application/json")

# ======================================================================
# TAB 4 - REVIEW QUEUE
# ======================================================================
with tab_review:
    section("Human decisions")
    st.caption("Every approval, rejection and override recorded by an underwriter (read from the audit log).")
    au = read_audit()
    dec = au[au["node"] == "human_decision"] if not au.empty else au
    if dec.empty:
        st.info("No human decisions recorded yet. Run an application and decide on it in the Underwrite tab.")
    else:
        rows = [{"time": r["ts"], "application": r["application_id"], "recommendation": r["_raw"].get("recommendation"),
                 "action": r["_raw"].get("human_action"), "reviewer": r["_raw"].get("reviewer"),
                 "override": r["_raw"].get("override"), "final_status": r["_raw"].get("final_status"),
                 "note": r["_raw"].get("note")} for _, r in dec.iterrows()]
        q = pd.DataFrame(rows)
        html(vz.kpi_row([
            ("Decisions", len(q), "recorded", "#6c5ce7", "🧾"),
            ("Approved", int((q["final_status"] == "APPROVED").sum()), "", "#06d6a0", "✅"),
            ("Rejected", int((q["final_status"] == "REJECTED").sum()), "", "#ef476f", "🚫"),
            ("More info", int((q["final_status"] == "MORE_INFO_REQUESTED").sum()), "", "#ffb703", "📎"),
            ("Overrides", int(q["override"].fillna(False).astype(bool).sum()), "against the system", "#8338ec", "⚠️"),
        ]))
        table(q.iloc[::-1])
        st.download_button("⬇️ Decisions (CSV)", q.to_csv(index=False), "human_decisions.csv", "text/csv")

# ======================================================================
# TAB 5 - AUDIT LOG
# ======================================================================
with tab_audit:
    section("📋 Audit log")
    au = read_audit()
    if au.empty:
        st.info("The audit log is empty. Process an application first.")
    else:
        html(vz.kpi_row([
            ("Log entries", len(au), "", "#6c5ce7", "🧾"),
            ("Applications", au["application_id"].nunique(), "distinct", "#00b4d8", "📁"),
            ("Human decisions", int((au["node"] == "human_decision").sum()), "", "#06d6a0", "👩‍💼"),
            ("Document requests", int((au["node"] == "request_documents").sum()), "", "#8338ec", "📎"),
        ]))
        show(vz.timeline(au))
        nodes = st.multiselect("Filter by agent", sorted(au["node"].unique()), default=sorted(au["node"].unique()))
        app_f = st.text_input("Filter by application ID", placeholder="e.g. APP-0001")
        view = au[au["node"].isin(nodes)]
        if app_f.strip():
            view = view[view["application_id"].str.contains(app_f.strip(), case=False)]
        table(view.drop(columns=["_raw"]).tail(300).iloc[::-1])
        d1, d2 = st.columns([1, 3])
        d1.download_button("⬇️ Download audit log", au.drop(columns=["_raw"]).to_csv(index=False), "audit_log.csv", "text/csv")
        with d2.expander("Danger zone"):
            if st.checkbox("I understand this permanently clears the audit trail") and st.button("Clear audit log"):
                C.AUDIT_LOG.unlink(missing_ok=True)
                st.rerun()

# ======================================================================
# TAB 6 - ARCHITECTURE & POLICY
# ======================================================================
with tab_arch:
    section("System architecture")
    html(vz.flow(["Loan file", "Extraction agent", "Finance calc", "Fraud agent", "Credit risk ML + SHAP",
                  "Policy RAG checks", "Recommendation + memo", "Human review (interrupt)", "Audit log"]))
    st.markdown("""
**Design rules.** The LLM never does arithmetic: income, EMI, DTI and default probability come from code and the ML model.
Verified income is the lower of the salary slip and the bank salary credits. A fraud signal outranks a decline.
An incomplete file gets a document request, never a guess. Protected attributes are not model inputs.
""")
    section("Policy explorer (clause retrieval)")
    qy = st.text_input("Ask the policy a question", placeholder="e.g. what happens if the credit score is below 650?")
    if qy.strip():
        hits = policy_index().retrieve(qy, 3)
        if hits:
            for h in hits:
                with st.expander(f"{h['id']} · {h['title']}  (relevance {h['score']})", expanded=True):
                    st.write(h["text"])
        else:
            st.info("No matching clause found.")
    with st.expander("All policy clauses"):
        for cl in policy_index().clauses.values():
            st.markdown(f"**{cl['id']} · {cl['title']}**  \n{cl['text']}")
    section("Policy thresholds enforced in code")
    thr = pd.DataFrame([
        ["P-3.1", "Debt to income", f"<= {C.DTI_REFER:.0%}", f"> {C.DTI_DECLINE:.0%}"],
        ["P-4.1", "Credit score", f">= {C.SCORE_PASS}", f"< {C.SCORE_REFER}"],
        ["P-2.2", "Loan to monthly income", f"<= {C.LTI_REFER}x", f"> {C.LTI_DECLINE}x"],
        ["P-7.1", "Probability of default", f"< {C.PD_REFER:.0%}", f"> {C.PD_DECLINE:.0%}"],
        ["P-4.2", "Bounces in 6 months", "<= 1", f">= {C.BOUNCE_DECLINE}"],
        ["P-5.1", "Slip vs bank income gap", f"within {C.INCOME_MISMATCH_TOL:.0%}", "fraud referral"],
        ["P-2.1", "Minimum verified income", f">= INR {C.MIN_INCOME:,}", "below"],
    ], columns=["clause", "rule", "pass", "decline / escalate"])
    table(thr)

html('<div style="text-align:center;color:#8b8ba7;font-size:.82rem;margin:40px 0 10px 0">'
     'Built by <b>Sujit</b> · <a href="https://github.com/sujit428755" target="_blank">GitHub</a> · Synthetic data only</div>')
