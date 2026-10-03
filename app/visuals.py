"""HTML helpers + Plotly figure builders for the underwriter workbench."""
import html

import plotly.express as px
import plotly.graph_objects as go

PALETTE = ["#6c5ce7", "#00b4d8", "#06d6a0", "#ffb703", "#ef476f", "#ff7b00", "#8338ec", "#118ab2"]
STATUS_COLORS = {"PASS": "#06d6a0", "WARN": "#ffb703", "FAIL": "#ef476f"}
REC_COLORS = {"APPROVE_RECOMMENDED": "#06d6a0", "REFER_MANUAL_REVIEW": "#ffb703",
              "REFER_FRAUD_REVIEW": "#ff7b00", "DECLINE_RECOMMENDED": "#ef476f",
              "REQUEST_DOCUMENTS": "#8338ec"}
REC_ICONS = {"APPROVE_RECOMMENDED": "✅", "REFER_MANUAL_REVIEW": "🟡", "REFER_FRAUD_REVIEW": "🕵️",
             "DECLINE_RECOMMENDED": "⛔", "REQUEST_DOCUMENTS": "📎"}


def _layout(fig, title=None, height=320):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1b1b3a")) if title else None,
        height=height, margin=dict(l=20, r=20, t=50 if title else 20, b=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#33334d"),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0, title=None))
    return fig


# ---------------------------------------------------------------- HTML
def kpi(label, value, sub="", color="#6c5ce7", icon=""):
    return (f'<div class="kpi" style="--c:{color}"><div class="kpi-icon">{icon}</div>'
            f'<div class="kpi-label">{html.escape(str(label))}</div><div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def kpi_row(items):
    return '<div class="kpi-row">' + "".join(kpi(*i) for i in items) + "</div>"


def badge(text, color):
    return f'<span class="badge" style="background:{color}">{html.escape(str(text))}</span>'


def rec_banner(rec):
    c = REC_COLORS.get(rec, "#6c5ce7")
    return (f'<div class="rec-banner" style="--c:{c}"><span class="rec-icon">{REC_ICONS.get(rec, "")}</span>'
            f'<div><div class="kpi-label">System recommendation</div>'
            f'<div class="rec-text">{rec.replace("_", " ")}</div></div>'
            f'<div class="rec-note">Automated recommendation only. A human underwriter makes the final call (P-7.2).</div></div>')


def flow(steps, active=None):
    parts = []
    for i, s in enumerate(steps):
        parts.append(f'<span class="step" style="--c:{PALETTE[i % 8]}">{html.escape(s)}</span>')
    return '<div class="flow">' + '<span class="arrow">→</span>'.join(parts) + "</div>"


# ---------------------------------------------------------------- figures
def gauge(value, title, lo, hi, fmt="%", invert=False):
    """Gauge for a 0-100 scale. lo/hi are the amber/red cut points (policy thresholds)."""
    good, mid, bad = ("#c9f5e5", "#fff1c9", "#ffd6dd")
    steps = ([dict(range=[0, lo], color=good), dict(range=[lo, hi], color=mid), dict(range=[hi, 100], color=bad)]
             if not invert else
             [dict(range=[0, lo], color=bad), dict(range=[lo, hi], color=mid), dict(range=[hi, 100], color=good)])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number=dict(suffix=fmt),
        title=dict(text=title, font=dict(size=14)),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#6c5ce7"), steps=steps)))
    return _layout(fig, None, 240)


def shap_bar(factors):
    f = list(reversed(factors))
    fig = go.Figure(go.Bar(
        x=[x["shap"] for x in f], y=[f"{x['feature']} = {x['value']}" for x in f], orientation="h",
        marker=dict(color=["#ef476f" if x["shap"] > 0 else "#06d6a0" for x in f]),
        text=[f"{x['shap']:+.2f}" for x in f], textposition="outside"))
    fig.update_xaxes(title="log-odds contribution (red raises risk, green lowers it)", zeroline=True)
    return _layout(fig, "Why the model scored it this way (SHAP)", 320)


def checks_donut(checks):
    counts = {s: sum(c["status"] == s for c in checks) for s in ("PASS", "WARN", "FAIL")}
    fig = go.Figure(go.Pie(labels=list(counts), values=list(counts.values()), hole=0.62, sort=False,
                           marker=dict(colors=[STATUS_COLORS[k] for k in counts], line=dict(color="white", width=2)),
                           textinfo="label+value"))
    fig.add_annotation(text=f"<b>{len(checks)}</b><br>checks", showarrow=False, font=dict(size=18))
    return _layout(fig, "Policy checks", 300)


def dti_waterfall(m):
    cur, new = m["dti_current"] * 100, (m["dti_post"] - m["dti_current"]) * 100
    fig = go.Figure(go.Waterfall(
        x=["Existing EMIs", "New loan EMI", "Post-loan DTI"], measure=["relative", "relative", "total"],
        y=[cur, new, 0], text=[f"{cur:.1f}%", f"+{new:.1f}%", f"{m['dti_post']*100:.1f}%"],
        increasing=dict(marker=dict(color="#6c5ce7")), totals=dict(marker=dict(color="#00b4d8"))))
    fig.add_hline(y=40, line_dash="dash", line_color="#ffb703", annotation_text="refer 40%")
    fig.add_hline(y=50, line_dash="dash", line_color="#ef476f", annotation_text="decline 50%")
    fig.update_yaxes(title="% of verified income")
    return _layout(fig, "Affordability: debt to income", 320)


def income_compare(x, m):
    labels = ["Declared (form)", "Salary slip", "Bank avg credit", "Verified (used)"]
    vals = [x.get("declared_income") or 0, x["slip_net_salary"], x["avg_salary_credit"], m["verified_income"]]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker=dict(color=["#ffb703", "#00b4d8", "#8338ec", "#06d6a0"]),
                           text=[f"{v:,.0f}" for v in vals], textposition="outside"))
    fig.update_yaxes(range=[0, max(vals) * 1.18], title="INR / month")
    return _layout(fig, "Income verification", 320)


def cashflow_stack(m):
    parts = {"Existing EMIs": m["existing_emi"], "New EMI": m["new_emi"],
             "Monthly spend": m["avg_monthly_spend"], "Surplus": max(m["monthly_surplus_after_emi"], 0)}
    fig = go.Figure(go.Pie(labels=list(parts), values=list(parts.values()), hole=0.55, sort=False,
                           marker=dict(colors=["#6c5ce7", "#00b4d8", "#ffb703", "#06d6a0"], line=dict(color="white", width=2))))
    return _layout(fig, "Where verified income goes (per month)", 300)


def balance_line(df):
    fig = px.line(df, x="date", y="balance", markers=False, color_discrete_sequence=["#6c5ce7"])
    fig.update_traces(fill="tozeroy", fillcolor="rgba(108,92,231,.12)")
    return _layout(fig, "Bank balance over the statement period", 280)


def rec_pie(series):
    c = series.value_counts()
    fig = go.Figure(go.Pie(labels=[i.replace("_", " ") for i in c.index], values=c.values, hole=0.55,
                           marker=dict(colors=[REC_COLORS.get(i, "#999") for i in c.index], line=dict(color="white", width=2))))
    return _layout(fig, "Recommendations", 320)


def confusion(df):
    ct = df.groupby(["expected", "got"]).size().reset_index(name="n")
    fig = px.density_heatmap(ct, x="got", y="expected", z="n", text_auto=True,
                             color_continuous_scale=["#f6f7fd", "#6c5ce7"])
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(tickangle=-25)
    return _layout(fig, "Expected vs. produced recommendation", 380)


def parity_bar(rates, threshold=0.8):
    fig = go.Figure(go.Bar(x=list(rates), y=[100 * v for v in rates.values()], marker_color=["#00b4d8", "#ef476f"],
                           text=[f"{100*v:.1f}%" for v in rates.values()], textposition="outside"))
    fig.update_yaxes(range=[0, 100], title="approve-recommendation rate %")
    return _layout(fig, "Approval rate by gender (not a model input)", 320)


def timeline(df):
    d = df.copy()
    fig = px.scatter(d, x="ts", y="node", color="node", hover_data=["application_id"],
                     color_discrete_sequence=PALETTE)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(showlegend=False)
    return _layout(fig, "Audit timeline", 320)
