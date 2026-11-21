"""
ASO dashboard -- run from the project root:   streamlit run ui/app.py
Pages: results overview, policy comparison, ablation, break-even explorer,
live what-if scheduler, carbon curves, legacy (v1) pipeline results.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.evaluation.world import World
from src.evaluation.policies import ASOPolicy, Ctx, DELAYS_SEC
from src.evaluation.sensitivity import expected_nsg, make_pod
from src.layer7_execution import live_events

BENCH = ROOT / "results" / "benchmark"
STAY = "Stay (no orchestration)"
st.set_page_config(page_title="ASO - Carbon-aware orchestration", layout="wide", page_icon="🌱")


@st.cache_data
def load_summary():
    f = BENCH / "summary.csv"
    return pd.read_csv(f) if f.exists() else None


@st.cache_data
def load_runs():
    f = BENCH / "benchmark_runs.csv"
    return pd.read_csv(f) if f.exists() else None


def need_results():
    st.warning("No benchmark results found. Run:  `python -m src.evaluation.benchmark --csv-path data/borg_traces_data.csv`")
    st.stop()


page = st.sidebar.radio("Page", ["Overview", "Policy comparison", "Ablation & confidence sweep",
                                 "Break-even explorer", "Live what-if scheduler", "Live cluster",
                                 "Optimizers (GA / PSO / NSGA-II)", "Carbon curves", "Legacy pipeline (v1)"])
st.sidebar.caption("Agentic AI-driven Sustainability Orchestrator (ASO)\n\nScale x1 = compressed trace durations; x120 / x720 are what-if job-length scenarios. Carbon intensity is simulated.")

# ------------------------------------------------------------------ Overview
if page == "Overview":
    st.title("🌱 Agentic AI-driven Sustainability Orchestrator")
    st.markdown("Carbon-aware spatio-temporal workload orchestration with migration-overhead, "
                "uncertainty (Monte Carlo digital twin) and SLA-risk gating.")
    s = load_summary()
    if s is None:
        need_results()
    c = st.columns(3)
    scale = c[0].selectbox("Job-duration scale", sorted(s.scale.unique()), index=len(s.scale.unique()) - 1)
    slack = c[1].selectbox("Deadline slack", sorted(s.slack.unique()))
    mode = c[2].selectbox("Migration mode", sorted(s["mode"].unique()))
    d = s[(s.scale == scale) & (s.slack == slack) & (s["mode"] == mode)].set_index("policy")
    aso = d.loc["ASO (proposed)"]
    k = st.columns(4)
    k[0].metric("Carbon saving vs Stay", f"{aso.carbon_saving_pct:.1f}%", f"95% CI [{aso.ci_lo:.1f}, {aso.ci_hi:.1f}]")
    k[1].metric("Extra SLA violations", f"{aso.extra_sla_violation_pct:.2f} pp")
    k[2].metric("Jobs migrated", f"{aso.migration_pct:.1f}%")
    k[3].metric("Mean delay", f"{aso.mean_delay_min:.1f} min")
    st.subheader("Same scenario, all policies")
    show = d[["carbon_saving_pct", "ci_lo", "ci_hi", "extra_sla_violation_pct", "migration_pct", "p_value_vs_stay"]].round(3)
    st.dataframe(show.sort_values("carbon_saving_pct", ascending=False), width="stretch")

# --------------------------------------------------------- Policy comparison
elif page == "Policy comparison":
    st.title("Policy comparison")
    s = load_summary()
    if s is None:
        need_results()
    mode = st.selectbox("Migration mode", sorted(s["mode"].unique()))
    s = s[(s["mode"] == mode) & (~s.policy.str.startswith("- ")) & (~s.policy.str.contains("confidence|report gate"))]
    s = s.assign(scenario=s.apply(lambda r: f"x{r.scale:g} / {r.slack}", axis=1))
    fig = px.bar(s, x="scenario", y="carbon_saving_pct", color="policy", barmode="group",
                 error_y=s.ci_hi - s.carbon_saving_pct, error_y_minus=s.carbon_saving_pct - s.ci_lo,
                 labels={"carbon_saving_pct": "Carbon saving vs Stay (%)"})
    fig.update_yaxes(range=[-60, 100])
    st.plotly_chart(fig, width="stretch")
    st.caption("Axis clipped at -60%: naive greedy policies can be thousands of % WORSE on short jobs because "
               "migration overhead dominates. See the table below for exact values.")
    pool = s[s.scale > 1].groupby("policy")[["carbon_saving_pct", "extra_sla_violation_pct", "migration_pct"]].mean().reset_index()
    st.caption("Trade-off plot excludes scale x1, where naive policies are thousands of % worse and would squash the axis.")
    fig2 = px.scatter(pool, x="extra_sla_violation_pct", y="carbon_saving_pct", text="policy", size_max=12,
                      labels={"extra_sla_violation_pct": "Extra SLA violations vs Stay (pp)",
                              "carbon_saving_pct": "Carbon saving (%, mean over scenarios)"})
    fig2.update_traces(textposition="top center", marker_size=12)
    st.plotly_chart(fig2, width="stretch")
    st.dataframe(s.drop(columns=["scenario"]).round(3), width="stretch")

# ------------------------------------------------------------------ Ablation
elif page == "Ablation & confidence sweep":
    st.title("Ablation study and confidence sweep")
    s = load_summary()
    if s is None:
        need_results()
    a = s[s.policy.str.startswith(("ASO", "- "))]
    a = a.groupby("policy")[["carbon_saving_pct", "extra_sla_violation_pct", "migration_pct"]].mean().reset_index()
    fig = go.Figure()
    fig.add_bar(x=a.policy, y=a.carbon_saving_pct, name="Carbon saving (%)")
    fig.add_bar(x=a.policy, y=a.extra_sla_violation_pct, name="Extra SLA violations (pp)", marker_color="crimson")
    fig.update_layout(barmode="group", xaxis_tickangle=-30)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(a.round(3), width="stretch")

# --------------------------------------------------------- Break-even
elif page == "Break-even explorer":
    st.title("When is migration worth it?")
    c = st.columns(3)
    mem = c[0].slider("Job memory / checkpoint (GB)", 0.5, 32.0, 4.0)
    kwh = c[1].select_slider("Network energy (kWh/GB)", [0.001, 0.003, 0.01, 0.03, 0.06], value=0.01)
    mode = c[2].radio("Migration mode", ["live", "new"], horizontal=True)
    durs = np.logspace(1, 5.3, 60)
    nsg = [expected_nsg(d, kwh, mode, mem) for d in durs]
    fig = go.Figure(go.Scatter(x=durs / 60, y=nsg, mode="lines"))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_xaxes(type="log", title="Job duration (minutes)")
    fig.update_yaxes(title="Expected NSG, DC_ASIA -> DC_EU (gCO2)")
    st.plotly_chart(fig, width="stretch")
    pos = [d for d, v in zip(durs, nsg) if v > 0]
    st.info(f"Break-even duration ≈ {min(pos) / 60:.1f} min" if pos else "Migration never pays off in this range.")

# ---------------------------------------------------------- What-if
elif page == "Live what-if scheduler":
    st.title("Live what-if scheduler")
    c = st.columns(4)
    region = c[0].selectbox("Job currently in", ["DC_ASIA", "DC_US", "DC_EU"])
    dur_min = c[1].number_input("Predicted runtime (min)", 1.0, 2000.0, 90.0)
    mem = c[2].number_input("Memory (GB)", 0.5, 64.0, 4.0)
    hour = c[3].slider("Arrival hour", 0, 23, 2)
    c = st.columns(4)
    slack = c[0].slider("Deadline slack (x runtime)", 1.1, 24.0, 3.0)
    cv = c[1].slider("Runtime uncertainty (relative σ)", 0.05, 1.0, 0.15)
    sla = c[2].slider("SLA-risk limit", 0.01, 0.9, 0.10)
    conf = c[3].slider("Required P(NSG>0)", 0.5, 0.99, 0.90)
    mode = st.radio("Migration mode", ["live", "new"], horizontal=True)
    w = World(0, migration_mode=mode)
    node = w.rep_node[region]
    dur = dur_min * 60
    pod = make_pod(dur, mem)
    pod = pod.__class__(**{**pod.__dict__, "source_node": node.node_id, "source_region": region,
                           "sla_deadline_sec": dur * slack})
    ctx = Ctx(pod, hour * 3600.0, dur, dur * cv)
    aso = ASOPolicy(n_mc=400, sla_limit=sla, confidence=conf)
    rows = []
    for r in w.regions:
        dst = node if r == region else w.rep_node[r]
        for d in DELAYS_SEC:
            if dst.node_id == node.node_id and d == 0:
                continue
            nsg, fin = aso.evaluate(w, ctx, dst, d)
            p, risk = float(np.mean(nsg > 0)), float(np.mean(fin > pod.sla_deadline_sec))
            rows.append(dict(target=r, delay_min=d / 60, mean_NSG_g=nsg.mean(), P_NSG_pos=p, SLA_risk=risk,
                             viable=(p >= conf and risk <= sla)))
    t = pd.DataFrame(rows)
    dec = aso.decide(w, ctx)
    reg = w.node_by_id[dec.node_id].region
    st.success(f"**Decision:** run in **{reg}**, delay **{dec.delay_sec / 60:.0f} min**  \n{dec.note}")
    st.dataframe(t.round(3).sort_values("mean_NSG_g", ascending=False), width="stretch")
    best = t.sort_values("mean_NSG_g", ascending=False).iloc[0]
    nsg, _ = aso.evaluate(w, ctx, w.rep_node[best.target] if best.target != region else node, best.delay_min * 60)
    fig = px.histogram(x=nsg, nbins=40, labels={"x": "NSG (gCO2) for best candidate"})
    fig.add_vline(x=0, line_dash="dash")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------- Live cluster
elif page == "Live cluster":
    st.title("🔴 Live cluster")
    c = st.columns([1, 1, 2])
    refresh = c[0].slider("Refresh every (s)", 1, 10, 2)
    source = c[1].radio("Cluster view", ["Auto (Kubernetes if reachable)", "Event log only"], horizontal=False)
    c[2].info("Start a stream in another terminal:\n\n"
              "`python -m src.layer7_execution.custom_scheduler --engine aso --reset-log` (real Kubernetes / Kind)\n\n"
              "`python -m src.layer7_execution.live_simulator --jobs 200` (no cluster needed)")

    def render_live(use_k8s):
        from src.evaluation.world import World
        from src.layer1_infra.cluster_config import create_default_node_pools
        evs = live_events.read_events()
        if not evs:
            st.warning("No events yet - start the scheduler or the simulator (see above).")
            return
        decisions = [e for e in evs if e["kind"] == "decision"]
        arrived = {e["pod_id"] for e in evs if e["kind"] == "arrive"}
        bound = {e["pod_id"] for e in evs if e["kind"] == "bind"}
        done = [e for e in evs if e["kind"] == "complete"]
        last_dec = {}
        for e in decisions:
            last_dec[e["pod_id"]] = e
        acts = pd.Series([d["action"] for d in last_dec.values()]).value_counts() if last_dec else pd.Series(dtype=int)
        latest = max((e.get("sim_t", 0) for e in evs if "sim_t" in e), default=0.0)
        start = next((e for e in evs if e["kind"] == "start"), {})
        snap = live_events.snapshot_cluster() if use_k8s else None

        k = st.columns(6)
        k[0].metric("Engine", start.get("engine", "?"))
        k[1].metric("Simulated clock", f"{int(latest // 3600) % 24:02d}:{int(latest % 3600 // 60):02d}")
        k[2].metric("Pods seen / bound", f"{len(arrived)} / {len(bound)}")
        k[3].metric("Pending", len(arrived - bound))
        k[4].metric("Migrated", int(acts.get("MIGRATE", 0)))
        k[5].metric("Delayed (to greener hour)", int(acts.get("DELAY", 0)))
        st.caption("Data source: " + ("real Kubernetes API (kind cluster)" if snap else "event log"))

        # ---- cluster map ----
        nodes = create_default_node_pools()
        if snap:
            on = {}
            for p in snap["pods"]:
                if p["node_id"] and p["phase"] in ("Running", "Pending"):
                    on.setdefault(p["node_id"], []).append(p["pod_id"])
        else:
            on = live_events.running_pods(evs)
        ci = (decisions[-1].get("ci_by_region") if decisions else None) or {}
        regions = sorted({n.region for n in nodes})
        cols = st.columns(len(regions))
        for col, r in zip(cols, regions):
            with col:
                val = ci.get(r)
                st.subheader(r)
                if val is not None:
                    st.caption(f"grid intensity now: **{val:.0f} gCO2/kWh**")
                    st.progress(min(1.0, val / 700.0))
                for n in [n for n in nodes if n.region == r]:
                    pods_here = on.get(n.node_id, [])
                    st.markdown(f"**{n.node_id}** &nbsp; `{len(pods_here)} running` &nbsp; <small>{int(n.mips_rating)} MIPS</small>",
                                unsafe_allow_html=True)
                    st.progress(min(1.0, len(pods_here) / 8.0))
                    if pods_here:
                        st.caption(", ".join(p[-14:] for p in pods_here[:5]) + (" ..." if len(pods_here) > 5 else ""))

        # ---- carbon tally + action mix ----
        left, right = st.columns([2, 1])
        with left:
            tally = [e for e in done if e.get("carbon_g") is not None]
            if tally and any("carbon_stay_g" in e for e in tally):
                t = pd.DataFrame(tally)
                fig = go.Figure()
                fig.add_scatter(y=t.carbon_g.cumsum(), name="ASO (chosen)", line=dict(color="#1B7F8C", width=3))
                fig.add_scatter(y=t.carbon_stay_g.cumsum(), name="Stay (no orchestration)", line=dict(color="#8E9AAF"))
                if "carbon_greedy_g" in t:
                    fig.add_scatter(y=t.carbon_greedy_g.cumsum(), name="Greedy lowest-CI", line=dict(color="#E9A23B"))
                fig.update_layout(title="Cumulative carbon of completed jobs (gCO2)", xaxis_title="completed jobs",
                                  height=330, margin=dict(t=40, b=10))
                st.plotly_chart(fig, width="stretch")
                sav = 100 * (1 - t.carbon_g.sum() / max(t.carbon_stay_g.sum(), 1e-9))
                viol = int(t.sla_violated.sum()) if "sla_violated" in t else 0
                st.success(f"ASO so far: **{sav:.1f}% less carbon than Stay** over {len(t)} completed jobs, {viol} SLA violation(s).")
            elif tally:
                t = pd.DataFrame(tally)
                st.metric("Estimated carbon of completed pods (model estimate)", f"{t.carbon_g.sum():.2f} gCO2")
                st.caption("Real-cluster mode reports Kepler/model-based estimates, not a counterfactual Stay baseline.")
            else:
                st.info("Waiting for the first pod to complete ...")
        with right:
            if len(acts):
                fig = go.Figure(go.Pie(labels=list(acts.index), values=list(acts.values), hole=0.5))
                fig.update_layout(title="Decisions", height=330, margin=dict(t=40, b=10))
                st.plotly_chart(fig, width="stretch")

        # ---- decision feed ----
        st.subheader("Decision feed")
        rows = []
        for e in decisions[-14:][::-1]:
            rows.append({"time": pd.to_datetime(e["ts"], unit="s").strftime("%H:%M:%S"), "pod": e["pod_id"][-18:],
                         "action": {"MIGRATE": "🔀 MIGRATE", "DELAY": "⏳ DELAY", "START": "▶ START"}.get(e["action"], e["action"]),
                         "from": e.get("source_node"), "to": e.get("target_node"),
                         "delay (sim min)": round(e.get("delay_s", 0) / 60, 1), "why": e.get("rationale", "")})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.fragment(run_every=refresh)(render_live)(source.startswith("Auto"))

# ------------------------------------------------------------- Optimizers
elif page == "Optimizers (GA / PSO / NSGA-II)":
    st.title("Placement optimisers: GA vs PSO vs NSGA-II")
    f = BENCH / "optimizer_comparison.csv"
    if not f.exists():
        st.warning("Run:  `python -m src.layer3_scheduling.ga_optimizer --csv-path data/borg_traces_data.csv`")
        st.stop()
    d = pd.read_csv(f)
    g = d.groupby("method")[["fitness", "carbon_vs_stay_pct", "latency_penalty", "runtime_s", "evaluations"]].mean().reset_index()
    g = g.sort_values("fitness")
    c = st.columns(2)
    c[0].plotly_chart(px.bar(g, x="method", y="fitness", title="Scalar fitness (lower is better)"), width="stretch")
    c[1].plotly_chart(px.bar(g, x="method", y="carbon_vs_stay_pct", title="Carbon saving vs Stay (%)"), width="stretch")
    cv = BENCH / "optimizer_convergence.json"
    if cv.exists():
        import json
        hist = json.load(open(cv))
        fig = go.Figure()
        for k, v in hist.items():
            if v:
                fig.add_scatter(y=v, name=k)
        fig.update_layout(title="Best fitness per generation / iteration (first batch)", xaxis_title="generation")
        st.plotly_chart(fig, width="stretch")
    st.dataframe(g.round(4), width="stretch")
    st.caption("All methods optimise the same objective with the same evaluation budget. When RAM capacity does not bind, "
               "placements are almost independent per pod, so a simple per-pod greedy search is already near-optimal; "
               "GA/NSGA-II earn their keep when capacity constraints interact.")

# --------------------------------------------------------- Carbon curves
elif page == "Carbon curves":
    st.title("Carbon-intensity model")
    w = World(0)
    t = np.linspace(0, 86400, 97)
    fig = go.Figure()
    for r in w.regions:
        fig.add_scatter(x=t / 3600, y=w.ci_forecast(r, t), name=f"{r} forecast")
        fig.add_scatter(x=t / 3600, y=w.ci_truth(r, t), name=f"{r} 'true' (with weather)", line=dict(dash="dot"))
    fig.update_xaxes(title="Hour of day"); fig.update_yaxes(title="gCO2/kWh")
    st.plotly_chart(fig, width="stretch")
    st.caption("Synthetic diurnal model. Drop a real hourly CSV into the carbon feed to validate on real grid data.")

# ------------------------------------------------------------- Legacy
else:
    st.title("Legacy v1 pipeline outputs")
    f1, f2 = ROOT / "results" / "hourly_summary.csv", ROOT / "results" / "layer5_candidate_results.csv"
    if f1.exists():
        h = pd.read_csv(f1)
        st.plotly_chart(px.line(h, x="hour", y="mean_nsg_gco2", markers=True), width="stretch")
        st.dataframe(h, width="stretch")
    if f2.exists():
        c = pd.read_csv(f2)
        st.plotly_chart(px.scatter(c, x="p_nsg_positive", y="sla_risk_score", color="is_statistically_viable",
                                   hover_data=["pod_id", "target_node_id"]), width="stretch")
    if not (f1.exists() or f2.exists()):
        st.info("No v1 result files found in results/.")