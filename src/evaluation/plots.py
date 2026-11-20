"""Matplotlib figures for the report, generated from results/benchmark/*.csv."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ASO_MAIN = "ASO (proposed)"
ABL = ["ASO (proposed)", "- no migration-cost model", "- no SLA gate", "- no uncertainty (point estimate)",
       "- no temporal shifting", "- no spatial migration", "ASO, report gate (SLA<=0.80)"]
BASE = ["Random", "Fastest node", "Greedy lowest-CI", "CI-gap threshold (100)",
        "Spatio-temporal greedy (no overhead/risk model)", "ASO (proposed)"]


def _scen(s):
    return f"x{s.scale:g}\n{s.slack}"


def make_all_plots(out: Path):
    out = Path(out)
    pdir = out / "plots"
    pdir.mkdir(exist_ok=True)
    s = pd.read_csv(out / "summary.csv")
    runs = pd.read_csv(out / "benchmark_runs.csv")
    s["scen"] = s.apply(_scen, axis=1)
    scens = list(dict.fromkeys(s.scen))

    # 1. carbon saving per policy, per scenario (with 95% CI)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    w = 0.8 / len(BASE)
    for i, p in enumerate(BASE):
        d = s[s.policy == p].set_index("scen").reindex(scens)
        y = d.carbon_saving_pct.values
        ax.bar(np.arange(len(scens)) + i * w, y, w, label=p.replace(" (no overhead/risk model)", "*"),
               yerr=[y - d.ci_lo.values, d.ci_hi.values - y], capsize=2)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(np.arange(len(scens)) + 0.4 - w / 2); ax.set_xticklabels(scens)
    ax.set_ylabel("Carbon saving vs Stay (%)"); ax.legend(fontsize=7, ncol=2)
    ax.set_title("Net carbon saving (execution + migration), bootstrap 95% CI\n(x = job-duration scale; tight/mixed = deadline slack)")
    fig.tight_layout(); fig.savefig(pdir / "policy_carbon_saving.png", dpi=160); plt.close(fig)

    # 2. Pareto: carbon saving vs extra SLA violations (pooled over scenarios)
    fig, ax = plt.subplots(figsize=(7, 5))
    pool = s.groupby("policy")[["carbon_saving_pct", "extra_sla_violation_pct"]].mean()
    for p, r in pool.iterrows():
        ax.scatter(r.extra_sla_violation_pct, r.carbon_saving_pct, s=80 if p == ASO_MAIN else 40,
                   marker="*" if p == ASO_MAIN else "o", zorder=3)
        ax.annotate(p[:28], (r.extra_sla_violation_pct, r.carbon_saving_pct), fontsize=6, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("Extra SLA violations vs Stay (percentage points)"); ax.set_ylabel("Carbon saving vs Stay (%)")
    ax.set_title("Carbon / SLA trade-off (top-left is best)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(pdir / "pareto_carbon_vs_sla.png", dpi=160); plt.close(fig)

    # 3. Ablation
    fig, ax = plt.subplots(figsize=(9, 4.5))
    a = s[s.policy.isin(ABL)].groupby("policy")[["carbon_saving_pct", "extra_sla_violation_pct"]].mean().reindex(ABL)
    x = np.arange(len(a))
    ax.bar(x - .2, a.carbon_saving_pct, .4, label="Carbon saving (%)")
    ax.bar(x + .2, a.extra_sla_violation_pct, .4, label="Extra SLA violations (pp)", color="tab:red")
    ax.set_xticks(x); ax.set_xticklabels([n.replace("- ", "-\n") for n in a.index], fontsize=6.5)
    ax.axhline(0, color="k", lw=.8); ax.legend(); ax.set_title("Ablation: contribution of each component (mean over scenarios)")
    fig.tight_layout(); fig.savefig(pdir / "ablation.png", dpi=160); plt.close(fig)

    # 4. Saving by arrival hour for ASO at the largest scale
    big = runs[runs.scale == runs.scale.max()]
    st = big[big.policy == "Stay (no orchestration)"].set_index(["slack", "mode", "seed", "pod_id"])
    aso = big[big.policy == ASO_MAIN].set_index(["slack", "mode", "seed", "pod_id"]).loc[st.index]
    h = pd.DataFrame({"hour": (st.hour // 2 * 2).values, "stay": st.carbon_g.values, "aso": aso.carbon_g.values})
    h = h.groupby("hour").sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(h.index, 100 * (1 - h.aso / h.stay), width=1.6)
    ax.set_xlabel("Job arrival hour"); ax.set_ylabel("ASO saving vs Stay (%)")
    ax.set_title(f"Time-of-day effect (duration scale x{runs.scale.max():g})"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(pdir / "saving_by_hour.png", dpi=160); plt.close(fig)
