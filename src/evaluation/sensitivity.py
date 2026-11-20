"""
Break-even / sensitivity analysis -- "when is migration actually worth it?"

For a representative job we compute the exact expected NSG (planner's forecast, no
sampling noise) over a grid of job duration x network energy intensity, for both
migration modes, and report the minimum job duration for which migrating from the
dirtiest region (DC_ASIA) to the cleanest (DC_EU) has NSG > 0.

    python -m src.evaluation.sensitivity
Outputs -> results/benchmark/breakeven.csv, plots/breakeven_heatmap.png
"""
from pathlib import Path
import numpy as np
import pandas as pd

from src.evaluation.world import World
from src.layer1_infra.trace_loader import WorkloadPod

ROOT = Path(__file__).resolve().parents[2]


def make_pod(duration_sec, mem_gb=4.0, cpu=2.0):
    return WorkloadPod("probe", 1, 0, "u", 100, cpu, mem_gb, mem_gb, 1.2, duration_sec, 0, 0,
                       "asia-node-1", "DC_ASIA", duration_sec * 100)


def expected_nsg(duration_sec, net_kwh_per_gb=0.01, mode="live", mem_gb=4.0, t=43200.0,
                 src="DC_ASIA", dst="DC_EU"):
    w = World(0, migration_mode=mode, network_kwh_per_gb=net_kwh_per_gb)
    pod = make_pod(duration_sec, mem_gb)
    s, d = w.rep_node[src], w.rep_node[dst]
    pod = pod.__class__(**{**pod.__dict__, "source_node": s.node_id, "source_region": src})
    stay = w.power_w(pod, s) * duration_sec / 3.6e6 * w.mean_ci_forecast(src, [t], [duration_sec])[0]
    mt, mc, _ = w.migration(pod, s, d, t)
    dur_d = duration_sec * w.speed_ratio(s, d)
    tgt = w.power_w(pod, d) * dur_d / 3.6e6 * w.mean_ci_forecast(dst, [t + mt], [dur_d])[0] + mc
    return stay - tgt


def breakeven_duration(net_kwh_per_gb, mode, mem_gb=4.0):
    lo, hi = 1.0, 7 * 86400.0
    if expected_nsg(hi, net_kwh_per_gb, mode, mem_gb) <= 0:
        return float("inf")
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if expected_nsg(mid, net_kwh_per_gb, mode, mem_gb) > 0:
            hi = mid
        else:
            lo = mid
    return hi


def run(out_dir=None):
    out = Path(out_dir or ROOT / "results" / "benchmark")
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in ("live", "new"):
        for kwh in (0.001, 0.003, 0.01, 0.03, 0.06):
            for mem in (1.0, 4.0, 16.0):
                rows.append(dict(mode=mode, net_kwh_per_gb=kwh, mem_gb=mem,
                                 breakeven_sec=breakeven_duration(kwh, mode, mem)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "breakeven.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for mode, ls in (("live", "-"), ("new", "--")):
            for mem in (1.0, 4.0, 16.0):
                d = df[(df["mode"] == mode) & (df.mem_gb == mem)]
                ax.plot(d.net_kwh_per_gb, d.breakeven_sec / 60, ls, marker="o", label=f"{mode}, {mem:g} GB")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Network energy intensity (kWh/GB)"); ax.set_ylabel("Break-even job duration (min)")
        ax.set_title("Minimum job length for which ASIA->EU migration is net-positive")
        ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "plots" / "breakeven.png", dpi=160); plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)
    return df


if __name__ == "__main__":
    print(run().round(2).to_string(index=False))
