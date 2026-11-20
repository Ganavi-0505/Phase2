"""
Policy benchmark + ablation study.

    python -m src.evaluation.benchmark --csv-path data/borg_traces_data.csv
    python -m src.evaluation.benchmark --synthetic            # smoke test only

Protocol
  * Workloads come from the Borg trace; first 70% (by start time) train the GBR
    runtime predictor, the remaining 30% are scheduled (no train/test leakage).
  * Each job gets a random arrival time in a 24 h day (per seed).
  * Every policy plans with forecasts; the decision is scored on a hidden ground
    truth (true CI incl. weather, true runtime, true SLA outcome).
  * Scenario grid: duration_scale x deadline-slack profile x migration mode.
  * Multiple seeds; savings reported with bootstrap 95% CIs over jobs and a paired
    Wilcoxon test against the no-orchestration baseline and the strongest baseline.
Outputs -> results/benchmark/
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.policies import Ctx, default_policies, ablation_policies
from src.evaluation.world import World, scale_workloads
from src.layer1_infra.trace_loader import TraceLoader
from src.layer4_digital_twin.gbr_predictor import GBRRuntimePredictor
from src.layer4_digital_twin.runtime_model_v2 import RuntimeModelV2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "benchmark"


def run_scenario(base_pods, scale, slack, mode, seed, n_jobs, n_mc, policies, gbr_cache, predictor="v2"):
    pods = scale_workloads(base_pods, scale, slack)
    pods = sorted(pods, key=lambda p: p.start_time_micro)
    cut = int(0.7 * len(pods))
    train, test = pods[:cut], pods[cut:]
    key = (scale, slack)
    if key not in gbr_cache:
        models = {"v1": GBRRuntimePredictor(random_state=42), "v2": RuntimeModelV2(random_state=42)}
        for m in models.values():
            m.train_time_split(train, train_ratio=0.8)
        gbr_cache[key] = models
    gbr = gbr_cache[key][predictor]

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(test), size=min(n_jobs, len(test)), replace=False)
    jobs = [test[i] for i in sorted(idx)]
    arrivals = rng.uniform(0, 86400, size=len(jobs))
    world = World(seed=seed, migration_mode=mode)

    rows = []
    for pod, t in zip(jobs, arrivals):
        mean, std = gbr.predict(pod)
        ctx = Ctx(pod, float(t), mean, std)
        for pol in policies:
            d = pol.decide(world, ctx)
            o = world.realise(pod, world.node_by_id[pod.source_node], world.node_by_id[d.node_id], float(t), d.delay_sec)
            rows.append(dict(policy=pol.name, scale=scale, slack=slack, mode=mode, seed=seed,
                             pod_id=pod.pod_id, hour=float(t) / 3600.0, priority=pod.priority,
                             duration_sec=pod.duration_sec, carbon_g=o.carbon_g, energy_kwh=o.energy_kwh,
                             migrated=o.migrated, delay_sec=o.delay_sec, sla_violated=o.sla_violated,
                             target_region=world.node_by_id[d.node_id].region))
    return rows, gbr.get_metrics()


def bootstrap_saving(carbon, base, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    carbon, base = np.asarray(carbon), np.asarray(base)
    m = len(carbon)
    est = 100.0 * (1 - carbon.sum() / base.sum())
    boots = np.empty(n)
    for i in range(n):
        j = rng.integers(0, m, m)
        boots[i] = 100.0 * (1 - carbon[j].sum() / base[j].sum())
    return est, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarise(df: pd.DataFrame, stay_name="Stay (no orchestration)") -> pd.DataFrame:
    from scipy.stats import wilcoxon
    out = []
    for (scale, slack, mode), g in df.groupby(["scale", "slack", "mode"]):
        key = ["seed", "pod_id"]
        stay = g[g.policy == stay_name].set_index(key)
        for pol, gp in g.groupby("policy"):
            gp = gp.set_index(key).loc[stay.index]
            sav, lo, hi = bootstrap_saving(gp.carbon_g.values, stay.carbon_g.values)
            diff = (stay.carbon_g - gp.carbon_g).values
            try:
                p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            except ValueError:
                p = 1.0
            out.append(dict(scale=scale, slack=slack, mode=mode, policy=pol, jobs=len(gp),
                            carbon_saving_pct=sav, ci_lo=lo, ci_hi=hi, p_value_vs_stay=p,
                            total_carbon_g=gp.carbon_g.sum(),
                            sla_violation_pct=100 * gp.sla_violated.mean(),
                            extra_sla_violation_pct=100 * (gp.sla_violated.mean() - stay.sla_violated.mean()),
                            migration_pct=100 * gp.migrated.mean(),
                            mean_delay_min=gp.delay_sec.mean() / 60.0,
                            energy_kwh=gp.energy_kwh.sum()))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default=str(ROOT / "data" / "borg_traces_data.csv"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--batch-size", type=int, default=2000, help="trace rows to load")
    ap.add_argument("--jobs", type=int, default=150, help="scheduled jobs per seed")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scales", type=float, nargs="+", default=[1, 120, 720])
    ap.add_argument("--slacks", nargs="+", default=["tight", "mixed"])
    ap.add_argument("--modes", nargs="+", default=["live"])
    ap.add_argument("--mc", type=int, default=300)
    ap.add_argument("--predictor", choices=["v1", "v2"], default="v2",
                    help="v1 = original GBR, v2 = log-target model with calibrated relative uncertainty")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    csv = Path(a.csv_path)
    if a.synthetic or not csv.exists():
        from src.evaluation.synthetic_trace import make_synthetic_borg_csv
        csv = out / "synthetic_trace.csv"
        make_synthetic_borg_csv(csv)
        print("[!] USING SYNTHETIC TRACE -- smoke test only, not valid project results.")
    base = TraceLoader(str(csv)).load_trace_batch(limit=a.batch_size, compression_factor=10.0)
    print(f"Loaded {len(base)} workloads from {csv.name}")

    policies = default_policies(a.mc) + [p for p in ablation_policies(a.mc) if p.name != "ASO (full)"]
    rows, gbr_cache, t0 = [], {}, time.time()
    for scale in a.scales:
        for slack in a.slacks:
            for mode in a.modes:
                for seed in range(a.seeds):
                    r, _ = run_scenario(base, scale, slack, mode, seed, a.jobs, a.mc, policies, gbr_cache, a.predictor)
                    rows += r
                print(f"  scale={scale:g} slack={slack} mode={mode} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "benchmark_runs.csv", index=False)
    summ = summarise(df)
    summ.to_csv(out / "summary.csv", index=False)
    pred = {f"scale={k[0]:g}": {n: m.get_metrics() for n, m in g.items()}
            for k, g in gbr_cache.items() if k[1] == a.slacks[0]}
    json.dump(pred, open(out / "predictor_metrics.json", "w"), indent=2)
    print(summ[summ.policy.str.startswith(("Stay", "Greedy", "Spatio", "ASO (proposed)"))]
          .round(3).to_string(index=False))
    try:
        from src.evaluation.plots import make_all_plots
        make_all_plots(out)
    except Exception as e:  # plots are optional
        print("plot generation skipped:", e)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
