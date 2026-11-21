"""
Genetic Algorithm (and PSO) energy-carbon placement optimisers  -- Layer 3.

Why this file exists
--------------------
The architecture diagrams (Fig 4.1 / 4.2) name a "GA / PSO energy-carbon ranking"
stage, but the code only contained NSGA-II (pymoo). NSGA-II is itself a
multi-objective genetic algorithm, yet there was no stand-alone GA/PSO and no
evidence that the chosen optimiser is a good one. This module adds:

  * GAPlacementOptimizer  - single-objective GA (tournament selection, uniform
                            crossover, per-gene mutation, elitism)
  * PSOPlacementOptimizer - discrete-rounded particle swarm
  * compare_optimizers    - runs Stay / Greedy / Random / GA / PSO / NSGA-II on the
                            SAME objective (the _BatchAssignmentProblem that NSGA-II
                            already uses) with the SAME evaluation budget.

Run:  python -m src.layer3_scheduling.ga_optimizer --csv-path data/borg_traces_data.csv
Output: results/benchmark/optimizer_comparison.csv and optimizer_convergence.json
"""
import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------- fitness helper
class PlacementObjective:
    """Wraps pymoo's _BatchAssignmentProblem as a scalar, budget-counted function."""

    def __init__(self, problem, pods, w_latency: float = 1.0, cap_penalty: float = 10.0):
        self.problem = problem
        self.n = problem.n_var
        self.hi = problem.xu.astype(int)
        self.w, self.cap = w_latency, cap_penalty
        self.evals = 0
        stay = np.array([self._stay_index(i) for i in range(self.n)], dtype=float)
        F, _ = self.raw(stay, count=False)
        self.ref_c = max(F[0], 1e-9)
        self.ref_l = float(sum(p.duration_sec for p in pods)) + 1.0
        self.ram_total = float(sum(n.total_ram_gb for n in problem.all_nodes))
        self.stay_vec = stay

    def _stay_index(self, i):
        elig = self.problem.eligible_node_lists[i]
        cur = self.problem.current_nodes[i].node_id
        for k, n in enumerate(elig):
            if n.node_id == cur:
                return k
        return 0

    def raw(self, x, count=True):
        out = {}
        self.problem._evaluate(np.asarray(x, dtype=float), out)
        if count:
            self.evals += 1
        return np.asarray(out["F"], float), np.asarray(out["G"], float)

    def __call__(self, x) -> float:
        F, G = self.raw(x)
        viol = float(np.clip(G, 0, None).sum()) / max(self.ram_total, 1e-9)
        return F[0] / self.ref_c + self.w * F[1] / self.ref_l + self.cap * viol

    def summary(self, x):
        F, G = self.raw(x, count=False)
        return dict(carbon_g=float(F[0]), latency_penalty=float(F[1]),
                    capacity_violation=float(np.clip(G, 0, None).sum()))


# --------------------------------------------------------------------------- GA
class GAPlacementOptimizer:
    def __init__(self, pop_size=40, n_gen=25, p_cross=0.9, p_mut=None, elitism=2,
                 tournament=3, seed=42):
        self.pop, self.gen, self.pc, self.pm = pop_size, n_gen, p_cross, p_mut
        self.elite, self.k = elitism, tournament
        self.rng = np.random.default_rng(seed)

    def optimize(self, f: PlacementObjective) -> Dict:
        rng, n, hi = self.rng, f.n, f.hi
        pm = self.pm if self.pm is not None else 1.0 / max(n, 1)
        pop = np.vstack([rng.integers(0, hi + 1) for _ in range(self.pop)])
        pop[0] = f.stay_vec.astype(int)                      # seed with "stay" (safe incumbent)
        fit = np.array([f(x) for x in pop])
        hist = [float(fit.min())]
        for _ in range(self.gen - 1):
            order = np.argsort(fit)
            new = [pop[i].copy() for i in order[: self.elite]]
            while len(new) < self.pop:
                a = self._select(pop, fit)
                b = self._select(pop, fit)
                child = np.where(rng.random(n) < 0.5, a, b) if rng.random() < self.pc else a.copy()
                mut = rng.random(n) < pm
                child[mut] = rng.integers(0, hi[mut] + 1)
                new.append(child)
            pop = np.vstack(new)
            fit = np.array([f(x) for x in pop])
            hist.append(float(fit.min()))
        best = pop[int(np.argmin(fit))]
        return dict(x=best, fitness=float(fit.min()), history=hist)

    def _select(self, pop, fit):
        idx = self.rng.integers(0, len(pop), self.k)
        return pop[idx[np.argmin(fit[idx])]].copy()


# -------------------------------------------------------------------------- PSO
class PSOPlacementOptimizer:
    def __init__(self, n_particles=40, n_iter=25, w=0.7, c1=1.5, c2=1.5, seed=42):
        self.n, self.iters, self.w, self.c1, self.c2 = n_particles, n_iter, w, c1, c2
        self.rng = np.random.default_rng(seed)

    def optimize(self, f: PlacementObjective) -> Dict:
        rng, d, hi = self.rng, f.n, f.hi.astype(float)
        x = rng.random((self.n, d)) * (hi + 1)
        x[0] = f.stay_vec
        v = np.zeros_like(x)
        dec = lambda z: np.clip(np.floor(z), 0, hi).astype(int)
        pbest, pfit = x.copy(), np.array([f(dec(p)) for p in x])
        g = pbest[int(np.argmin(pfit))].copy(); gfit = float(pfit.min())
        hist = [gfit]
        for _ in range(self.iters - 1):
            r1, r2 = rng.random((self.n, d)), rng.random((self.n, d))
            v = self.w * v + self.c1 * r1 * (pbest - x) + self.c2 * r2 * (g - x)
            x = np.clip(x + v, 0, hi + 0.999)
            fit = np.array([f(dec(p)) for p in x])
            better = fit < pfit
            pbest[better], pfit[better] = x[better], fit[better]
            if pfit.min() < gfit:
                gfit = float(pfit.min()); g = pbest[int(np.argmin(pfit))].copy()
            hist.append(gfit)
        return dict(x=dec(g), fitness=gfit, history=hist)


# ------------------------------------------------------------------- comparison
def _random_search(f: PlacementObjective, budget, seed):
    rng = np.random.default_rng(seed)
    best, bf, hist = None, np.inf, []
    for _ in range(budget):
        x = rng.integers(0, f.hi + 1)
        v = f(x)
        if v < bf:
            best, bf = x, v
        hist.append(bf)
    return dict(x=best, fitness=bf, history=hist)


def _greedy(f: PlacementObjective):
    """Per-pod independent best choice (ignores batch capacity interaction)."""
    x = f.stay_vec.copy()
    for i in range(f.n):
        best, bf = x[i], np.inf
        for k in range(int(f.hi[i]) + 1):
            y = x.copy(); y[i] = k
            v = f(y)
            if v < bf:
                best, bf = k, v
        x[i] = best
    return dict(x=x.astype(int), fitness=f(x), history=[])


def _nsga2(problem, f: PlacementObjective, pop, gens, seed):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    res = minimize(problem, NSGA2(pop_size=pop), ("n_gen", gens), seed=seed, verbose=False)
    if res.X is None:
        return None
    X = np.atleast_2d(res.X)
    vals = [f(np.clip(np.rint(x), 0, f.hi).astype(int)) for x in X]
    j = int(np.argmin(vals))
    return dict(x=np.clip(np.rint(X[j]), 0, f.hi).astype(int), fitness=float(vals[j]), history=[],
                pareto_size=int(len(X)))


def compare_optimizers(frames, kepler, pop=40, gens=25, seed=42, w_latency=1.0) -> List[Dict]:
    from src.layer3_scheduling.nsga2_filter import _BatchAssignmentProblem
    from src.layer4_digital_twin.clark_model import ClarkMigrationModel
    pods = [fr.pod for fr in frames]
    all_nodes = frames[0].target_node_states
    elig = []
    for p in pods:
        e = [n for n in all_nodes if n.total_ram_gb >= p.ram_request_gb] or [all_nodes[0]]
        elig.append(e)
    problem = _BatchAssignmentProblem(pods, frames, [fr.current_node for fr in frames], elig, all_nodes,
                                      kepler, ClarkMigrationModel())
    budget = pop * gens
    rows, conv = [], {}

    def run(name, fn):
        f = PlacementObjective(problem, pods, w_latency)
        t0 = time.perf_counter()
        r = fn(f)
        dt = time.perf_counter() - t0
        if r is None:
            return
        s = f.summary(r["x"])
        stay = f.summary(f.stay_vec)
        rows.append(dict(method=name, fitness=r["fitness"], carbon_g=s["carbon_g"],
                         carbon_vs_stay_pct=100 * (1 - s["carbon_g"] / max(stay["carbon_g"], 1e-12)),
                         latency_penalty=s["latency_penalty"], capacity_violation=s["capacity_violation"],
                         evaluations=(budget if name.startswith("NSGA") else f.evals), runtime_s=dt))
        conv[name] = r.get("history", [])

    run("Stay", lambda f: dict(x=f.stay_vec.astype(int), fitness=f(f.stay_vec), history=[]))
    run("Greedy (per-pod)", _greedy)
    run("Random search", lambda f: _random_search(f, budget, seed))
    run("GA", lambda f: GAPlacementOptimizer(pop, gens, seed=seed).optimize(f))
    run("PSO", lambda f: PSOPlacementOptimizer(pop, gens, seed=seed).optimize(f))
    run("NSGA-II (pymoo)", lambda f: _nsga2(problem, f, pop, gens, seed))
    return rows, conv


def main():
    import pandas as pd
    from src.layer1_infra.cluster_config import create_default_node_pools
    from src.layer1_infra.trace_loader import TraceLoader
    from src.layer2_telemetry.feature_store import FeatureStore
    from src.layer2_telemetry.telemetry_source import TelemetrySource

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default=str(ROOT / "data" / "borg_traces_data.csv"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--batch-pods", type=int, default=30)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--out", default=str(ROOT / "results" / "benchmark"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    csv = Path(a.csv_path)
    if a.synthetic or not csv.exists():
        from src.evaluation.synthetic_trace import make_synthetic_borg_csv
        csv = out / "synthetic_trace.csv"; make_synthetic_borg_csv(csv)
        print("[!] USING SYNTHETIC TRACE -- smoke test only.")
    pods = TraceLoader(str(csv)).load_trace_batch(limit=max(2000, a.batches * a.batch_pods), compression_factor=10.0)
    tel = TelemetrySource(sim_seed=42)
    fs, nodes = FeatureStore(tel), create_default_node_pools()
    rng = np.random.default_rng(0)
    all_rows, all_conv = [], {}
    for b in range(a.batches):
        idx = rng.choice(len(pods), a.batch_pods, replace=False)
        t = float(rng.uniform(0, 86400))
        frames = [fs.build_telemetry_frame(pods[i], nodes, timestamp_sec=t) for i in idx]
        rows, conv = compare_optimizers(frames, tel, a.pop, a.gens, seed=b)
        for r in rows:
            r["batch"] = b
        all_rows += rows
        all_conv = conv if b == 0 else all_conv
        print(f"batch {b + 1}/{a.batches} done", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out / "optimizer_comparison.csv", index=False)
    json.dump(all_conv, open(out / "optimizer_convergence.json", "w"))
    summ = df.groupby("method")[["fitness", "carbon_vs_stay_pct", "latency_penalty", "capacity_violation",
                                 "evaluations", "runtime_s"]].mean().sort_values("fitness")
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()