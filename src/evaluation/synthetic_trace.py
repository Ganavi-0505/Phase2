"""
Synthetic Borg-like trace generator.

ONLY for smoke tests / demos when data/borg_traces_data.csv is not available.
Produces the columns that TraceLoader reads. Never report numbers from this
generator as results on the real Google Borg trace.
"""
import json
import numpy as np
import pandas as pd


def make_synthetic_borg_csv(path, n=1500, seed=7):
    rng = np.random.default_rng(seed)
    cpus = np.clip(rng.lognormal(-2.5, 0.9, n), 0.005, 0.5)
    mem = np.clip(rng.lognormal(-3.2, 0.9, n), 0.002, 0.4)
    priority = rng.choice([0, 25, 100, 103, 110, 116, 119, 120, 200, 360], n,
                          p=[.12, .08, .12, .12, .14, .06, .06, .18, .08, .04])
    base = np.where(priority >= 120, 300.0, np.where(priority >= 100, 600.0, 50.0))
    long_tail = rng.random(n) < (0.05 + 0.2 * (cpus > np.quantile(cpus, 0.8)))
    raw = np.where(long_tail, rng.choice([900.0, 1800.0, 3600.0], n), base)
    raw = raw * rng.choice([0.1, 1.0, 1.0, 1.0], n)
    start = np.sort(rng.integers(0, 2_000_000_000_000, n))
    end = start + (raw * 1e6).astype(np.int64)
    df = pd.DataFrame({
        "collection_id": 1_000_000 + np.arange(n),
        "instance_index": rng.integers(0, 50, n),
        "machine_id": rng.integers(1, 10_000, n),
        "user": [f"user_{i % 40}" for i in range(n)],
        "priority": priority,
        "resource_request": [json.dumps({"cpus": float(c), "memory": float(m)})
                             for c, m in zip(cpus, mem)],
        "assigned_memory": np.clip(mem * rng.uniform(0.6, 1.4, n), 0.001, None),
        "cycles_per_instruction": np.clip(rng.lognormal(0.3, 0.4, n), 0.5, 8),
        "start_time": start, "end_time": end,
    })
    df.to_csv(path, index=False)
    return path
