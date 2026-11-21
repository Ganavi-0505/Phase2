"""
Cluster-less LIVE run: streams scheduling events exactly like the real scheduler,
so the dashboard "Live cluster" page works on any laptop (no Docker/Kind needed).

For every arriving job it records, side by side, what the ASO planner chose and what
two reference policies (Stay, Greedy lowest-CI) would have done, each scored on the
hidden ground truth -> the dashboard can plot the running carbon tally live.

    python -m src.layer7_execution.live_simulator --rate 2 --time-scale 600 --jobs 150
(time-scale = simulated seconds per wall second; 600 => a 24 h day in 2.4 minutes)
"""
import argparse
import heapq
import time
from pathlib import Path

import numpy as np

from src.evaluation.policies import ASOPolicy, GreedyCIPolicy, StayPolicy, Ctx
from src.evaluation.world import World, scale_workloads
from src.layer1_infra.trace_loader import TraceLoader
from src.layer4_digital_twin.runtime_model_v2 import RuntimeModelV2
from src.layer7_execution import live_events as ev

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default=str(ROOT / "data" / "borg_traces_data.csv"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rate", type=float, default=2.0, help="job arrivals per wall second")
    ap.add_argument("--time-scale", type=float, default=600.0, help="simulated seconds per wall second")
    ap.add_argument("--duration-scale", type=float, default=240.0, help="job-length scale (x1 = Borg compressed)")
    ap.add_argument("--jobs", type=int, default=150)
    ap.add_argument("--start-hour", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--mode", choices=["live", "new"], default="live")
    a = ap.parse_args()

    csv = Path(a.csv_path)
    if a.synthetic or not csv.exists():
        from src.evaluation.synthetic_trace import make_synthetic_borg_csv
        csv = ROOT / "results" / "live" / "synthetic_trace.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        make_synthetic_borg_csv(csv)
        print("[!] synthetic trace in use (demo only)")
    base = TraceLoader(str(csv)).load_trace_batch(limit=2000, compression_factor=10.0)
    pods = sorted(scale_workloads(base, a.duration_scale, "mixed"), key=lambda p: p.start_time_micro)
    cut = int(0.7 * len(pods))
    model = RuntimeModelV2(random_state=42)
    model.train_time_split(pods[:cut], train_ratio=0.8)
    test = pods[cut:]

    world = World(seed=a.seed, migration_mode=a.mode)
    aso, greedy, stay = ASOPolicy(), GreedyCIPolicy(), StayPolicy()
    rng = np.random.default_rng(a.seed)
    if not a.no_reset:
        ev.reset()
    ev.emit("start", engine="aso-simulator", time_scale=a.time_scale, duration_scale=a.duration_scale, mode=a.mode)

    t0 = time.time()
    sim0 = a.start_hour * 3600.0
    sim_now = lambda: sim0 + (time.time() - t0) * a.time_scale
    pending = []        # heap of (wall_due, seq, kind, payload)
    seq = 0
    next_arrival = 0.0
    n_sent = 0
    print(f"live simulator: {a.jobs} jobs at {a.rate}/s, 1 wall s = {a.time_scale:g} sim s. Ctrl+C to stop.")
    try:
        while n_sent < a.jobs or pending:
            now = time.time() - t0
            if n_sent < a.jobs and now >= next_arrival:
                pod = test[int(rng.integers(len(test)))]
                pod = pod.__class__(**{**pod.__dict__, "pod_id": f"{pod.pod_id}-{n_sent}"})
                t = sim_now()
                mean, std = model.predict(pod)
                ctx = Ctx(pod, t, mean, std)
                src = world.node_by_id[pod.source_node]
                d = aso.decide(world, ctx)
                dst = world.node_by_id[d.node_id]
                o_aso = world.realise(pod, src, dst, t, d.delay_sec)
                o_stay = world.realise(pod, src, src, t, 0.0)
                g = greedy.decide(world, ctx)
                o_gr = world.realise(pod, src, world.node_by_id[g.node_id], t, 0.0)
                action = ("DELAY" if d.delay_sec > 0 else "MIGRATE" if dst.node_id != src.node_id else "START")
                ev.emit("arrive", pod_id=pod.pod_id, source_node=src.node_id, sim_t=t, priority=pod.priority,
                        duration_s=pod.duration_sec, deadline_s=pod.sla_deadline_sec)
                ev.emit("decision", pod_id=pod.pod_id, action=action, source_node=src.node_id,
                        target_node=dst.node_id, target_region=dst.region, delay_s=d.delay_sec, sim_t=t,
                        rationale=d.note, ci_src=float(world.ci_forecast(src.region, t)[0]),
                        ci_dst=float(world.ci_forecast(dst.region, t)[0]),
                        ci_by_region={r: float(world.ci_forecast(r, t)[0]) for r in world.regions})
                bind_wall = now + d.delay_sec / a.time_scale
                if d.delay_sec > 0:
                    ev.emit("delay", pod_id=pod.pod_id, delay_s=d.delay_sec, sim_t=t)
                seq += 1
                heapq.heappush(pending, (bind_wall, seq, "bind", dict(pod_id=pod.pod_id, node_id=dst.node_id)))
                seq += 1
                heapq.heappush(pending, (bind_wall + (o_aso.finish_sec - d.delay_sec) / a.time_scale, seq, "complete",
                    dict(pod_id=pod.pod_id, node_id=dst.node_id, carbon_g=o_aso.carbon_g, carbon_stay_g=o_stay.carbon_g,
                         carbon_greedy_g=o_gr.carbon_g, energy_kwh=o_aso.energy_kwh, sla_violated=bool(o_aso.sla_violated),
                         greedy_sla_violated=bool(o_gr.sla_violated), migrated=bool(o_aso.migrated),
                         runtime_s=o_aso.finish_sec - d.delay_sec)))
                n_sent += 1
                next_arrival = now + float(rng.exponential(1.0 / a.rate))
            while pending and pending[0][0] <= time.time() - t0:
                _, _, kind, payload = heapq.heappop(pending)
                ev.emit(kind, sim_t=sim_now(), **payload)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    ev.emit("info", message="simulation finished")
    print("done")


if __name__ == "__main__":
    main()