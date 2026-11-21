"""
Custom Scheduler Watch Loop (Layer 7: "Scheduler Extender")  -- v2

Watches for Pods with spec.schedulerName == "sustainability-scheduler" that are
Pending, decides where (and when) to run them, and BINDS them through the real
Kubernetes Binding API (k8s_deployer.bind_pod_to_node).

Two decision engines (--engine):
  aso     (default) the SAME Monte-Carlo planner that was benchmarked in
          src/evaluation (spatial migration + temporal shifting, runtime model v2,
          SLA-risk gate). Live results therefore correspond to the evaluated policy.
  legacy  the original Phase-2 chain: NSGA-II shortlist -> Digital Twin -> Layer 5 -> PPO gate.

Every step is appended to results/live/events.jsonl so the dashboard's
"Live cluster" page can show decisions, bindings and completions as they happen.

Time compression: demo pods run for seconds, so --time-scale says how many simulated
seconds pass per wall-clock second (carbon intensity follows the simulated day) and a
pod's duration is re-interpreted accordingly (--duration-scale, default = --time-scale).

    python -m src.layer7_execution.custom_scheduler --engine aso --time-scale 600
"""
import time
from dataclasses import replace
from typing import Dict, List, Optional

from src.layer1_infra import k8s_deployer
from src.layer1_infra.cluster_config import create_default_node_pools, NodeConfig
from src.layer7_execution import live_events as ev
from src.layer7_execution.cooldown_manager import CooldownManager

SCHEDULER_NAME = "sustainability-scheduler"
POLL_INTERVAL_SEC = 3.0
MAX_DELAY_CYCLES = 3


class CustomScheduler:
    def __init__(self, csv_path_for_gbr_training: str, train_batch_size: int = 500, seed: int = 42,
                 engine: str = "aso", time_scale: float = 600.0, duration_scale: Optional[float] = None,
                 start_hour: float = 0.0):
        assert engine in ("aso", "legacy")
        self.engine, self.seed = engine, seed
        self.time_scale = float(time_scale)
        self.duration_scale = float(duration_scale or time_scale)   # a pod running d wall-seconds lasts d*time_scale sim-seconds
        self._t0, self._sim0 = time.time(), start_hour * 3600.0
        self.node_pools: List[NodeConfig] = create_default_node_pools()
        self.cooldown = CooldownManager()
        self._delay_counts: Dict[str, int] = {}
        self._hold_until: Dict[str, float] = {}       # pod_id -> wall time before which we do not re-plan
        self._reported_done = set()
        self._bound: Dict[str, dict] = {}              # k8s pod name -> info for completion events
        if engine == "aso":
            self._init_aso(csv_path_for_gbr_training, train_batch_size)
        else:
            self._init_legacy(csv_path_for_gbr_training, train_batch_size)
        ev.emit("start", engine=engine, time_scale=self.time_scale, duration_scale=self.duration_scale)

    # ------------------------------------------------------------------ setup
    def _init_aso(self, csv, n):
        from src.evaluation.policies import ASOPolicy
        from src.evaluation.world import World, scale_workloads
        from src.layer1_infra.trace_loader import TraceLoader
        from src.layer4_digital_twin.runtime_model_v2 import RuntimeModelV2
        print("[CustomScheduler] Training runtime model v2 ...")
        pods = scale_workloads(TraceLoader(csv).load_trace_batch(limit=n), self.duration_scale, "mixed")
        self.rtm = RuntimeModelV2(random_state=self.seed)
        self.rtm.train_time_split(pods, train_ratio=0.8)
        print(f"[CustomScheduler] runtime model: {self.rtm.get_metrics()}")
        self.world = World(seed=self.seed)
        self.planner = ASOPolicy()
        self.planner_no_delay = ASOPolicy(use_temporal=False)
        self._scale_workloads = scale_workloads

    def _init_legacy(self, csv, n):
        from src.layer1_infra.trace_loader import TraceLoader
        from src.layer2_telemetry.telemetry_source import TelemetrySource
        from src.layer2_telemetry.feature_store import FeatureStore
        from src.layer3_scheduling.nsga2_filter import NSGA2PreFilter
        from src.layer4_digital_twin.gbr_predictor import GBRRuntimePredictor
        from src.layer4_digital_twin.clark_model import ClarkMigrationModel
        from src.layer4_digital_twin.simpy_uncertainty import SimPyCarbonSampler
        from src.layer4_digital_twin.monte_carlo import MonteCarloEngine
        from src.layer6_decision.ppo_agent import PPOAgent, DEFAULT_MODEL_PATH
        from src.layer6_decision.decision_gate import DecisionGate
        from src.layer8_feedback.reward_store import RewardStore
        self.telemetry = TelemetrySource(sim_seed=self.seed)
        self.feature_store = FeatureStore(self.telemetry)
        print("[CustomScheduler] Training GBR predictor for live decisions...")
        pods = TraceLoader(csv).load_trace_batch(limit=n)
        self.gbr = GBRRuntimePredictor(random_state=self.seed)
        self.gbr.train_time_split(pods, train_ratio=0.8)
        self.nsga2 = NSGA2PreFilter(self.telemetry, top_k=3, pop_size=40, n_gen=25, seed=self.seed)
        self.clark = ClarkMigrationModel()
        self.simpy_sampler = SimPyCarbonSampler(self.telemetry._sim_carbon)
        self.mc_engine = MonteCarloEngine(self.gbr, self.clark, self.simpy_sampler, self.telemetry,
                                          num_runs=300, random_seed=self.seed)
        self.ppo_agent = PPOAgent()
        try:
            self.ppo_agent.load(DEFAULT_MODEL_PATH)
            print(f"[CustomScheduler] Loaded trained policy from {DEFAULT_MODEL_PATH}")
        except FileNotFoundError:
            print(f"[CustomScheduler] No trained policy at {DEFAULT_MODEL_PATH} -- "
                  f"run `python -m src.layer6_decision.train_policy` first.")
            raise
        self.gate = DecisionGate(self.ppo_agent)
        self.reward_store = RewardStore()

    # ------------------------------------------------------------------ helpers
    def sim_time(self) -> float:
        return self._sim0 + (time.time() - self._t0) * self.time_scale

    def _node(self, node_id: str) -> NodeConfig:
        return next((n for n in self.node_pools if n.node_id == node_id), self.node_pools[0])

    def _bind(self, k8s_pod, pod, action: str, target: NodeConfig, **extra) -> bool:
        try:
            k8s_deployer.bind_pod_to_node(k8s_pod.metadata.name, target.node_id, k8s_deployer.DEFAULT_NAMESPACE)
        except Exception as exc:
            print(f"[CustomScheduler]   -> Bind API reported: {exc}")
            ev.emit("info", level="error", pod_id=pod.pod_id, message=f"bind failed: {exc}")
            return False
        if action == "MIGRATE":
            self.cooldown.start_cooldown(target.node_id)
        self._delay_counts.pop(pod.pod_id, None)
        self._hold_until.pop(pod.pod_id, None)
        self._bound[k8s_pod.metadata.name] = dict(pod=pod, node_id=target.node_id, wall_bound=time.time(),
                                                   sim_bound=self.sim_time())
        ev.emit("bind", pod_id=pod.pod_id, node_id=target.node_id, region=target.region, action=action,
                sim_t=self.sim_time(), **extra)
        print(f"[CustomScheduler]   -> Bound {k8s_pod.metadata.name} to node {target.node_id}.")
        return True

    # ----------------------------------------------------------- ASO engine path
    def _handle_aso(self, k8s_pod) -> None:
        from src.evaluation.policies import Ctx
        pod = k8s_deployer.workload_pod_from_annotations(k8s_pod)
        if time.time() < self._hold_until.get(pod.pod_id, 0.0):
            return                                              # delayed: wait for the greener window
        src = self._node(pod.source_node)
        spod = self._scale_workloads([pod], self.duration_scale, "mixed")[0]
        t = self.sim_time()
        mean, std = self.rtm.predict(spod)
        ctx = Ctx(spod, t, mean, std)
        first = pod.pod_id not in self._delay_counts
        planner = self.planner if first else self.planner_no_delay   # at most one delay per pod
        d = planner.decide(self.world, ctx)
        dst = self._node(d.node_id)
        action = "DELAY" if d.delay_sec > 0 else ("MIGRATE" if dst.node_id != src.node_id else "START")
        ci = {r: float(self.world.ci_forecast(r, t)[0]) for r in self.world.regions}
        ev.emit("arrive", pod_id=pod.pod_id, source_node=src.node_id, sim_t=t, priority=pod.priority,
                duration_s=spod.duration_sec, deadline_s=spod.sla_deadline_sec) if first else None
        ev.emit("decision", pod_id=pod.pod_id, action=action, source_node=src.node_id, target_node=dst.node_id,
                target_region=dst.region, delay_s=d.delay_sec, sim_t=t, rationale=d.note,
                ci_src=ci[src.region], ci_dst=ci[dst.region], ci_by_region=ci)
        print(f"[CustomScheduler] {pod.pod_id}: {action} -> {dst.node_id} (delay {d.delay_sec / 60:.0f} sim-min). {d.note}")
        if action == "DELAY":
            self._delay_counts[pod.pod_id] = 1
            self._hold_until[pod.pod_id] = time.time() + d.delay_sec / self.time_scale
            ev.emit("delay", pod_id=pod.pod_id, delay_s=d.delay_sec, sim_t=t)
            return
        if action == "MIGRATE" and self.cooldown.is_in_cooldown(dst.node_id):
            print(f"[CustomScheduler]   -> {dst.node_id} in cooldown; starting in place on {src.node_id}.")
            action, dst = "START", src
        self._bind(k8s_pod, pod, action, dst, target_node=dst.node_id)

    # --------------------------------------------------------------- legacy path
    def _handle_legacy(self, k8s_pod) -> None:
        from src.layer5_outputs.nsg_calculator import Layer5OutputCalculator
        pod = k8s_deployer.workload_pod_from_annotations(k8s_pod)
        current_node = self._node(pod.source_node)
        timestamp_sec = self.sim_time() % 86400
        frame = self.feature_store.build_telemetry_frame(pod, self.node_pools, timestamp_sec=timestamp_sec)
        candidates = self.nsga2.quick_rank_single(frame)
        if not candidates:
            print(f"[CustomScheduler] No feasible candidates for {pod.pod_id}; skipping this cycle.")
            return
        results = []
        for cand in candidates:
            mc = self.mc_engine.run_simulation(frame, cand.target_node)
            results.append(Layer5OutputCalculator.evaluate_candidate(
                mc, pod, frame.current_node, cand.target_node, timestamp_sec=timestamp_sec))
        decision = self.gate.decide_best_candidate(results)
        if decision.action == "DELAY":
            n = self._delay_counts.get(pod.pod_id, 0) + 1
            self._delay_counts[pod.pod_id] = n
            if n >= MAX_DELAY_CYCLES:
                best = min(results, key=lambda r: r.sla_risk_score)
                decision.action, decision.target_node_id = "START", best.target_node_id
                decision.rationale += f" Forced start after {n} delayed cycles (lowest SLA risk candidate)."
        chosen = next((r for r in results if r.target_node_id == decision.target_node_id), results[0])
        target = self._node(decision.target_node_id)
        action = decision.action
        ev.emit("decision", pod_id=pod.pod_id, action=action, source_node=current_node.node_id,
                target_node=target.node_id, target_region=target.region, sim_t=timestamp_sec,
                rationale=decision.rationale, nsg_g=chosen.nsg_mean_gco2, p_pos=chosen.p_nsg_positive,
                sla_risk=chosen.sla_risk_score)
        print(f"[CustomScheduler] {pod.pod_id}: {action} -> {target.node_id}. {decision.rationale}")
        if action == "MIGRATE" and self.cooldown.is_in_cooldown(target.node_id):
            print(f"[CustomScheduler]   -> {target.node_id} in cooldown; starting in place on {current_node.node_id}.")
            action, target = "START", current_node
            chosen = next((r for r in results if r.target_node_id == current_node.node_id), chosen)
        if action in ("MIGRATE", "START"):
            if self._bind(k8s_pod, pod, action, target):
                self.reward_store.record_decision(
                    pod.pod_id, action, chosen, target.node_id,
                    context={"cpu_request": pod.cpu_request, "ram_request_gb": pod.ram_request_gb,
                             "assigned_memory_gb": pod.assigned_memory_gb, "duration_sec": pod.duration_sec,
                             "timestamp_sec": timestamp_sec, "nominal_node_id": current_node.node_id})
        else:
            ev.emit("delay", pod_id=pod.pod_id, sim_t=timestamp_sec)

    # ------------------------------------------------------------- completions
    def _report_completions(self) -> None:
        """Emit a 'complete' event (with estimated carbon) for bound pods that finished."""
        for name, info in list(self._bound.items()):
            if name in self._reported_done:
                continue
            runtime = k8s_deployer.get_pod_runtime_sec(name)
            if runtime is None:
                continue
            self._reported_done.add(name)
            pod, node = info["pod"], self._node(info["node_id"])
            carbon = None
            if self.engine == "aso":
                sim_rt = runtime * self.time_scale if runtime else pod.duration_sec * self.duration_scale
                spod = self._scale_workloads([pod], self.duration_scale, "mixed")[0]
                ci = float(self.world.mean_ci_forecast(node.region, [info["sim_bound"]], [sim_rt])[0])
                carbon = self.world.power_w(spod, node) * sim_rt / 3.6e6 * ci   # model estimate, not measured
            ev.emit("complete", pod_id=pod.pod_id, node_id=node.node_id, runtime_s=runtime,
                    carbon_g=carbon, estimated=True, sim_t=self.sim_time())

    def run_forever(self) -> None:
        print(f"[CustomScheduler] engine={self.engine}. Watching for Pending pods with schedulerName='{SCHEDULER_NAME}'...")
        handler = self._handle_aso if self.engine == "aso" else self._handle_legacy
        while True:
            try:
                for k8s_pod in k8s_deployer.list_pending_unscheduled_pods(SCHEDULER_NAME):
                    handler(k8s_pod)
                self._report_completions()
            except Exception as exc:
                print(f"[CustomScheduler] Watch loop error (will retry): {exc}")
                ev.emit("info", level="error", message=str(exc))
            time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run the live custom scheduler loop.")
    p.add_argument("--csv-path", default="data/borg_traces_data.csv")
    p.add_argument("--train-batch-size", type=int, default=2000)
    p.add_argument("--engine", choices=["aso", "legacy"], default="aso")
    p.add_argument("--time-scale", type=float, default=600.0, help="simulated seconds per wall second")
    p.add_argument("--duration-scale", type=float, default=None, help="pod duration multiplier (default: --time-scale)")
    p.add_argument("--start-hour", type=float, default=0.0)
    p.add_argument("--reset-log", action="store_true", help="clear results/live/events.jsonl first")
    a = p.parse_args()
    if a.reset_log:
        ev.reset()
    CustomScheduler(a.csv_path, a.train_batch_size, engine=a.engine, time_scale=a.time_scale,
                    duration_scale=a.duration_scale, start_hour=a.start_hour).run_forever()