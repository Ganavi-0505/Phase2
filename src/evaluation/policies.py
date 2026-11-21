"""
Scheduling policies compared in the benchmark.

Baselines (no learning, no uncertainty model)
  StayPolicy            : run where the trace put the job (no orchestration)
  RandomPolicy          : random node
  FastestNodePolicy     : always the highest-MIPS node (performance-first)
  GreedyCIPolicy        : lowest instantaneous carbon intensity, ignores overhead & SLA
  ThresholdPolicy       : migrate if CI gap > 100 gCO2/kWh (common heuristic)
  WaitAwhileGreedy      : spatio-temporal forecast minimiser, ignores migration cost and risk
                          (stands in for existing carbon-aware schedulers)
Proposed
  ASOPolicy             : Monte-Carlo planner = NSG = C_stay - (C_target + C_migration),
                          confidence gate P(NSG>0) >= 0.90, SLA-risk gate, spatial + temporal
                          actions. Flags switch components off for the ablation study.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from src.config import SLA_RISK_LIMIT, NSG_CONFIDENCE
from src.evaluation.world import World, stable_seed
from src.layer1_infra.trace_loader import WorkloadPod

DELAYS_SEC = (0.0, 900.0, 1800.0, 3600.0, 7200.0, 14400.0, 28800.0)


@dataclass
class Ctx:
    pod: WorkloadPod
    t_arrive: float
    pred_mean: float      # GBR runtime estimate on the source node (sec)
    pred_std: float


@dataclass
class Decision:
    node_id: str
    delay_sec: float = 0.0
    note: str = ""


class Policy:
    name = "base"
    def decide(self, world: World, ctx: Ctx) -> Decision:
        raise NotImplementedError


class StayPolicy(Policy):
    name = "Stay (no orchestration)"
    def decide(self, world, ctx):
        return Decision(ctx.pod.source_node)


class RandomPolicy(Policy):
    name = "Random"
    def decide(self, world, ctx):
        rng = np.random.default_rng(stable_seed("rand", world.seed, ctx.pod.pod_id))
        return Decision(world.nodes[int(rng.integers(len(world.nodes)))].node_id)


class FastestNodePolicy(Policy):
    name = "Fastest node"
    def decide(self, world, ctx):
        return Decision(max(world.nodes, key=lambda n: n.mips_rating).node_id)


class GreedyCIPolicy(Policy):
    name = "Greedy lowest-CI"
    def decide(self, world, ctx):
        r = min(world.regions, key=lambda r: world.ci_forecast(r, ctx.t_arrive)[0])
        return Decision(_node_in(world, r, ctx.pod).node_id)


class ThresholdPolicy(Policy):
    name = "CI-gap threshold (100)"
    def __init__(self, gap=100.0):
        self.gap = gap
    def decide(self, world, ctx):
        src = world.node_by_id[ctx.pod.source_node]
        cur = world.ci_forecast(src.region, ctx.t_arrive)[0]
        r = min(world.regions, key=lambda r: world.ci_forecast(r, ctx.t_arrive)[0])
        if cur - world.ci_forecast(r, ctx.t_arrive)[0] > self.gap:
            return Decision(_node_in(world, r, ctx.pod).node_id)
        return Decision(src.node_id)


class WaitAwhileGreedy(Policy):
    name = "Spatio-temporal greedy (no overhead/risk model)"
    def decide(self, world, ctx):
        pod, src = ctx.pod, world.node_by_id[ctx.pod.source_node]
        best, best_c = Decision(src.node_id), np.inf
        for r in world.regions:
            dst = _node_in(world, r, pod)
            dur = ctx.pred_mean * world.speed_ratio(src, dst)
            for d in DELAYS_SEC:
                if d + dur > pod.sla_deadline_sec:   # deterministic feasibility only
                    continue
                ci = world.mean_ci_forecast(r, [ctx.t_arrive + d], [dur])[0]
                c = world.power_w(pod, dst) * dur / 3.6e6 * ci
                if c < best_c:
                    best, best_c = Decision(dst.node_id, d), c
        return best


def _node_in(world: World, region: str, pod: WorkloadPod):
    """Source node if the job already lives in `region`, else the region's first node."""
    src = world.node_by_id[pod.source_node]
    return src if src.region == region else world.rep_node[region]


class ASOPolicy(Policy):
    """Agentic Sustainability Orchestrator planner (vectorised Monte-Carlo digital twin)."""

    def __init__(self, name="ASO (proposed)", n_mc=300, sla_limit=SLA_RISK_LIMIT,
                 confidence=NSG_CONFIDENCE, use_uncertainty=True, use_migration_cost=True,
                 use_sla_gate=True, use_temporal=True, use_spatial=True):
        self.name = name
        self.n_mc, self.sla_limit, self.conf = n_mc, sla_limit, confidence
        self.use_unc, self.use_mig, self.use_sla = use_uncertainty, use_migration_cost, use_sla_gate
        self.use_temporal, self.use_spatial = use_temporal, use_spatial

    def evaluate(self, world: World, ctx: Ctx, dst, delay):
        """MC samples of (NSG, finish time) for one (target, delay) candidate vs. staying now."""
        pod, src = ctx.pod, world.node_by_id[ctx.pod.source_node]
        rng = np.random.default_rng(stable_seed("aso", world.seed, pod.pod_id, dst.node_id, delay))
        n = self.n_mc if self.use_unc else 1
        cv = float(np.clip(ctx.pred_std / max(ctx.pred_mean, 1.0), 0.08, 1.0)) if self.use_unc else 0.0
        z = rng.normal(size=n) if self.use_unc else np.zeros(1)
        dur_stay = ctx.pred_mean * np.exp(cv * z)                     # shared latent runtime (CRN)
        dur_dst = dur_stay * world.speed_ratio(src, dst) * np.exp(
            (world.hw_sigma if self.use_unc else 0.0) * (rng.normal(size=n) if self.use_unc else 0.0))
        mig_t, mig_c, _ = world.migration(pod, src, dst, ctx.t_arrive + delay, truth=False)  # planner uses forecast only
        if not self.use_mig:
            mig_t, mig_c = 0.0, 0.0
        s = world.ci_sigma if self.use_unc else 0.0
        e_stay = np.exp(s * (rng.normal(size=n) if self.use_unc else 0.0))
        e_dst = np.exp(s * (rng.normal(size=n) if self.use_unc else 0.0))
        ci_stay = world.mean_ci_forecast(src.region, np.full(n, ctx.t_arrive), dur_stay) * e_stay
        start = ctx.t_arrive + delay + mig_t
        ci_dst = world.mean_ci_forecast(dst.region, np.full(n, start), dur_dst) * e_dst
        c_stay = world.power_w(pod, src) * dur_stay / 3.6e6 * ci_stay
        c_dst = world.power_w(pod, dst) * dur_dst / 3.6e6 * ci_dst + mig_c
        nsg = c_stay - c_dst
        finish = delay + mig_t + dur_dst
        return nsg, finish

    def decide(self, world, ctx):
        pod, src = ctx.pod, world.node_by_id[ctx.pod.source_node]
        regions = world.regions if self.use_spatial else [src.region]
        delays = DELAYS_SEC if self.use_temporal else (0.0,)
        best, best_gain = Decision(src.node_id, 0.0, "stay: no viable candidate"), 0.0
        for r in regions:
            dst = _node_in(world, r, pod)
            for d in delays:
                if dst.node_id == src.node_id and d == 0.0:
                    continue
                nsg, finish = self.evaluate(world, ctx, dst, d)
                p_pos = float(np.mean(nsg > 0))
                risk = float(np.mean(finish > pod.sla_deadline_sec))
                ok_conf = p_pos >= self.conf if self.use_unc else float(np.mean(nsg)) > 0
                ok_sla = (risk <= self.sla_limit) if self.use_sla else True
                gain = float(np.mean(nsg))
                if ok_conf and ok_sla and gain > best_gain:
                    best_gain = gain
                    best = Decision(dst.node_id, d, f"NSG={gain:.4f}g P={p_pos:.2f} risk={risk:.2f}")
        return best


def default_policies(n_mc=300):
    return [
        StayPolicy(), RandomPolicy(), FastestNodePolicy(), GreedyCIPolicy(), ThresholdPolicy(),
        WaitAwhileGreedy(),
        ASOPolicy(n_mc=n_mc),
    ]


def ablation_policies(n_mc=300):
    return [
        ASOPolicy(name="ASO (full)", n_mc=n_mc),
        ASOPolicy(name="- no migration-cost model", n_mc=n_mc, use_migration_cost=False),
        ASOPolicy(name="- no SLA gate", n_mc=n_mc, use_sla_gate=False),
        ASOPolicy(name="- no uncertainty (point estimate)", n_mc=n_mc, use_uncertainty=False),
        ASOPolicy(name="- no temporal shifting", n_mc=n_mc, use_temporal=False),
        ASOPolicy(name="- no spatial migration", n_mc=n_mc, use_spatial=False),
        ASOPolicy(name="ASO, report gate (SLA<=0.80)", n_mc=n_mc, sla_limit=0.80),
        ASOPolicy(name="ASO confidence 0.60", n_mc=n_mc, confidence=0.60),
        ASOPolicy(name="ASO confidence 0.75", n_mc=n_mc, confidence=0.75),
        ASOPolicy(name="ASO confidence 0.95", n_mc=n_mc, confidence=0.95),
    ]