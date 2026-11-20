"""
Ground-truth "world" used to *evaluate* scheduling policies.

Key methodological point (new in v2): the scheduler PLANS with a forecast model
(diurnal carbon curve + Monte Carlo uncertainty + GBR runtime estimate), but its
decision is SCORED against an independent ground truth:
  * true carbon intensity = diurnal curve x smooth "weather" deviations the
    planner cannot see,
  * true runtime = the trace duration (not the GBR estimate), with hardware noise,
  * true SLA outcome = realised completion time vs. the deadline.
So prediction errors genuinely hurt a policy instead of being graded by its own model.

Everything that is a model assumption (not measured) is labelled as such.
"""
from dataclasses import dataclass, replace
from typing import Dict, List
import zlib

import numpy as np

from src.config import NETWORK_KWH_PER_GB
from src.layer1_infra.cluster_config import create_default_node_pools, NodeConfig
from src.layer1_infra.kepler_emulator import KeplerEmulator
from src.layer1_infra.trace_loader import WorkloadPod
from src.layer2_telemetry.carbon_feed import CarbonFeed
from src.layer4_digital_twin.clark_model import ClarkMigrationModel

DAY = 86400.0
_GRID = np.linspace(0.0, 1.0, 12)


class _NoNoise:
    """Makes CarbonFeed return its deterministic curve (no random noise)."""
    @staticmethod
    def normal(*_a, **_k):
        return 0.0


def stable_seed(*parts) -> int:
    """Process-independent seed (Python's built-in hash() is randomised per run)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


# ---------------------------------------------------------------- workloads
# Borg 2019 priority tiers -> deadline slack (deadline = slack x duration).
SLACK_PROFILES: Dict[str, Dict[str, float]] = {
    "tight": {"free": 1.5, "batch": 1.5, "mid": 1.5, "prod": 1.5},
    "mixed": {"free": 12.0, "batch": 6.0, "mid": 3.0, "prod": 1.3},
    "relaxed": {"free": 24.0, "batch": 12.0, "mid": 6.0, "prod": 2.0},
}


def priority_tier(priority: int) -> str:
    if priority >= 120:
        return "prod"
    if priority >= 116:
        return "mid"
    if priority >= 100:
        return "batch"
    return "free"


def scale_workloads(pods: List[WorkloadPod], duration_scale: float = 1.0,
                    slack_profile: str = "mixed") -> List[WorkloadPod]:
    """Undo (part of) the timeline compression and assign priority-aware deadlines."""
    prof = SLACK_PROFILES[slack_profile]
    out = []
    for p in pods:
        d = max(1.0, p.duration_sec * duration_scale)
        out.append(replace(p, duration_sec=round(d, 3),
                           sla_deadline_sec=round(d * prof[priority_tier(p.priority)], 3)))
    return out


# -------------------------------------------------------------------- world
@dataclass
class Outcome:
    carbon_g: float
    energy_kwh: float
    finish_sec: float
    migrated: bool
    delay_sec: float
    sla_violated: bool


class World:
    def __init__(self, seed: int = 0, ci_forecast_sigma: float = 0.12,
                 hw_noise_sigma: float = 0.08, migration_mode: str = "live",
                 network_kwh_per_gb: float = NETWORK_KWH_PER_GB):
        assert migration_mode in ("live", "new")
        self.seed = int(seed)
        self.nodes: List[NodeConfig] = create_default_node_pools()
        self.node_by_id = {n.node_id: n for n in self.nodes}
        self.regions = sorted({n.region for n in self.nodes})
        self.rep_node = {r: next(n for n in self.nodes if n.region == r) for r in self.regions}
        self.kepler = KeplerEmulator()
        self.clark = ClarkMigrationModel()
        self.feed = CarbonFeed(seed=seed)
        self.ci_sigma = ci_forecast_sigma
        self.hw_sigma = hw_noise_sigma
        self.migration_mode = migration_mode
        self.net_kwh_per_gb = network_kwh_per_gb
        # smooth "weather" deviations of the TRUE grid (invisible to the planner)
        rng = np.random.default_rng(stable_seed("weather", seed))
        self._weather = {
            r: [(rng.uniform(0.03, 0.07), p, rng.uniform(0, 2 * np.pi))
                for p in (3 * 3600.0, 7 * 3600.0, 19 * 3600.0)]
            for r in self.regions
        }

    # ---- carbon intensity -------------------------------------------------
    def ci_forecast(self, region: str, t) -> np.ndarray:
        """The planner's belief: deterministic diurnal curve (gCO2/kWh).

        Vectorised re-statement of CarbonFeed.get_carbon_intensity with the random
        noise term removed (a unit test checks both agree)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        tod = t % DAY
        if region == "DC_EU":
            ci = np.clip(280.0 - 140.0 * np.sin(np.pi * tod / DAY), 90.0, 450.0)
        elif region == "DC_US":
            ci = np.clip(420.0 + 70.0 * np.cos(2 * np.pi * (tod - 28800.0) / DAY), 250.0, 580.0)
        elif region == "DC_ASIA":
            ci = np.clip(540.0 + 40.0 * np.sin(2 * np.pi * tod / DAY), 380.0, 680.0)
        else:
            ci = np.full_like(tod, 350.0)
        return ci

    def ci_truth(self, region: str, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        base = self.ci_forecast(region, t)
        dev = np.zeros_like(base)
        for amp, period, phase in self._weather[region]:
            dev += amp * np.sin(2 * np.pi * t / period + phase)
        return np.clip(base * (1.0 + dev), 50.0, 700.0)

    def _mean_ci(self, fn, region: str, t0, dur) -> np.ndarray:
        t0 = np.atleast_1d(np.asarray(t0, float))
        dur = np.broadcast_to(np.atleast_1d(np.asarray(dur, float)), t0.shape)
        tt = t0[:, None] + dur[:, None] * _GRID[None, :]
        return fn(region, tt).mean(axis=1)

    def mean_ci_forecast(self, region, t0, dur):
        return self._mean_ci(self.ci_forecast, region, t0, dur)

    def mean_ci_truth(self, region, t0, dur):
        return self._mean_ci(self.ci_truth, region, t0, dur)

    # ---- power / migration -----------------------------------------------
    def power_w(self, pod: WorkloadPod, node: NodeConfig) -> float:
        util = min(1.0, pod.cpu_request / max(node.mips_rating / 1000.0, 1e-3))
        return self.kepler.estimate_pod_power(util, pod.assigned_memory_gb,
                                              node.idle_power_w, node.max_power_w)

    def speed_ratio(self, src: NodeConfig, dst: NodeConfig) -> float:
        return src.mips_rating / dst.mips_rating

    def migration(self, pod: WorkloadPod, src: NodeConfig, dst: NodeConfig, t: float):
        """Returns (duration_sec, carbon_g, energy_kwh). Zero for same node."""
        if src.node_id == dst.node_id:
            return 0.0, 0.0, 0.0
        est = self.clark.estimate_migration_cost(pod, src, dst, 0.0,
                                                 is_new_placement=(self.migration_mode == "new"))
        mem = max(0.05, pod.assigned_memory_gb)
        payload = mem * 1.3 if self.migration_mode == "live" else max(0.01, 0.05 * mem)
        kwh = payload * self.net_kwh_per_gb
        ci = 0.5 * (float(self.ci_truth(src.region, t)[0]) + float(self.ci_truth(dst.region, t)[0]))
        return est.migration_duration_sec, kwh * ci, kwh

    # ---- ground-truth realisation ------------------------------------------
    def _xi(self, pod: WorkloadPod, node: NodeConfig) -> float:
        rng = np.random.default_rng(stable_seed("hw", self.seed, pod.pod_id, node.node_id))
        return float(rng.normal())

    def true_duration(self, pod: WorkloadPod, src: NodeConfig, dst: NodeConfig) -> float:
        return pod.duration_sec * self.speed_ratio(src, dst) * float(np.exp(self.hw_sigma * self._xi(pod, dst)))

    def realise(self, pod: WorkloadPod, src: NodeConfig, dst: NodeConfig,
                t_arrive: float, delay_sec: float = 0.0) -> Outcome:
        mig_t, mig_c, mig_kwh = self.migration(pod, src, dst, t_arrive + delay_sec)
        dur = self.true_duration(pod, src, dst)
        start = t_arrive + delay_sec + mig_t
        ci = float(self.mean_ci_truth(dst.region, [start], [dur])[0])
        kwh = self.power_w(pod, dst) * dur / 3_600_000.0
        finish = delay_sec + mig_t + dur
        return Outcome(carbon_g=kwh * ci + mig_c, energy_kwh=kwh + mig_kwh, finish_sec=finish,
                       migrated=src.node_id != dst.node_id, delay_sec=delay_sec,
                       sla_violated=finish > pod.sla_deadline_sec)
