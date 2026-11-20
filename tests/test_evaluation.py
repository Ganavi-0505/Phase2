import numpy as np
import pytest

from src.evaluation.world import World, _NoNoise, stable_seed, scale_workloads
from src.evaluation.policies import ASOPolicy, StayPolicy, GreedyCIPolicy, Ctx
from src.evaluation.sensitivity import expected_nsg, breakeven_duration, make_pod


def test_forecast_matches_carbon_feed_without_noise():
    w = World(0)
    t = np.linspace(0, 2 * 86400, 40)
    for r in w.regions:
        ref = [w.feed.get_carbon_intensity(r, float(x), rng=_NoNoise) for x in t]
        assert np.allclose(ref, w.ci_forecast(r, t))


def test_stable_seed_is_deterministic():
    assert stable_seed("a", 1) == stable_seed("a", 1) != stable_seed("a", 2)


def test_same_node_has_zero_migration_and_zero_nsg():
    w = World(0)
    pod = make_pod(3600)
    src = w.node_by_id[pod.source_node]
    assert w.migration(pod, src, src, 0.0) == (0.0, 0.0, 0.0)
    o = w.realise(pod, src, src, 1000.0)
    assert not o.migrated and o.delay_sec == 0


def test_short_jobs_are_not_migrated_but_greedy_does():
    w = World(0)
    pod = make_pod(20.0)
    ctx = Ctx(pod, 0.0, 20.0, 2.0)
    assert ASOPolicy(n_mc=200).decide(w, ctx).node_id == pod.source_node
    assert GreedyCIPolicy().decide(w, ctx).node_id != pod.source_node


def test_long_jobs_are_migrated_by_aso():
    w = World(0)
    pod = make_pod(8 * 3600.0)
    ctx = Ctx(pod, 0.0, 8 * 3600.0, 0.1 * 8 * 3600)
    d = ASOPolicy(n_mc=200).decide(w, ctx)
    assert d.node_id != pod.source_node or d.delay_sec > 0


def test_planner_is_reproducible():
    w = World(3)
    pod = make_pod(7200.0)
    ctx = Ctx(pod, 5000.0, 7000.0, 700.0)
    a = ASOPolicy(n_mc=100).decide(w, ctx)
    b = ASOPolicy(n_mc=100).decide(World(3), ctx)
    assert (a.node_id, a.delay_sec) == (b.node_id, b.delay_sec)


def test_breakeven_monotone_in_network_energy():
    lo = breakeven_duration(0.001, "live")
    hi = breakeven_duration(0.06, "live")
    assert lo < hi
    assert expected_nsg(86400, 0.01, "live") > 0 > expected_nsg(5, 0.01, "live")


def test_scale_workloads_priority_slack():
    pod = make_pod(100.0)
    import dataclasses
    prod = dataclasses.replace(pod, priority=200)
    free = dataclasses.replace(pod, priority=10)
    p, f = scale_workloads([prod, free], 10, "mixed")
    assert p.sla_deadline_sec < f.sla_deadline_sec
    assert p.duration_sec == 1000.0
