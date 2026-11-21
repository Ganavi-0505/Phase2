"""
Live event stream + cluster snapshot (Layer 7 -> dashboard).

Every scheduler (the real Kubernetes one in custom_scheduler.py AND the cluster-less
live_simulator.py) appends one JSON object per line to results/live/events.jsonl.
The Streamlit "Live cluster" page tails that file and, if a real cluster is
reachable, also asks the Kubernetes API where pods are running right now.

Event kinds
  start     scheduler/simulator started        (engine, time_scale)
  arrive    a pod entered the pending queue
  decision  planner chose an action            (action, target_node, nsg, p_pos, sla_risk, rationale, ...)
  delay     pod held back until a greener window
  bind      pod bound / started on a node
  complete  pod finished                       (runtime_s, carbon_g, sla_violated, ...)
"""
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT / "results" / "live"
EVENTS = LIVE_DIR / "events.jsonl"
_lock = threading.Lock()


def reset(path: Path = EVENTS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def emit(kind: str, path: Path = EVENTS, **fields: Any) -> Dict[str, Any]:
    rec = {"ts": time.time(), "kind": kind, **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    return rec


def read_events(path: Path = EVENTS, last: Optional[int] = None) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:      # half-written last line
                continue
    return out[-last:] if last else out


def running_pods(events: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """node_id -> pod ids that were bound but have not completed (derived from events)."""
    where: Dict[str, str] = {}
    for e in events:
        if e["kind"] == "bind":
            where[e["pod_id"]] = e["node_id"]
        elif e["kind"] == "complete":
            where.pop(e["pod_id"], None)
    nodes: Dict[str, List[str]] = {}
    for pod, node in where.items():
        nodes.setdefault(node, []).append(pod)
    return nodes


def snapshot_cluster(namespace: str = "sustainability-sim") -> Optional[Dict[str, Any]]:
    """
    Real cluster view via the Kubernetes API, or None if no cluster is reachable.
    Maps real node names to the project's logical node-id/region labels
    (set in kind-config.yaml) and lists pods in the demo namespace.
    """
    try:
        from src.layer1_infra import k8s_deployer
        if not k8s_deployer.is_cluster_reachable(timeout_sec=2.0):
            return None
        from kubernetes import client
        v1 = client.CoreV1Api()
        name_to = {}
        nodes = []
        for n in v1.list_node(_request_timeout=3).items:
            lab = n.metadata.labels or {}
            nid = lab.get("node-id")
            if not nid:
                continue
            region = lab.get("topology.kubernetes.io/region", "?")
            ready = any(c.type == "Ready" and c.status == "True" for c in (n.status.conditions or []))
            name_to[n.metadata.name] = (nid, region)
            nodes.append({"node_id": nid, "region": region, "ready": ready})
        pods = []
        for p in v1.list_namespaced_pod(namespace=namespace, _request_timeout=3).items:
            nid = name_to.get(p.spec.node_name, (None, None))[0]
            pods.append({"name": p.metadata.name,
                         "pod_id": (p.metadata.labels or {}).get("pod-id", p.metadata.name),
                         "node_id": nid, "phase": p.status.phase})
        return {"source": "kubernetes", "nodes": nodes, "pods": pods, "ts": time.time()}
    except Exception:
        return None