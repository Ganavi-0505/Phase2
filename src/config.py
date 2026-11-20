"""
Central decision thresholds shared by the pipeline, the PPO environment and the
evaluation harness.

The first report used SLA_RISK_LIMIT = 0.80, i.e. it accepted a placement even
when the Digital Twin predicted an 80% chance of missing the deadline. That is
not defensible for a production gate, so the default is now 0.10 (a 90% on-time
guarantee). Override with the environment variables below to reproduce the old
numbers (ASO_SLA_LIMIT=0.80) or to run sensitivity studies.
"""
import os

SLA_RISK_LIMIT: float = float(os.environ.get("ASO_SLA_LIMIT", "0.10"))
NSG_CONFIDENCE: float = float(os.environ.get("ASO_NSG_CONFIDENCE", "0.90"))

# Network energy intensity for data movement, kWh per GB. Published estimates
# span roughly 0.001-0.06 kWh/GB depending on scope (WAN only vs end-to-end).
# 0.01 is a mid-range planning value; the sensitivity study sweeps it.
NETWORK_KWH_PER_GB: float = float(os.environ.get("ASO_NET_KWH_PER_GB", "0.01"))
