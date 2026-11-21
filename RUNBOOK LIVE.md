# Live Kubernetes run + dashboard (Windows PowerShell, Docker Desktop running)

```powershell
# 0. one-time
pip install -r requirements.txt
kind create cluster --name aso --config kind-config.yaml
kubectl apply -f kepler-manifest.yaml          # optional: real energy telemetry
kubectl get nodes --show-labels                # expect 9 workers with node-id / region labels

# 1. Terminal A - the scheduler (writes results/live/events.jsonl)
python -m src.layer7_execution.custom_scheduler --engine aso --reset-log --time-scale 600

# 2. Terminal B - submit demo pods that opt in to the scheduler
python -m src.layer7_execution.pod_submission --from-trace data/borg_traces_data.csv --count 20 --interval-sec 4 --balance-regions

# 3. Terminal C - dashboard -> page "Live cluster"
streamlit run ui/app.py
```
No Docker? Use the cluster-less stream (same dashboard page):
`python -m src.layer7_execution.live_simulator --jobs 200 --rate 2 --time-scale 600`

Optimiser study (GA / PSO / NSGA-II): `python -m src.layer3_scheduling.ga_optimizer --csv-path data/borg_traces_data.csv`
Re-run the benchmark after replacing world.py/policies.py: `python -m src.evaluation.benchmark ...`

Notes
* `--engine legacy` runs the original NSGA-II + Digital Twin + PPO chain (needs `train_policy` first).
* Kind has no CRIU, so "MIGRATE" = bind to a different node before start (documented limit).
* Real-cluster carbon on the dashboard is a model estimate; the Stay/Greedy counterfactual line appears only in the simulator.