# ASO v2 - what changed and how to run it (Windows PowerShell)

```powershell
pip install -r requirements.txt
python -m pytest tests/test_evaluation.py -q

# 1. main experiment (needs data\borg_traces_data.csv; ~10-20 min on a laptop)
python -m src.evaluation.benchmark --csv-path data/borg_traces_data.csv --jobs 200 --seeds 5 `
    --scales 1 120 720 --slacks tight mixed relaxed --modes live new --mc 300

# 2. break-even analysis
python -m src.evaluation.sensitivity

# 3. dashboard
streamlit run ui/app.py
```
Outputs: results/benchmark/{summary.csv, benchmark_runs.csv, predictor_metrics.json, breakeven.csv, plots/*.png}

## Files changed (existing)
| file | change |
|---|---|
| src/layer4_digital_twin/monte_carlo.py | `hash()` -> `zlib.crc32` (results were not reproducible across runs) |
| src/layer6_decision/policy_env.py | SLA limit / confidence now come from `src/config.py` (0.80 -> 0.10) |
| src/layer5_outputs/nsg_calculator.py | same shared thresholds |
| src/layer5_outputs/visualizations.py | default SLA line 0.80 -> 0.10 |
| requirements.txt | + streamlit, plotly |

## Files added
src/config.py, src/evaluation/{world,policies,benchmark,sensitivity,plots,synthetic_trace}.py,
src/layer4_digital_twin/runtime_model_v2.py, ui/app.py, tests/test_evaluation.py

Set `ASO_SLA_LIMIT=0.80` to reproduce the first report's numbers.
