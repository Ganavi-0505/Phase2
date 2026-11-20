"""
Runtime predictor v2 (drop-in for GBRRuntimePredictor: same train_time_split /
predict / get_metrics / get_feature_importance API).

Why it exists: v1 predicts raw seconds and reports ONE absolute residual std for
every job. On the compressed Borg trace (durations cluster at a few values) that
gives R^2 ~ 0.36 and, worse, a huge relative uncertainty for short jobs, so the
Monte-Carlo gate rejects almost every migration. v2 fixes both:

  * models log(duration) (multiplicative errors, robust to the heavy tail),
  * adds log-scale and priority-tier features,
  * estimates RELATIVE uncertainty from a held-out calibration block using a
    conformal-style quantile of |log residual| -> `predict` returns
    (median, median*sigma_log) so downstream code sees a heteroscedastic std,
  * reports interval coverage so the uncertainty is *validated*, not assumed.
"""
from typing import Dict, List, Tuple
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.layer1_infra.trace_loader import WorkloadPod


def _tier(p: int) -> int:
    return 3 if p >= 120 else 2 if p >= 116 else 1 if p >= 100 else 0


class RuntimeModelV2:
    def __init__(self, random_state: int = 42, coverage: float = 0.90):
        self.random_state = random_state
        self.coverage = coverage
        self.model = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.05, max_depth=4, max_iter=250,
            min_samples_leaf=10, l2_regularization=1.0, random_state=random_state)
        self.is_trained = False
        self.sigma_log = 0.5
        self.residual_std = 5.0
        self.metrics: Dict[str, float] = {}
        self.feature_names = ["log_cpu", "log_ram", "log_assigned_mem", "mem_ratio", "cpi",
                              "priority", "tier", "log_cpu_x_mem"]

    @staticmethod
    def _X(pods: List[WorkloadPod]) -> np.ndarray:
        rows = []
        for p in pods:
            cpu, ram, am = max(p.cpu_request, 1e-3), max(p.ram_request_gb, 1e-3), max(p.assigned_memory_gb, 1e-3)
            rows.append([np.log(cpu), np.log(ram), np.log(am), am / ram, p.cpi,
                         p.priority, _tier(p.priority), np.log(cpu) + np.log(am)])
        return np.asarray(rows, float)

    @staticmethod
    def _y(pods):
        return np.log(np.maximum([p.duration_sec for p in pods], 1.0))

    def train_time_split(self, pods: List[WorkloadPod], train_ratio: float = 0.8):
        if len(pods) < 30:
            raise ValueError("Need at least 30 workloads.")
        pods = sorted(pods, key=lambda p: p.start_time_micro)
        n = len(pods)
        n_test = n - int(n * train_ratio)
        n_cal = max(10, int(0.15 * n))
        fit, cal, test = pods[: n - n_test - n_cal], pods[n - n_test - n_cal: n - n_test], pods[n - n_test:]
        self.model.fit(self._X(fit), self._y(fit))
        self.is_trained = True
        # conformal-style calibration of relative error on a block not used for fitting
        res_cal = np.abs(self._y(cal) - self.model.predict(self._X(cal)))
        q = float(np.quantile(res_cal, self.coverage))
        self.sigma_log = max(0.05, q / 1.645)            # normal-equivalent sigma for the target coverage
        # refit on fit+cal for the final model
        both = fit + cal
        self.model.fit(self._X(both), self._y(both))

        y_log, p_log = self._y(test), self.model.predict(self._X(test))
        y, yhat = np.exp(y_log), np.exp(p_log)
        base = float(np.exp(np.mean(self._y(both))))
        mae = float(mean_absolute_error(y, yhat))
        rmse = float(np.sqrt(mean_squared_error(y, yhat)))
        base_rmse = float(np.sqrt(mean_squared_error(y, np.full_like(y, base))))
        lo, hi = p_log - 1.645 * self.sigma_log, p_log + 1.645 * self.sigma_log
        self.residual_std = float(np.std(y - yhat))
        self.metrics = {
            "train_samples": float(len(both)), "test_samples": float(len(test)),
            "mae": mae, "rmse": rmse, "r2": float(r2_score(y, yhat)),
            "r2_log": float(r2_score(y_log, p_log)),
            "smape_percent": float(np.mean(2 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat))) * 100),
            "baseline_rmse": base_rmse, "rmse_improvement_percent": (1 - rmse / base_rmse) * 100,
            "sigma_log": self.sigma_log,
            "interval_coverage_pct": float(np.mean((y_log >= lo) & (y_log <= hi)) * 100),
            "target_coverage_pct": self.coverage * 100,
            "residual_std": self.residual_std,
        }
        return round(rmse, 3), round(self.metrics["r2"], 3)

    def predict(self, pod: WorkloadPod) -> Tuple[float, float]:
        if not self.is_trained:
            return max(2.0, pod.duration_sec), self.residual_std
        mu = float(np.exp(self.model.predict(self._X([pod]))[0]))
        mu = max(2.0, mu)
        return round(mu, 2), round(mu * self.sigma_log, 4)

    def get_metrics(self) -> Dict[str, float]:
        return dict(self.metrics)

    def get_feature_importance(self) -> Dict[str, float]:
        return {}  # use sklearn.inspection.permutation_importance if needed (HGB has no built-in)
