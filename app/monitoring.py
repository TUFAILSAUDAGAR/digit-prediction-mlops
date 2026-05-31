"""
Prediction drift monitor.

Tracks a rolling window of model predictions and raises an alert when the
predicted class distribution deviates significantly from a reference
distribution (the training-set label distribution for MNIST ≈ uniform).

Two drift signals are provided:
  1. PSI  (Population Stability Index)  — standard industry metric
  2. Chi-squared p-value                — statistical significance test

The monitor is thread-safe and exposed through two FastAPI endpoints
added to main.py:
  GET /monitoring/stats   — current window stats
  GET /monitoring/drift   — drift report (psi, chi2, alert flag)

Usage (standalone):
    from app.monitoring import DriftMonitor
    monitor = DriftMonitor()
    monitor.record(predicted_digit=3)
    report = monitor.drift_report()
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Reference distribution: MNIST digit class frequencies (approx. uniform)
_MNIST_CLASS_FREQ = np.array(
    [0.0987, 0.1121, 0.0993, 0.1024, 0.0974,
     0.0902, 0.0986, 0.1044, 0.0975, 0.0994],
    dtype=float,
)
_REFERENCE_DIST = _MNIST_CLASS_FREQ / _MNIST_CLASS_FREQ.sum()

PSI_ALERT_THRESHOLD = 0.2   # > 0.2 = significant shift
CHI2_ALPHA = 0.05            # p-value significance level
DEFAULT_WINDOW = 1000        # rolling window size


class DriftMonitor:
    """Thread-safe rolling-window prediction drift monitor."""

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW,
        reference_dist: np.ndarray = _REFERENCE_DIST,
        num_classes: int = 10,
    ):
        self.window_size = window_size
        self.reference_dist = reference_dist
        self.num_classes = num_classes
        self._window: deque[int] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._alert_history: list[dict] = []

    def record(self, predicted_digit: int):
        """Record a single prediction."""
        with self._lock:
            self._window.append(int(predicted_digit))

    def _observed_dist(self) -> np.ndarray:
        counts = np.zeros(self.num_classes, dtype=float)
        for p in self._window:
            if 0 <= p < self.num_classes:
                counts[p] += 1
        total = counts.sum()
        if total == 0:
            return np.ones(self.num_classes) / self.num_classes
        return counts / total

    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray, epsilon: float = 1e-6) -> float:
        e = np.clip(expected, epsilon, None)
        a = np.clip(actual, epsilon, None)
        return float(np.sum((a - e) * np.log(a / e)))

    def drift_report(self) -> dict:
        with self._lock:
            n = len(self._window)
            if n == 0:
                return {"status": "no_data", "n": 0}

            observed = self._observed_dist()

        psi = self._psi(self.reference_dist, observed)

        # Chi-squared test
        expected_counts = self.reference_dist * n
        observed_counts = observed * n
        chi2_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

        alert = psi > PSI_ALERT_THRESHOLD or p_value < CHI2_ALPHA

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n": n,
            "window_size": self.window_size,
            "psi": round(psi, 6),
            "psi_threshold": PSI_ALERT_THRESHOLD,
            "chi2_stat": round(float(chi2_stat), 4),
            "chi2_p_value": round(float(p_value), 6),
            "chi2_alpha": CHI2_ALPHA,
            "alert": alert,
            "observed_distribution": {str(i): round(float(observed[i]), 4) for i in range(self.num_classes)},
            "reference_distribution": {str(i): round(float(self.reference_dist[i]), 4) for i in range(self.num_classes)},
        }

        if alert:
            logger.warning(
                "Drift alert! PSI=%.4f (threshold=%.2f) p_value=%.4f (alpha=%.2f)",
                psi, PSI_ALERT_THRESHOLD, p_value, CHI2_ALPHA,
            )
            self._alert_history.append(report)

        return report

    def stats(self) -> dict:
        with self._lock:
            n = len(self._window)
            if n == 0:
                return {"n": 0, "counts": {str(i): 0 for i in range(self.num_classes)}}
            counts: dict[str, int] = {str(i): 0 for i in range(self.num_classes)}
            for p in self._window:
                counts[str(p)] = counts.get(str(p), 0) + 1
        return {"n": n, "window_size": self.window_size, "counts": counts}

    def alert_history(self) -> list[dict]:
        return list(self._alert_history)

    def reset(self):
        with self._lock:
            self._window.clear()
