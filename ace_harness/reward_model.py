"""Adaptive Reward Model (AdaReMo): Ridge regression that maps historical
allocation weights → realized weighted portfolio return. Purely additive
signal — injected into the Solver's prompt as a weak prior alongside the
experience memory, but never overrides the allocation or the risk harness.

Records one (allocation_vector, portfolio_return_pct) pair per period.
Needs at least min_samples periods before it produces any signal, so the
first few months always run signal-free, then the model activates.
"""
import numpy as np


class AdaptiveRewardModel:
    def __init__(self, alpha: float = 1.0, min_samples: int = 3):
        self.alpha = alpha
        self.min_samples = min_samples
        self._assets = None   # fixed ordering after first record()
        self._X = []          # allocation vectors (fractions, sum <= 1)
        self._y = []          # realized weighted portfolio returns (%)
        self._weights = None
        self._fitted = False

    def _featurize(self, allocations: dict) -> np.ndarray:
        return np.array([allocations.get(a, 0.0) / 100.0 for a in self._assets])

    def record(self, allocations: dict, portfolio_return_pct: float):
        """Record one allocation → realized return pair (call after each period)."""
        if self._assets is None:
            self._assets = sorted(allocations.keys())
        self._X.append(self._featurize(allocations))
        self._y.append(float(portfolio_return_pct))
        self._fitted = False

    def fit(self):
        """Fit Ridge regression on accumulated data. No-op if too few samples."""
        if len(self._X) < self.min_samples or self._assets is None:
            return
        X = np.array(self._X)
        y = np.array(self._y)
        n = X.shape[1]
        try:
            self._weights = np.linalg.solve(X.T @ X + self.alpha * np.eye(n), X.T @ y)
            self._fitted = True
        except np.linalg.LinAlgError:
            pass

    def signal_text(self, top_k: int = 3) -> str:
        """Return a short text block for the Solver prompt. Empty string
        until enough data has accumulated to be meaningful."""
        if not self._fitted or self._weights is None or self._assets is None:
            return ""
        scores = {
            a: float(self._weights[i])
            for i, a in enumerate(self._assets)
            if a != "CASH"
        }
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top = [(a, v) for a, v in ranked[:top_k] if v > 0]
        bottom = [(a, v) for a, v in ranked[-top_k:] if v < 0]
        parts = []
        if top:
            parts.append("Predicted overweight: " + ", ".join(f"{a}(+{v:.2f}%)" for a, v in top))
        if bottom:
            parts.append("Predicted underweight: " + ", ".join(f"{a}({v:.2f}%)" for a, v in bottom))
        if not parts:
            return ""
        n = len(self._X)
        return (
            f"ADAPTIVE REWARD SIGNAL ({n} periods of Ridge regression on allocation-return pairs — "
            f"treat as a weak prior, not a directive):\n" + "\n".join(parts)
        )
