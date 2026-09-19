"""Adaptive Reward Model (AdaReMo): tracks per-asset realized monthly returns
from the running backtest and uses them — via both direct averaging and Ridge
regression — to produce a sizing signal for the Solver.

The screener shows current momentum indicators. AdaReMo shows what assets
have ACTUALLY delivered in this specific backtest over the periods run so far.
That's a different and complementary signal: the screener predicts from current
market state; AdaReMo reports from realized outcomes.

Two layers of signal:
  1. Per-asset realized return history — direct evidence, shown first.
     "ASSET_X returned +9.1%, +12.4% over the last 2 periods."
  2. Ridge regression on allocation weights → portfolio return — tells the
     Solver which assets its sizing decisions have historically correlated
     with better portfolio-level outcomes, controlling for co-movement.
"""
import numpy as np


class AdaptiveRewardModel:
    def __init__(self, alpha: float = 1.0, min_samples: int = 2):
        self.alpha = alpha
        self.min_samples = min_samples
        self._assets = None          # fixed ordering after first record()
        self._X = []                 # allocation vectors (fractions)
        self._y = []                 # realized weighted portfolio returns (%)
        self._asset_returns = {}     # asset -> list[float] of per-period realized returns
        self._weights = None
        self._fitted = False

    def _featurize(self, allocations: dict) -> np.ndarray:
        return np.array([allocations.get(a, 0.0) / 100.0 for a in self._assets])

    def record(self, allocations: dict, per_asset_returns: dict):
        """Record one period's allocation + per-asset realized returns.

        allocations:       {asset: pct_weight, ...} as executed
        per_asset_returns: {asset: realized_return_pct, ...} for the period
        """
        if self._assets is None:
            self._assets = sorted(allocations.keys())

        self._X.append(self._featurize(allocations))

        # Weighted portfolio return for Ridge target
        weighted = sum(
            (allocations.get(a, 0.0) / 100.0) * ret
            for a, ret in per_asset_returns.items()
        )
        self._y.append(float(weighted))

        # Per-asset return history
        for asset, ret in per_asset_returns.items():
            self._asset_returns.setdefault(asset, []).append(float(ret))

        self._fitted = False

    def fit(self):
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

    def signal_text(self, top_k: int = 4) -> str:
        """Return a sizing-directive block for the Solver prompt.
        Empty string until min_samples periods have accumulated."""
        n = len(self._y)
        if n < self.min_samples:
            return ""

        parts = []

        # --- Layer 1: raw per-asset return history ---
        asset_avgs = {
            a: (sum(rets) / len(rets), rets[-1], len(rets))
            for a, rets in self._asset_returns.items()
            if len(rets) >= 1 and a != "CASH"
        }
        if asset_avgs:
            ranked = sorted(asset_avgs.items(), key=lambda kv: kv[1][0], reverse=True)
            top_hist = [(a, avg, last, cnt) for a, (avg, last, cnt) in ranked[:top_k] if avg > 0.5]
            bot_hist = [(a, avg, last, cnt) for a, (avg, last, cnt) in ranked[-top_k:] if avg < -0.5]
            if top_hist:
                entries = ", ".join(
                    f"{a}(avg +{avg:.1f}%, last {'+' if last>=0 else ''}{last:.1f}%, n={cnt})"
                    for a, avg, last, cnt in top_hist
                )
                parts.append(f"Realized outperformers — INCREASE allocation: {entries}")
            if bot_hist:
                entries = ", ".join(
                    f"{a}(avg {avg:.1f}%, last {'+' if last>=0 else ''}{last:.1f}%, n={cnt})"
                    for a, avg, last, cnt in bot_hist
                )
                parts.append(f"Realized underperformers — REDUCE or AVOID: {entries}")

        # --- Layer 2: Ridge regression weights ---
        if self._fitted and self._weights is not None:
            scores = {
                a: float(self._weights[i])
                for i, a in enumerate(self._assets)
                if a != "CASH"
            }
            ranked_r = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            ridge_top = [(a, v) for a, v in ranked_r[:top_k] if v > 0]
            ridge_bot = [(a, v) for a, v in ranked_r[-top_k:] if v < 0]
            if ridge_top:
                parts.append(
                    "Sizing model (Ridge, {} periods) — higher weight historically correlated with better returns: {}".format(
                        n, ", ".join(f"{a}(+{v:.2f}%)" for a, v in ridge_top)
                    )
                )
            if ridge_bot:
                parts.append(
                    "Sizing model — lower weight historically correlated with better returns: {}".format(
                        ", ".join(f"{a}({v:.2f}%)" for a, v in ridge_bot)
                    )
                )

        if not parts:
            return ""

        return (
            f"ADAPTIVE REWARD SIGNAL — {n} realized periods from this backtest.\n"
            f"Use this evidence to SIZE positions: increase weight on outperformers, "
            f"reduce on underperformers, subject to screener confirmation.\n"
            + "\n".join(f"  • {p}" for p in parts)
        )
