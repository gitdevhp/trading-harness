"""Adaptive Reward Model (AdaReMo): tracks per-asset realized monthly returns
from the running backtest and produces a risk-adjusted sizing signal.

The screener shows current momentum indicators. AdaReMo shows what assets
have ACTUALLY delivered in this specific backtest, risk-adjusted — an asset
returning +3% every month beats one that returned +10% then -8%.

Three layers of signal, ordered by reliability:
  1. Risk-adjusted ranking (Sharpe-like score = w_avg / w_std). Primary
     sort criterion. Assets with consistent, high returns rank above
     high-return but volatile assets.
  2. Max drawdown per asset — shows which assets punished the portfolio
     most severely when they moved against us.
  3. Ridge regression on allocation weights → Sortino-weighted portfolio
     return (downside penalized 2×) — learns which sizing decisions
     historically produced better risk-adjusted outcomes.
"""
import numpy as np


class AdaptiveRewardModel:
    def __init__(self, alpha: float = 1.0, min_samples: int = 2, decay: float = 0.9):
        self.alpha = alpha
        self.min_samples = min_samples
        self.decay = decay
        self._assets = None
        self._X = []                  # allocation vectors (fractions)
        self._y = []                  # Sortino-weighted portfolio returns (%)
        self._asset_returns = {}      # asset -> list[float] of per-period returns
        self._cum_returns = {}        # asset -> current cumulative index (starts at 1.0)
        self._peaks = {}              # asset -> running peak of cum_returns
        self._max_drawdowns = {}      # asset -> worst drawdown seen (negative %)
        self._weights = None
        self._fitted = False

    def _featurize(self, allocations: dict) -> np.ndarray:
        return np.array([allocations.get(a, 0.0) / 100.0 for a in self._assets])

    def record(self, allocations: dict, per_asset_returns: dict):
        """Record one period's allocation + per-asset realized returns."""
        if self._assets is None:
            self._assets = sorted(allocations.keys())

        self._X.append(self._featurize(allocations))

        # Sortino-weighted portfolio return: downside penalized 2× vs upside
        period_rets = []
        for a, ret in per_asset_returns.items():
            w = allocations.get(a, 0.0) / 100.0
            period_rets.append(w * ret)
        raw_return = sum(period_rets)
        downside = sum(r for r in period_rets if r < 0)
        sortino_y = raw_return - abs(downside)  # extra penalty for losses
        self._y.append(float(sortino_y))

        # Per-asset history + drawdown tracking
        for asset, ret in per_asset_returns.items():
            self._asset_returns.setdefault(asset, []).append(float(ret))
            prev_cum = self._cum_returns.get(asset, 1.0)
            new_cum = prev_cum * (1.0 + ret / 100.0)
            self._cum_returns[asset] = new_cum
            new_peak = max(self._peaks.get(asset, 1.0), new_cum)
            self._peaks[asset] = new_peak
            dd = (new_cum - new_peak) / new_peak * 100.0
            self._max_drawdowns[asset] = min(self._max_drawdowns.get(asset, 0.0), dd)

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

    def _weighted_avg(self, rets: list) -> float:
        if not rets:
            return 0.0
        w = [self.decay ** i for i in range(len(rets) - 1, -1, -1)]
        tw = sum(w)
        return sum(wi * r for wi, r in zip(w, rets)) / tw

    def _weighted_std(self, rets: list) -> float:
        if len(rets) < 2:
            return 0.0
        w = [self.decay ** i for i in range(len(rets) - 1, -1, -1)]
        tw = sum(w)
        mu = sum(wi * r for wi, r in zip(w, rets)) / tw
        var = sum(wi * (r - mu) ** 2 for wi, r in zip(w, rets)) / tw
        return var ** 0.5

    def _sharpe_score(self, rets: list) -> float:
        """Exponentially weighted Sharpe-like score (return / volatility).
        Falls back to raw w_avg when std is negligible (consistently good asset)."""
        mu = self._weighted_avg(rets)
        std = self._weighted_std(rets)
        if std < 0.5:
            return mu * 3.0  # tiny vol → treat as 3× the raw return score
        return mu / std

    def signal_text(self, top_k: int = 5) -> str:
        """Return a risk-adjusted sizing-directive block for the Solver prompt.
        Empty string until min_samples periods have accumulated."""
        n = len(self._y)
        if n < self.min_samples:
            return ""

        parts = []

        # --- Layer 1: risk-adjusted ranking (Sharpe-like score) ---
        asset_stats = {}
        for a, rets in self._asset_returns.items():
            if a == "CASH" or not rets:
                continue
            score = self._sharpe_score(rets)
            w_avg = self._weighted_avg(rets)
            w_std = self._weighted_std(rets)
            max_dd = self._max_drawdowns.get(a, 0.0)
            trend = rets[-1] - (rets[-2] if len(rets) >= 2 else rets[-1])
            asset_stats[a] = (score, w_avg, w_std, max_dd, trend)

        if asset_stats:
            ranked = sorted(asset_stats.items(), key=lambda kv: kv[1][0], reverse=True)
            top_hist = [(a, *s) for a, s in ranked[:top_k] if s[1] > 0]   # positive avg
            bot_hist = [(a, *s) for a, s in ranked[-top_k:] if s[1] < 0]  # negative avg

            if top_hist:
                entries = ", ".join(
                    f"{a}(avg {'+' if avg>=0 else ''}{avg:.1f}%±{std:.1f}%, "
                    f"maxDD {dd:.1f}%"
                    f"{', improving' if trend > 0.5 else ', declining' if trend < -0.5 else ''})"
                    for a, score, avg, std, dd, trend in top_hist
                )
                parts.append(
                    f"Best risk-adjusted assets — INCREASE allocation (ranked by return/volatility): {entries}"
                )
            if bot_hist:
                entries = ", ".join(
                    f"{a}(avg {avg:.1f}%±{std:.1f}%, maxDD {dd:.1f}%"
                    f"{', improving' if trend > 0.5 else ', declining' if trend < -0.5 else ''})"
                    for a, score, avg, std, dd, trend in bot_hist
                )
                parts.append(
                    f"Worst risk-adjusted assets — REDUCE or AVOID: {entries}"
                )

            # Highlight any asset with severe drawdown regardless of avg return
            severe_dd = [
                (a, s[3]) for a, s in asset_stats.items()
                if s[3] < -15.0 and a not in {x[0] for x in bot_hist}
            ]
            if severe_dd:
                severe_dd.sort(key=lambda x: x[1])
                parts.append(
                    "Severe drawdown warning — cap position size: {}".format(
                        ", ".join(f"{a}(maxDD {dd:.1f}%)" for a, dd in severe_dd[:top_k])
                    )
                )

        # --- Layer 2: Ridge weights (learned on Sortino-adjusted target) ---
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
                    "Sizing model (Ridge/{} periods, Sortino target) — OVERWEIGHT for better risk-adjusted returns: {}".format(
                        n, ", ".join(a for a, _ in ridge_top)
                    )
                )
            if ridge_bot:
                parts.append(
                    "Sizing model — UNDERWEIGHT (historically dragged risk-adjusted returns): {}".format(
                        ", ".join(a for a, _ in ridge_bot)
                    )
                )

        if not parts:
            return ""

        return (
            f"ADAPTIVE REWARD SIGNAL — {n} realized periods (recent-weighted, risk-adjusted).\n"
            f"PRIMARY GOAL: maximize return per unit of risk/drawdown. "
            f"SIZE UP assets with high return and low volatility; SIZE DOWN volatile or drawdown-prone assets.\n"
            + "\n".join(f"  • {p}" for p in parts)
        )


class SimpleRewardModel:
    """Plain Reward Model (ReMo): tracks per-asset realized returns and produces
    a raw-return sizing signal — no risk-adjustment, no drawdown tracking.

    Two layers of signal:
      1. Exponentially weighted average return per asset — ranks assets by what
         they actually delivered in this backtest.
      2. Ridge regression on allocation weights → raw portfolio return — learns
         which sizing decisions historically produced better outcomes.
    """

    def __init__(self, alpha: float = 1.0, min_samples: int = 2, decay: float = 0.9):
        self.alpha = alpha
        self.min_samples = min_samples
        self.decay = decay
        self._assets = None
        self._X = []
        self._y = []                 # raw weighted portfolio returns (%)
        self._asset_returns = {}     # asset -> list[float] of per-period returns
        self._weights = None
        self._fitted = False

    def _featurize(self, allocations: dict) -> np.ndarray:
        return np.array([allocations.get(a, 0.0) / 100.0 for a in self._assets])

    def record(self, allocations: dict, per_asset_returns: dict):
        """Record one period's allocation + per-asset realized returns."""
        if self._assets is None:
            self._assets = sorted(allocations.keys())

        self._X.append(self._featurize(allocations))

        raw_return = sum(
            allocations.get(a, 0.0) / 100.0 * ret
            for a, ret in per_asset_returns.items()
        )
        self._y.append(float(raw_return))

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

    def _weighted_avg(self, rets: list) -> float:
        if not rets:
            return 0.0
        w = [self.decay ** i for i in range(len(rets) - 1, -1, -1)]
        tw = sum(w)
        return sum(wi * r for wi, r in zip(w, rets)) / tw

    def signal_text(self, top_k: int = 5) -> str:
        """Return a sizing-directive block for the Solver prompt.
        Empty string until min_samples periods have accumulated."""
        n = len(self._y)
        if n < self.min_samples:
            return ""

        parts = []

        # --- Layer 1: raw return ranking ---
        asset_avgs = {
            a: self._weighted_avg(rets)
            for a, rets in self._asset_returns.items()
            if a != "CASH" and rets
        }

        if asset_avgs:
            ranked = sorted(asset_avgs.items(), key=lambda kv: kv[1], reverse=True)
            top_hist = [(a, avg) for a, avg in ranked[:top_k] if avg > 0]
            bot_hist = [(a, avg) for a, avg in ranked[-top_k:] if avg < 0]

            if top_hist:
                entries = ", ".join(
                    f"{a}(avg {'+' if avg >= 0 else ''}{avg:.1f}%)" for a, avg in top_hist
                )
                parts.append(f"Best realized returns — INCREASE allocation: {entries}")
            if bot_hist:
                entries = ", ".join(f"{a}(avg {avg:.1f}%)" for a, avg in bot_hist)
                parts.append(f"Worst realized returns — REDUCE or AVOID: {entries}")

        # --- Layer 2: Ridge weights ---
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
                    "Sizing model (Ridge/{} periods) — OVERWEIGHT for better returns: {}".format(
                        n, ", ".join(a for a, _ in ridge_top)
                    )
                )
            if ridge_bot:
                parts.append(
                    "Sizing model — UNDERWEIGHT (historically dragged returns): {}".format(
                        ", ".join(a for a, _ in ridge_bot)
                    )
                )

        if not parts:
            return ""

        return (
            f"REWARD SIGNAL — {n} realized periods (recent-weighted).\n"
            f"PRIMARY GOAL: maximize realized portfolio return. "
            f"SIZE UP assets with consistently high recent returns; SIZE DOWN assets with negative recent returns.\n"
            + "\n".join(f"  • {p}" for p in parts)
        )
