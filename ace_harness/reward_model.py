"""
Adaptive Reward Model — learns from realized portfolio returns to provide
a lightweight numeric signal alongside the LLM-based memory and Debater.

Design:
  - Pure numpy Ridge regression. No new dependencies.
  - Features: per-asset normalized weights + CASH% + HHI + top-1 weight.
  - Activates after MIN_DATA_POINTS months; returns "" before that so the
    existing pipeline is completely unchanged while the model is cold.
  - Purely additive: score() returns a guidance string injected into the
    Solver prompt and Debater screener context — never overrides any
    existing decision, allocation, or harness logic.
  - Persisted as JSON alongside the playbook so it survives crashes and
    accumulates across re-runs on the same tag.

Reward signal: buy-and-hold portfolio return from the executed allocation
weights and the price change from rebalance date to next rebalance date.
This is the cleanest causal link — what return did THIS allocation produce?
"""
import json

import numpy as np


MIN_DATA_POINTS = 4   # months before the model starts scoring
RIDGE_ALPHA = 5.0     # L2 regularization strength — kept high given small n


def compute_portfolio_return(targets_pct, decision_prices, realized_prices):
    """
    Buy-and-hold return (%) of the executed allocation from decision_date
    to the next rebalance date.

    targets_pct     : dict {anon_asset: weight_pct, "CASH": cash_pct}
    decision_prices : dict {anon_asset: price_at_decision}
    realized_prices : dict {anon_asset: price_at_next_rebalance}
    """
    r = 0.0
    norm = sum(v for v in targets_pct.values())
    if norm <= 0:
        return 0.0
    for asset, w_pct in targets_pct.items():
        if asset == "CASH":
            continue
        w = w_pct / norm
        p0 = decision_prices.get(asset, 0.0)
        p1 = realized_prices.get(asset, p0)
        if p0 > 0:
            r += w * (p1 / p0 - 1.0)
    return r * 100.0


class AdaptiveRewardModel:
    """
    Ridge regression reward model mapping allocation features to predicted
    next-month portfolio return (%).

    Features per observation (p = n_assets + 3):
      [w_1, ..., w_n]   per-asset normalized weights  (sums to ≤1)
      CASH              cash fraction
      HHI               Herfindahl-Hirschman Index = sum(w_i^2), a
                        concentration measure — high = concentrated
      Top1Wt            largest single-asset weight
    """

    def __init__(self, asset_names):
        self.asset_names = list(asset_names)
        self._feature_names = self.asset_names + ["CASH", "HHI", "Top1Wt"]
        self._X = []          # list of feature arrays
        self._y = []          # list of realized returns (%)
        # fitted model state
        self._coef = None
        self._intercept = 0.0
        self._mean = None
        self._std = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _featurize(self, targets_pct):
        weights = np.array([targets_pct.get(a, 0.0) / 100.0
                             for a in self.asset_names], dtype=float)
        cash = targets_pct.get("CASH", 0.0) / 100.0
        hhi = float(np.dot(weights, weights))
        top1 = float(weights.max()) if len(weights) else 0.0
        return np.concatenate([weights, [cash, hhi, top1]])

    def _fit(self):
        X = np.array(self._X)
        y = np.array(self._y)
        self._mean = X.mean(axis=0)
        raw_std = X.std(axis=0)
        self._std = np.where(raw_std < 1e-8, 1.0, raw_std)
        Xs = (X - self._mean) / self._std

        # Ridge closed-form with explicit intercept column (not regularized)
        ones = np.ones((len(Xs), 1))
        Xa = np.hstack([ones, Xs])
        reg = RIDGE_ALPHA * np.eye(Xa.shape[1])
        reg[0, 0] = 0.0   # don't penalize intercept
        w = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ y)
        self._intercept = float(w[0])
        self._coef = w[1:]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, targets_pct, realized_return_pct):
        """Add one (allocation, realized_return) observation and refit."""
        self._X.append(self._featurize(targets_pct))
        self._y.append(float(realized_return_pct))
        if len(self._X) >= MIN_DATA_POINTS:
            self._fit()

    def score(self, targets_pct):
        """
        Return a guidance string for the Solver/Debater prompt, or "" if
        the model is not yet warm (< MIN_DATA_POINTS observations).
        The caller injects this string as-is; it deliberately reads as a
        soft sanity check, not an instruction.
        """
        if self._coef is None:
            return ""
        x = self._featurize(targets_pct)
        xs = (x - self._mean) / self._std
        pred = float(self._intercept + self._coef @ xs)

        contribs = [(self._feature_names[i], float(self._coef[i] * xs[i]))
                    for i in range(len(self._coef))]
        contribs.sort(key=lambda t: abs(t[1]), reverse=True)
        top3_str = ", ".join(f"{n}({v:+.1f}pp)" for n, v in contribs[:3])

        n = len(self._X)
        return (
            f"ADAPTIVE REWARD SIGNAL ({n} months of data): "
            f"predicted next-month return for this proposal = {pred:+.1f}%. "
            f"Top allocation drivers: {top3_str}. "
            f"Treat as a soft sanity check on conviction sizing — not a hard target."
        )

    def n_observations(self):
        return len(self._X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        with open(path, "w") as f:
            json.dump({
                "asset_names": self.asset_names,
                "X": [x.tolist() for x in self._X],
                "y": self._y,
            }, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        obj = cls(d["asset_names"])
        obj._X = [np.array(x) for x in d["X"]]
        obj._y = d["y"]
        if len(obj._X) >= MIN_DATA_POINTS:
            obj._fit()
        return obj
