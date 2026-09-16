"""Optional risk-management layers — post-processing filters that sit on
top of whatever a decision_fn produces. Deliberately NOT one of the
self-improvement structures; independent axis, can sit on top of any of
them (or none). Create one fresh instance per backtest run: both classes
are stateful (trailing peaks) and that state must not leak between runs.

Two interchangeable implementations, same interface
(get_params/get_param_bounds/update_params/save_params/apply):

  GPTInstitutionalRiskHarness — conviction overlay -> vol targeting ->
      breadth overlay -> conditional stops -> portfolio drawdown guard.
      Matches your yesharnessgpt.py script.
  SimpleTrailingRiskHarness — per-asset trailing-peak stop-loss, SMA
      de-risking, negative-momentum de-risking. Matches your
      yesharness.py script — deliberately simpler, fewer knobs.

Each class exposes its tunable constants as a `params` dict rather than
hardcoding them, with a class-level PARAM_BOUNDS — so Experience Memory
can tune them over a run (see agents.RiskTuner + harnesses.make_inter_task
/ make_dual_timescale) without ever being able to exceed a safe range.
"""
import json
import numpy as np
import pandas as pd


class GPTInstitutionalRiskHarness:
    DEFAULT_PARAMS = {
        "target_vol": 0.18,              # vol-targeting scales exposure down toward this when max_vol is breached
        "max_vol": 0.215,                # portfolio vol above which scaling kicks in
        "stop_vol_multiplier": 1.8,      # per-asset trailing stop = clip(multiplier * 20d vol, 0.10, 0.25)
        "breadth_min_multiplier": 0.25,  # exposure floor when <20% of the universe is above its 200d SMA
    }
    PARAM_BOUNDS = {
        "target_vol": (0.08, 0.30),
        "max_vol": (0.12, 0.35),
        "stop_vol_multiplier": (1.0, 3.0),
        "breadth_min_multiplier": (0.10, 0.50),
    }

    def __init__(self, universe, params: dict = None):
        self.universe = universe
        self.params = dict(self.DEFAULT_PARAMS)
        if params:
            self.params.update({k: v for k, v in params.items() if k in self.DEFAULT_PARAMS})
        self.asset_peaks = {}
        self.portfolio_peak_value = 0.0

    def get_params(self) -> dict:
        return dict(self.params)

    def get_param_bounds(self) -> dict:
        return dict(self.PARAM_BOUNDS)

    def update_params(self, deltas: dict):
        """Apply small deltas proposed by RiskTuner, hard-clamped to
        PARAM_BOUNDS regardless of what was proposed. Unknown keys or
        non-numeric deltas are silently skipped."""
        for key, delta in (deltas or {}).items():
            if key not in self.params:
                continue
            try:
                delta = float(delta)
            except (TypeError, ValueError):
                continue
            lo, hi = self.PARAM_BOUNDS[key]
            self.params[key] = float(np.clip(self.params[key] + delta, lo, hi))

    def save_params(self, path: str):
        with open(path, "w") as f:
            json.dump(self.params, f, indent=2)

    def _portfolio_vol(self, weights, returns_df):
        if len(returns_df) < 10 or weights.sum() == 0:
            return 0.15
        cov = returns_df.cov().values * 252.0
        cov = 0.75 * cov + 0.25 * np.diag(np.diag(cov))
        var = float(weights.T @ cov @ weights)
        return np.sqrt(max(var, 1e-8))

    def _conviction_overlay(self, raw_weights, current_date):
        u = self.universe
        adjusted = {}
        for asset in u.anon_universe:
            w = raw_weights.get(asset, 0.0)
            if w == 0:
                adjusted[asset] = 0.0
                continue
            closes = u.data_cache[asset].loc[:current_date]["Close"]
            cp = float(closes.iloc[-1])
            mom20 = (cp - float(closes.iloc[-20])) / float(closes.iloc[-20]) if len(closes) >= 20 else 0.0
            mom60 = (cp - float(closes.iloc[-60])) / float(closes.iloc[-60]) if len(closes) >= 60 else 0.0
            mom120 = (cp - float(closes.iloc[-120])) / float(closes.iloc[-120]) if len(closes) >= 120 else 0.0
            composite_mom = 0.5 * mom20 + 0.3 * mom60 + 0.2 * mom120
            sma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else cp
            sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else cp
            trend_signal = (cp / sma50 - 1.0) + (sma50 / sma200 - 1.0)
            score = np.clip(0.6 * composite_mom + 0.4 * trend_signal, -0.5, 0.5)
            adjusted[asset] = w * (1.0 + score)
        return adjusted

    def _apply_risk_target(self, weights_dict, current_date):
        u = self.universe
        w_vec = np.array([weights_dict.get(a, 0.0) for a in u.anon_universe])
        if w_vec.sum() <= 0:
            return weights_dict
        ret_dict = {a: u.data_cache[a].loc[:current_date]["Close"].pct_change().dropna() for a in u.anon_universe}
        ret_df = pd.DataFrame(ret_dict).tail(126).fillna(0.0)
        port_vol = self._portfolio_vol(w_vec / 100.0, ret_df)
        if port_vol > self.params["max_vol"]:
            scale = self.params["target_vol"] / port_vol
            w_vec = w_vec * scale
        return {a: float(w_vec[i]) for i, a in enumerate(u.anon_universe)}

    def _apply_breadth_overlay(self, weights_dict, current_date):
        u = self.universe
        above_200 = 0
        for asset in u.anon_universe:
            closes = u.data_cache[asset].loc[:current_date]["Close"]
            cp = float(closes.iloc[-1])
            sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else cp
            if cp >= sma200:
                above_200 += 1
        ratio = above_200 / len(u.anon_universe)
        multiplier = (
            1.0 if ratio >= 0.6 else
            0.80 if ratio >= 0.4 else
            0.50 if ratio >= 0.2 else
            self.params["breadth_min_multiplier"]
        )
        return {a: w * multiplier for a, w in weights_dict.items()}

    def _apply_conditional_stops(self, weights_dict, current_date):
        u = self.universe
        out = {}
        for asset in u.anon_universe:
            w = weights_dict.get(asset, 0.0)
            closes = u.data_cache[asset].loc[:current_date]["Close"]
            cp = float(closes.iloc[-1])
            self.asset_peaks[asset] = max(self.asset_peaks.get(asset, cp), cp)
            peak = self.asset_peaks[asset]
            drawdown = (peak - cp) / peak if peak > 0 else 0.0
            vol20 = float(closes.tail(20).pct_change().std() * np.sqrt(252)) if len(closes) >= 20 else 0.15
            dynamic_stop = np.clip(self.params["stop_vol_multiplier"] * vol20, 0.10, 0.25)
            sma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else cp
            sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else cp
            tech_breakdown = cp < sma50 and sma50 < sma200
            if drawdown >= dynamic_stop or tech_breakdown:
                w = 0.0
            elif drawdown >= (0.6 * dynamic_stop):
                w *= 0.5
            out[asset] = w
        return out

    def _apply_portfolio_drawdown_guard(self, weights_dict, current_portfolio_value):
        self.portfolio_peak_value = max(self.portfolio_peak_value, current_portfolio_value)
        if self.portfolio_peak_value <= 0:
            return weights_dict
        dd = (self.portfolio_peak_value - current_portfolio_value) / self.portfolio_peak_value
        if dd >= 0.30:
            scale = 0.65
        elif dd >= 0.20:
            scale = 0.80
        elif dd >= 0.10:
            scale = 0.90
        else:
            scale = 1.0
        return {a: w * scale for a, w in weights_dict.items()}

    def apply(self, raw_allocs: dict, current_date: str, portfolio_value: float = None) -> dict:
        w = self._conviction_overlay(raw_allocs, current_date)
        w = self._apply_risk_target(w, current_date)
        w = self._apply_breadth_overlay(w, current_date)
        w = self._apply_conditional_stops(w, current_date)
        w = self._apply_portfolio_drawdown_guard(w, portfolio_value or 0.0)

        total_equity = sum(w.values())
        cash = max(0.0, 100.0 - total_equity)
        if total_equity > 100.0:
            w = {a: (v / total_equity) * 100.0 for a, v in w.items()}
            cash = 0.0
        w["CASH"] = round(cash, 2)

        tot = sum(w.values())
        return {k: round(v / tot * 100.0, 2) for k, v in w.items()} if tot > 0 else w


class SimpleTrailingRiskHarness:
    """Matches apply_risk_harness() from your yesharness.py script:
    per-asset trailing-peak drawdown stop (full exit), price-below-200d-
    SMA de-risking (halve), negative-120d-momentum de-risking (cut 25%).
    Deliberately simpler than GPTInstitutionalRiskHarness — three rules,
    three tunable knobs, no vol-targeting or breadth overlay.
    """
    DEFAULT_PARAMS = {
        "stop_drawdown": 0.15,   # full exit once an asset is down this much from its trailing peak
        "sma_cut": 0.50,         # exposure multiplier when price < 200d SMA
        "momentum_cut": 0.25,    # fraction cut when 120d momentum is negative (e.g. 0.25 -> keep 75%)
    }
    PARAM_BOUNDS = {
        "stop_drawdown": (0.05, 0.35),
        "sma_cut": (0.20, 0.80),
        "momentum_cut": (0.05, 0.50),
    }

    def __init__(self, universe, params: dict = None):
        self.universe = universe
        self.params = dict(self.DEFAULT_PARAMS)
        if params:
            self.params.update({k: v for k, v in params.items() if k in self.DEFAULT_PARAMS})
        self.position_peaks = {a: 0.0 for a in universe.anon_universe}

    def get_params(self) -> dict:
        return dict(self.params)

    def get_param_bounds(self) -> dict:
        return dict(self.PARAM_BOUNDS)

    def update_params(self, deltas: dict):
        for key, delta in (deltas or {}).items():
            if key not in self.params:
                continue
            try:
                delta = float(delta)
            except (TypeError, ValueError):
                continue
            lo, hi = self.PARAM_BOUNDS[key]
            self.params[key] = float(np.clip(self.params[key] + delta, lo, hi))

    def save_params(self, path: str):
        with open(path, "w") as f:
            json.dump(self.params, f, indent=2)

    def apply(self, raw_allocs: dict, current_date: str, portfolio_value: float = None) -> dict:
        u = self.universe
        harnessed = dict(raw_allocs)
        cash_reallocated = 0.0

        for asset in u.anon_universe:
            if asset not in harnessed:
                continue
            closes = u.data_cache[asset].loc[:current_date]["Close"]
            cp = float(closes.iloc[-1])
            self.position_peaks[asset] = max(self.position_peaks.get(asset, cp), cp)
            sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else cp
            mom120 = ((cp - float(closes.iloc[-120])) / float(closes.iloc[-120]) * 100.0) if len(closes) >= 120 else 0.0
            peak = self.position_peaks[asset]
            drawdown = (peak - cp) / peak if peak > 0 else 0.0

            w = harnessed[asset]
            if drawdown >= self.params["stop_drawdown"]:
                cash_reallocated += w
                w = 0.0
            elif cp < sma200:
                cash_reallocated += w * self.params["sma_cut"]
                w *= self.params["sma_cut"]
            elif mom120 < 0:
                cash_reallocated += w * self.params["momentum_cut"]
                w *= (1.0 - self.params["momentum_cut"])

            harnessed[asset] = round(w, 2)

        harnessed["CASH"] = round(harnessed.get("CASH", 0.0) + cash_reallocated, 2)
        total = sum(harnessed.values())
        return {k: round(v / total * 100.0, 2) for k, v in harnessed.items()} if total > 0 else harnessed


class SimpleMomentumHarness:
    """Matches apply_simple_harness() from your "fair portbench" script:
    just two rules — price < 200d SMA cuts exposure (elif) negative
    120d momentum cuts further. No trailing-peak stop-loss at all — the
    simplest of the three risk harnesses in this package.
    """
    DEFAULT_PARAMS = {
        "sma_cut": 0.50,       # exposure multiplier when price < 200d SMA
        "momentum_cut": 0.25,  # fraction cut when 120d momentum is negative
    }
    PARAM_BOUNDS = {
        "sma_cut": (0.20, 0.80),
        "momentum_cut": (0.05, 0.50),
    }

    def __init__(self, universe, params: dict = None):
        self.universe = universe
        self.params = dict(self.DEFAULT_PARAMS)
        if params:
            self.params.update({k: v for k, v in params.items() if k in self.DEFAULT_PARAMS})

    def get_params(self) -> dict:
        return dict(self.params)

    def get_param_bounds(self) -> dict:
        return dict(self.PARAM_BOUNDS)

    def update_params(self, deltas: dict):
        for key, delta in (deltas or {}).items():
            if key not in self.params:
                continue
            try:
                delta = float(delta)
            except (TypeError, ValueError):
                continue
            lo, hi = self.PARAM_BOUNDS[key]
            self.params[key] = float(np.clip(self.params[key] + delta, lo, hi))

    def save_params(self, path: str):
        with open(path, "w") as f:
            json.dump(self.params, f, indent=2)

    def apply(self, raw_allocs: dict, current_date: str, portfolio_value: float = None) -> dict:
        u = self.universe
        out = {a: float(raw_allocs.get(a, 0.0)) for a in u.anon_universe}
        cash = float(raw_allocs.get("CASH", 0.0))

        for asset in u.anon_universe:
            closes = u.data_cache[asset].loc[:current_date]["Close"]
            cp = float(closes.iloc[-1])
            sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else cp
            mom120 = (cp / float(closes.iloc[-120]) - 1.0) if len(closes) >= 120 else 0.0

            if cp < sma200:
                cash += out[asset] * self.params["sma_cut"]
                out[asset] *= self.params["sma_cut"]
            elif mom120 < 0:
                cash += out[asset] * self.params["momentum_cut"]
                out[asset] *= (1.0 - self.params["momentum_cut"])

        out["CASH"] = cash
        total = sum(out.values())
        return {k: round(v / total * 100.0, 2) for k, v in out.items()} if total > 0 else out


class ConvictionHarness:
    """Matches apply_institutional_harness() from your latest script —
    a LIGHTER institutional harness than GPTInstitutionalRiskHarness:
    conviction overlay (momentum/trend composite score) -> portfolio
    vol-targeting -> market-breadth multiplier -> cash fill. Deliberately
    has NO per-asset trailing stop and NO portfolio drawdown guard
    (GPTInstitutionalRiskHarness has both of those; this one doesn't —
    those two steps were dropped in this version of your script).
    """
    DEFAULT_PARAMS = {
        "target_vol": 0.18,              # vol-targeting scales exposure down toward this when max_vol is breached
        "max_vol": 0.215,                # portfolio vol above which scaling kicks in
        "breadth_min_multiplier": 0.25,  # exposure floor when <20% of the universe is above its 200d SMA
    }
    PARAM_BOUNDS = {
        "target_vol": (0.08, 0.30),
        "max_vol": (0.12, 0.35),
        "breadth_min_multiplier": (0.10, 0.50),
    }

    def __init__(self, universe, params: dict = None):
        self.universe = universe
        self.params = dict(self.DEFAULT_PARAMS)
        if params:
            self.params.update({k: v for k, v in params.items() if k in self.DEFAULT_PARAMS})

    def get_params(self) -> dict:
        return dict(self.params)

    def get_param_bounds(self) -> dict:
        return dict(self.PARAM_BOUNDS)

    def update_params(self, deltas: dict):
        for key, delta in (deltas or {}).items():
            if key not in self.params:
                continue
            try:
                delta = float(delta)
            except (TypeError, ValueError):
                continue
            lo, hi = self.PARAM_BOUNDS[key]
            self.params[key] = float(np.clip(self.params[key] + delta, lo, hi))

    def save_params(self, path: str):
        with open(path, "w") as f:
            json.dump(self.params, f, indent=2)

    def _portfolio_vol(self, weights, returns_df):
        if len(returns_df) < 10 or weights.sum() <= 0:
            return 0.15
        cov = returns_df.cov().values * 252.0
        cov = 0.75 * cov + 0.25 * np.diag(np.diag(cov))
        return float(np.sqrt(max(float(weights.T @ cov @ weights), 1e-8)))

    def apply(self, raw_allocs: dict, current_date: str, portfolio_value: float = None) -> dict:
        u = self.universe
        adjusted = {a: float(raw_allocs.get(a, 0.0)) for a in u.anon_universe}

        for asset in u.anon_universe:
            if adjusted[asset] <= 0:
                continue
            closes = u.data_cache[asset].loc[:current_date]["Close"]
            cp = float(closes.iloc[-1])
            m20 = cp / float(closes.iloc[-20]) - 1.0 if len(closes) >= 20 else 0.0
            m60 = cp / float(closes.iloc[-60]) - 1.0 if len(closes) >= 60 else 0.0
            m120 = cp / float(closes.iloc[-120]) - 1.0 if len(closes) >= 120 else 0.0
            composite = 0.5 * m20 + 0.3 * m60 + 0.2 * m120
            sma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else cp
            sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else cp
            trend = (cp / sma50 - 1.0) + (sma50 / sma200 - 1.0)
            score = float(np.clip(0.6 * composite + 0.4 * trend, -0.5, 0.5))
            adjusted[asset] *= (1.0 + score)

        vec = np.array([adjusted[a] for a in u.anon_universe], dtype=float)
        if vec.sum() > 0:
            returns = {a: u.data_cache[a].loc[:current_date]["Close"].pct_change().dropna() for a in u.anon_universe}
            ret_df = pd.DataFrame(returns).tail(126).fillna(0.0)
            vol = self._portfolio_vol(vec / 100.0, ret_df)
            if vol > self.params["max_vol"]:
                vec = vec * (self.params["target_vol"] / vol)
            adjusted = {a: float(vec[i]) for i, a in enumerate(u.anon_universe)}

        above_200 = 0
        for asset in u.anon_universe:
            closes = u.data_cache[asset].loc[:current_date]["Close"]
            cp = float(closes.iloc[-1])
            sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else cp
            above_200 += int(cp >= sma200)
        ratio = above_200 / len(u.anon_universe)
        multiplier = (
            1.0 if ratio >= 0.60 else
            0.80 if ratio >= 0.40 else
            0.50 if ratio >= 0.20 else
            self.params["breadth_min_multiplier"]
        )
        adjusted = {a: v * multiplier for a, v in adjusted.items()}

        cash = max(0.0, 100.0 - sum(adjusted.values()))
        combined = {**adjusted, "CASH": cash}
        clean = {k: max(0.0, v) for k, v in combined.items()}
        total = sum(clean.values())
        if total <= 0:
            # exact match to normalize_allocations() in your script: an all-zero
            # result (e.g. every asset stopped out) falls back to 100% cash,
            # not an all-zero allocation that sums to nothing.
            return {a: 0.0 for a in u.anon_universe} | {"CASH": 100.0}
        return {k: round(v / total * 100.0, 6) for k, v in clean.items()}
