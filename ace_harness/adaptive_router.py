"""
Adaptive system router for the ACE harness.

At each rebalance, computes two observable universe-structure features
from market data — momentum breadth and universe size — and maps them to
one of four systems:

    DEBATER  (intra-task only)       — broad universe, bullish regime
    ACE      (dual_permanent, full)  — concentrated universe, bullish
    MEMORY   (inter-task only)       — broad but mixed/bearish regime
    HARNESS  (baseline)              — concentrated + bear, LLM less reliable

Routing rationale (empirically grounded in ablation results):
  • Large universe (N ≥ 30): Debater's broad reasoning over the full asset
    list adds value when there are many candidates to compare.
  • Small universe (N < 30): ACE's tight memory + one-shot debate cycle
    works better when the Solver already has a focused view.
  • Bear regime (price-breadth < BEAR_THRESHOLD): LLM judgment is noisier
    in choppy markets; a plain risk harness is more stable.
  • Otherwise: memory or full debate based on universe size.

All features are computed directly from data_cache — no screener text parsing,
no look-ahead, no external calls.
"""
import numpy as np

# Routing thresholds — tunable without touching any core system
N_LARGE = 30          # assets; ≥ this → "large universe" branch
BEAR_THRESHOLD = 0.40  # fraction of assets above 200d-SMA; below = bear regime
BULL_THRESHOLD = 0.55  # above = confident bull regime

SYSTEM_DEBATER = "debater"
SYSTEM_ACE     = "ace"
SYSTEM_MEMORY  = "memory"
SYSTEM_HARNESS = "harness"


def compute_features(universe, current_date: str) -> dict:
    """Return a dict of numeric routing features from already-cached market data.

    No network calls — reads from universe.data_cache which was populated
    by universe.prefetch() before the backtest loop starts.
    """
    n_stocks = len(universe.raw_universe)
    above_200 = 0
    mom1m_vals = []

    for asset in universe.anon_universe:
        try:
            closes = universe.data_cache[asset].loc[:current_date]["Close"]
            if closes.empty:
                continue
            cp = float(closes.iloc[-1])
            sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else cp
            if cp > sma200:
                above_200 += 1
            if len(closes) >= 21:
                mom21 = (cp - float(closes.iloc[-21])) / float(closes.iloc[-21]) * 100.0
                mom1m_vals.append(mom21)
        except Exception:
            continue

    valid = len([a for a in universe.anon_universe
                 if not universe.data_cache.get(a, None) is None])
    denom = max(valid, 1)
    price_breadth = above_200 / denom
    return_dispersion = float(np.std(mom1m_vals)) if len(mom1m_vals) >= 2 else 0.0
    mom_breadth = sum(1 for m in mom1m_vals if m > 0) / max(len(mom1m_vals), 1)

    return {
        "n_stocks":          n_stocks,
        "price_breadth":     round(price_breadth, 3),   # fraction above 200d-SMA
        "mom_breadth":       round(mom_breadth, 3),      # fraction with positive 1M-Mom
        "return_dispersion": round(return_dispersion, 2), # cross-sectional spread
    }


def route(features: dict) -> tuple:
    """Map features to (system_name, reason_tag).

    system_name is one of the SYSTEM_* constants above.
    reason_tag is a short, loggable string explaining the decision.
    """
    n = features["n_stocks"]
    breadth = features["price_breadth"]

    is_large = n >= N_LARGE
    is_bear  = breadth < BEAR_THRESHOLD
    is_bull  = breadth >= BULL_THRESHOLD

    if is_bear:
        # Bear regime: LLM intervention adds noise, trust the harness
        return SYSTEM_HARNESS, "bear_regime"

    if is_large:
        if is_bull:
            # Broad + bullish → Debater's cross-asset reasoning helps most
            return SYSTEM_DEBATER, "large_bull"
        else:
            # Broad + mixed → memory context helps, skip debate overhead
            return SYSTEM_MEMORY, "large_mixed"
    else:
        # Concentrated universe
        if is_bull:
            # Full ACE system: tight debate + memory in a focused bull market
            return SYSTEM_ACE, "concentrated_bull"
        else:
            # Concentrated + mixed → memory only (debate overhead not worth it)
            return SYSTEM_MEMORY, "concentrated_mixed"
