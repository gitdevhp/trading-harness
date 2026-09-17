"""
Portfolio metrics and comparison chart for ACE harness backtest results.

Computes: total return, CAGR, Sharpe, Sortino, Calmar, max drawdown, ann. vol,
alpha, and beta (vs. equal-weight benchmark), then plots all systems plus six
quantitative baselines (Equal-Weight, Risk-Parity, 60/40, Min-Variance,
Cov-Risk-Parity, Black-Litterman) on the same chart.

Usage (from the directory containing ace_harness/):

    python -m ace_harness.plot_results results/*.json \
        --tickers AAPL MSFT NVDA AVGO AMD ADBE QCOM TXN AMAT MU AMZN TSLA \
                  PEP COST HON GOOGL META NFLX \
        --start 2024-01-01 --end 2024-12-31

The --tickers/--start/--end arguments drive the quantitative baselines only.
If omitted the baselines are skipped.
"""
import argparse
import json
import os

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


TRADING_DAYS = 252
FEE_RATE = 0.0015  # 15 bps


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_result(path):
    with open(path) as f:
        data = json.load(f)
    dates = [r["date"] for r in data]
    values = np.array([r["portfolio_value"] for r in data], dtype=float)
    label = os.path.splitext(os.path.basename(path))[0]
    label = label.replace("monthly_", "").replace("_results", "")
    return {"label": label, "dates": dates, "values": values}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _daily_returns(values):
    return np.diff(values) / values[:-1]


def _cagr(values, n_days):
    if n_days <= 0 or values[0] <= 0:
        return 0.0
    return float((values[-1] / values[0]) ** (TRADING_DAYS / n_days) - 1)


def _sharpe(daily_ret, rf_daily=0.0):
    ex = daily_ret - rf_daily
    std = np.std(ex)
    return float(np.mean(ex) * TRADING_DAYS / (std * np.sqrt(TRADING_DAYS) + 1e-9))


def _sortino(daily_ret, rf_daily=0.0):
    ex = daily_ret - rf_daily
    neg = ex[ex < 0]
    downside = np.sqrt(np.mean(neg ** 2)) if len(neg) > 0 else 1e-9
    return float(np.mean(ex) * TRADING_DAYS / (downside * np.sqrt(TRADING_DAYS) + 1e-9))


def _max_drawdown(values):
    peak = np.maximum.accumulate(values)
    dd = (values - peak) / peak
    return float(dd.min())


def _calmar(values, n_days):
    cagr = _cagr(values, n_days)
    mdd = abs(_max_drawdown(values))
    return float(cagr / (mdd + 1e-9))


def _alpha_beta(portfolio_ret, benchmark_ret):
    if len(portfolio_ret) < 2 or len(benchmark_ret) < 2:
        return 0.0, 1.0
    cov_mat = np.cov(portfolio_ret, benchmark_ret)
    beta = float(cov_mat[0, 1] / (cov_mat[1, 1] + 1e-12))
    alpha_daily = float(np.mean(portfolio_ret) - beta * np.mean(benchmark_ret))
    alpha_ann = alpha_daily * TRADING_DAYS
    return alpha_ann, beta


def compute_metrics(result, benchmark_values=None):
    values = result["values"]
    n_days = len(values) - 1
    daily_ret = _daily_returns(values)

    total_return = float((values[-1] / values[0] - 1) * 100)
    cagr = _cagr(values, n_days) * 100
    ann_vol = float(np.std(daily_ret) * np.sqrt(TRADING_DAYS) * 100)
    sharpe = _sharpe(daily_ret)
    sortino = _sortino(daily_ret)
    mdd = _max_drawdown(values) * 100
    calmar = _calmar(values, n_days)

    alpha, beta = 0.0, 1.0
    if benchmark_values is not None and len(benchmark_values) == len(values):
        bret = _daily_returns(benchmark_values)
        alpha, beta = _alpha_beta(daily_ret, bret)
        alpha *= 100  # as percent

    return {
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "ann_vol_pct": round(ann_vol, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_drawdown_pct": round(mdd, 2),
        "alpha_ann_pct": round(alpha, 2),
        "beta": round(beta, 3),
    }


# ---------------------------------------------------------------------------
# Quantitative baselines  (require pandas + price data from MarketUniverse)
# ---------------------------------------------------------------------------

def _price_df(universe, start, end):
    """Return a DataFrame (date x asset) of daily close prices."""
    if not _HAS_PANDAS:
        return None
    tickers = universe.tickers
    records = {}
    for t in tickers:
        prices = universe.close_prices_range(t, start, end)
        if prices:
            records[t] = prices
    if not records:
        return None
    df = pd.DataFrame(records)
    df = df.dropna()
    return df


def _rebalance_dates(price_df, freq="month"):
    """First trading day of each new calendar month.

    Returns index values in the same type as price_df.index so that
    ``d in rebal_set`` comparisons inside _run_strategy always work
    regardless of whether the index is strings or Timestamps.
    """
    orig_index = list(price_df.index)
    ts_index = pd.DatetimeIndex(orig_index)
    dates = []
    prev_month = None
    for orig, ts in zip(orig_index, ts_index):
        m = (ts.year, ts.month)
        if m != prev_month:
            dates.append(orig)
            prev_month = m
    return dates


def _run_strategy(price_df, initial_capital, weight_fn):
    """Generic daily NAV calculator.
    weight_fn(price_df_up_to_today, rebal_idx) -> np.array of weights (sum 1).
    Called only on rebalance dates. Applies FEE_RATE * turnover on each rebal.
    """
    dates = list(price_df.index)
    n = len(price_df.columns)
    rebal_set = set(_rebalance_dates(price_df))

    cash = initial_capital
    shares = np.zeros(n)
    weights = np.ones(n) / n  # equal-weight start
    daily_nav = [initial_capital]

    for i, d in enumerate(dates):
        prices = price_df.iloc[i].values
        if np.any(prices <= 0):
            daily_nav.append(daily_nav[-1])
            continue

        value = float(np.dot(shares, prices) + cash)

        if d in rebal_set:
            new_w = weight_fn(price_df.iloc[: i + 1], i)
            new_w = np.clip(new_w, 0, None)
            s = new_w.sum()
            if s <= 0:
                new_w = np.ones(n) / n
            else:
                new_w /= s

            target_values = new_w * value
            old_values = shares * prices
            turnover = float(np.sum(np.abs(target_values - old_values)) / (value + 1e-9))
            fee = turnover * FEE_RATE * value
            value -= fee

            shares = (new_w * value) / prices
            cash = 0.0
            weights = new_w

        daily_nav.append(float(np.dot(shares, prices) + cash))

    return np.array(daily_nav[1:])  # drop the seed, align with price_df rows


def equal_weight_curve(price_df, initial_capital):
    def _w(df, i):
        n = df.shape[1]
        return np.ones(n) / n
    return _run_strategy(price_df, initial_capital, _w)


def risk_parity_curve(price_df, initial_capital, lookback=63):
    def _w(df, i):
        window = df.iloc[max(0, i - lookback): i + 1]
        if len(window) < 5:
            return np.ones(df.shape[1]) / df.shape[1]
        ret = window.pct_change().dropna().values
        vol = np.std(ret, axis=0)
        vol = np.where(vol < 1e-8, 1e-8, vol)
        inv_vol = 1.0 / vol
        return inv_vol / inv_vol.sum()
    return _run_strategy(price_df, initial_capital, _w)


def sixty_forty_curve(price_df, initial_capital, bond_return_annual=0.045):
    """60% equity (equal-weight within the universe) + 40% notional bond.
    Bond earns a fixed daily return; added as a synthetic column."""
    if not _HAS_PANDAS:
        return None
    bond_daily = (1 + bond_return_annual) ** (1.0 / TRADING_DAYS) - 1
    n_stocks = price_df.shape[1]
    stock_w = 0.60 / n_stocks
    bond_w = 0.40

    dates = list(price_df.index)
    rebal_set = set(_rebalance_dates(price_df))

    cash = initial_capital
    shares = np.zeros(n_stocks)
    bond_value = 0.0

    # Initial allocation
    prices0 = price_df.iloc[0].values
    value0 = initial_capital * (1 - FEE_RATE * 0.60)
    shares = (stock_w * value0) / prices0
    bond_value = bond_w * value0
    cash = 0.0

    daily_nav = []
    for i, d in enumerate(dates):
        prices = price_df.iloc[i].values
        bond_value *= (1 + bond_daily)
        value = float(np.dot(shares, prices) + bond_value + cash)

        if d in rebal_set and i > 0:
            target_stock = 0.60 * value / n_stocks
            target_bond = 0.40 * value
            old_stock_vals = shares * prices
            turnover = float(np.sum(np.abs(target_stock - old_stock_vals)) / (value + 1e-9))
            fee = turnover * FEE_RATE * value
            value -= fee
            shares = (0.60 * value / n_stocks) / prices
            bond_value = 0.40 * value
            cash = 0.0

        daily_nav.append(float(np.dot(shares, prices) + bond_value + cash))

    return np.array(daily_nav)


def min_variance_curve(price_df, initial_capital, lookback=126):
    if not _HAS_SCIPY:
        return None

    def _w(df, i):
        n = df.shape[1]
        window = df.iloc[max(0, i - lookback): i + 1]
        if len(window) < 10:
            return np.ones(n) / n
        ret = window.pct_change().dropna().values
        cov = np.cov(ret.T) + np.eye(n) * 1e-8
        w0 = np.ones(n) / n
        result = minimize(
            lambda w: w @ cov @ w,
            w0,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"ftol": 1e-9, "maxiter": 200},
        )
        return result.x if result.success else w0

    return _run_strategy(price_df, initial_capital, _w)


def cov_risk_parity_curve(price_df, initial_capital, lookback=126):
    if not _HAS_SCIPY:
        return None

    def _w(df, i):
        n = df.shape[1]
        window = df.iloc[max(0, i - lookback): i + 1]
        if len(window) < 10:
            return np.ones(n) / n
        ret = window.pct_change().dropna().values
        cov = np.cov(ret.T) + np.eye(n) * 1e-8
        target = 1.0 / n

        def objective(w):
            port_var = w @ cov @ w
            rc = w * (cov @ w) / (port_var + 1e-12)
            return float(np.sum((rc - target) ** 2))

        w0 = np.ones(n) / n
        result = minimize(
            objective, w0, method="SLSQP",
            bounds=[(1e-4, 1)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"ftol": 1e-12, "maxiter": 500},
        )
        return result.x if result.success else w0

    return _run_strategy(price_df, initial_capital, _w)


def black_litterman_curve(price_df, initial_capital, lookback=126, tau=0.025, risk_aversion=2.5):
    if not _HAS_SCIPY:
        return None

    def _w(df, i):
        n = df.shape[1]
        window = df.iloc[max(0, i - lookback): i + 1]
        if len(window) < 21:
            return np.ones(n) / n

        ret = window.pct_change().dropna().values
        cov = np.cov(ret.T) + np.eye(n) * 1e-8
        w_eq = np.ones(n) / n
        pi = risk_aversion * cov @ w_eq  # equilibrium expected returns

        # 1-month momentum views
        recent = window.iloc[-21:]
        Q = ((recent.iloc[-1] / recent.iloc[0]) - 1).values
        P = np.eye(n)
        Omega = tau * np.diag(np.diag(cov))

        tau_cov = tau * cov
        M = np.linalg.inv(np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(Omega) @ P)
        mu_bl = M @ (np.linalg.inv(tau_cov) @ pi + P.T @ np.linalg.inv(Omega) @ Q)

        def neg_sharpe(w):
            port_ret = w @ mu_bl
            port_vol = np.sqrt(w @ cov @ w + 1e-12)
            return -port_ret / (port_vol + 1e-9)

        w0 = np.ones(n) / n
        result = minimize(
            neg_sharpe, w0, method="SLSQP",
            bounds=[(0, 0.30)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"ftol": 1e-9, "maxiter": 300},
        )
        return result.x if result.success else w0

    return _run_strategy(price_df, initial_capital, _w)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_BASELINE_COLORS = {
    "Equal-Weight":      "#1f77b4",
    "Risk-Parity":       "#ff7f0e",
    "60/40":             "#8c564b",
    "Min-Variance":      "#e377c2",
    "Cov-Risk-Parity":   "#7f7f7f",
    "Black-Litterman":   "#bcbd22",
}

_SYSTEM_COLORS = [
    "#2ca02c", "#d62728", "#9467bd", "#17becf", "#aec7e8",
    "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
]


def generate_report(results, output_dir, tickers=None, start=None, end=None,
                    initial_capital=1_000_000.0):
    """
    results: list of dicts with keys 'label', 'dates', 'values'
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build equal-weight baseline from price data (needed for alpha/beta)
    ew_values = None
    baselines = {}

    if tickers and start and end and _HAS_PANDAS:
        try:
            from ace_harness.market import MarketUniverse
            universe = MarketUniverse(tickers)
            universe.prefetch(start, end)

            # Build price DataFrame from prefetched data_cache
            records = {}
            for anon, df_raw in universe.data_cache.items():
                closes = df_raw["Close"].loc[start:end]
                if not closes.empty:
                    records[anon] = closes.values.tolist()

            if records:
                # Use common trading days
                common = universe.common_trading_days(start, end)
                cols = list(records.keys())
                matrix = {}
                for anon, df_raw in universe.data_cache.items():
                    if anon in records:
                        s = df_raw["Close"].loc[start:end]
                        s.index = s.index.strftime("%Y-%m-%d")
                        matrix[anon] = [float(s.get(d, float("nan"))) for d in common]
                df = pd.DataFrame(matrix, index=common).dropna()
                if len(df) > 5:
                    ew_vals = equal_weight_curve(df, initial_capital)
                    baselines["Equal-Weight"] = ew_vals
                    ew_values = ew_vals

                    rp = risk_parity_curve(df, initial_capital)
                    if rp is not None:
                        baselines["Risk-Parity"] = rp

                    s60 = sixty_forty_curve(df, initial_capital)
                    if s60 is not None:
                        baselines["60/40"] = s60

                    if _HAS_SCIPY:
                        mv = min_variance_curve(df, initial_capital)
                        if mv is not None:
                            baselines["Min-Variance"] = mv

                        crp = cov_risk_parity_curve(df, initial_capital)
                        if crp is not None:
                            baselines["Cov-Risk-Parity"] = crp

                        bl = black_litterman_curve(df, initial_capital)
                        if bl is not None:
                            baselines["Black-Litterman"] = bl
        except Exception as e:
            print(f"[plot_results] baseline computation skipped: {e}")

    # Compute metrics for each system
    metric_rows = []
    for res in results:
        bench = ew_values if ew_values is not None and len(ew_values) == len(res["values"]) else None
        m = compute_metrics(res, bench)
        m["label"] = res["label"]
        metric_rows.append(m)

    # Compute metrics for baselines
    for name, vals in baselines.items():
        fake_res = {"values": vals}
        bench = ew_values if ew_values is not None and len(ew_values) == len(vals) else None
        m = compute_metrics(fake_res, bench)
        m["label"] = name
        metric_rows.append(m)

    # Print table
    cols = ["label", "total_return_pct", "cagr_pct", "ann_vol_pct",
            "sharpe", "sortino", "calmar", "max_drawdown_pct", "alpha_ann_pct", "beta"]
    headers = ["System", "TotalRet%", "CAGR%", "AnnVol%", "Sharpe", "Sortino", "Calmar", "MaxDD%", "Alpha%", "Beta"]
    col_w = 13

    sep = "-" * (20 + col_w * (len(headers) - 1))
    print(sep)
    print(f"{'System':<20}" + "".join(f"{h:>{col_w}}" for h in headers[1:]))
    print(sep)
    for row in metric_rows:
        vals_str = "".join(f"{row[c]:>{col_w}.2f}" if isinstance(row[c], float) else f"{row[c]:>{col_w}}"
                           for c in cols[1:])
        print(f"{row['label']:<20}{vals_str}")
    print(sep)

    # Plot — use integer x-axis for all series so they align regardless of date format
    if _HAS_MPL and results:
        fig, ax = plt.subplots(figsize=(14, 7))

        for i, res in enumerate(results):
            color = _SYSTEM_COLORS[i % len(_SYSTEM_COLORS)]
            v = res["values"]
            normalized = v / v[0] * 100
            ax.plot(range(len(normalized)), normalized, label=res["label"], color=color, linewidth=2)

        for name, vals in baselines.items():
            color = _BASELINE_COLORS.get(name, "#aaaaaa")
            normalized = vals / vals[0] * 100
            ax.plot(range(len(normalized)), normalized, label=name, color=color,
                    linewidth=1.5, linestyle="--", alpha=0.85)

        ax.set_title("ACE Harness Systems vs Quantitative Baselines (NAV indexed to 100)")
        ax.set_ylabel("NAV (indexed)")
        ax.set_xlabel("Trading Days")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        out_path = os.path.join(output_dir, "ace_comparison.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nChart saved -> {out_path}")

    # Save metrics JSON
    out_json = os.path.join(output_dir, "ace_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metric_rows, f, indent=2)
    print(f"Metrics saved -> {out_json}")

    return metric_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot and compare ACE harness backtest results")
    parser.add_argument("result_files", nargs="*", help="JSON result files from run_monthly.py")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--tickers", nargs="+",
                         default=["AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ADBE", "QCOM", "TXN",
                                  "AMAT", "MU", "AMZN", "TSLA", "PEP", "COST", "HON", "GOOGL", "META", "NFLX"])
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--initial_capital", type=float, default=1_000_000.0)
    args = parser.parse_args()

    if not args.result_files:
        import glob
        args.result_files = sorted(glob.glob(os.path.join(args.output_dir, "*_results.json")))

    results = []
    for path in args.result_files:
        try:
            results.append(load_result(path))
        except Exception as e:
            print(f"Could not load {path}: {e}")

    if not results:
        print("No results found. Run run_monthly.py first.")
        return

    generate_report(
        results, args.output_dir,
        tickers=args.tickers, start=args.start, end=args.end,
        initial_capital=args.initial_capital,
    )


if __name__ == "__main__":
    main()
