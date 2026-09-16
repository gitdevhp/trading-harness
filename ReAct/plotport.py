import argparse
import json
import math
import os
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEE_RATE = 0.0015
TRADING_DAYS_PER_YEAR = 252.0
SUPPORTED_PLOT_EXTENSIONS = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".eps", ".ps", ".tif", ".tiff", ".webp", ".bmp", ".gif"}


def resolve_output_plot_path(path: str) -> str:
    if not path:
        return "results_compare.png"
    root, ext = os.path.splitext(path)
    if ext.lower() in SUPPORTED_PLOT_EXTENSIONS:
        return path
    return f"{root}.png" if ext else f"{path}.png"


def _is_record(x: Any) -> bool:
    if not isinstance(x, dict):
        return False
    has_date = any(k in x for k in ("date", "datetime", "timestamp", "time"))
    has_value = any(k in x for k in ("portfolio_value", "portfolioValue", "nav", "value", "equity", "portfolio_nav"))
    return has_date and has_value


def _extract_records(data: Any, file_path: str) -> List[dict]:
    # Plain daily-record list.
    if isinstance(data, list):
        rows = [x for x in data if _is_record(x)]
        if rows:
            return rows

    # Wrapped formats. PortBench's newer closed-loop format uses daily_nav.
    if isinstance(data, dict):
        keys = [
            "daily_nav", "daily_results", "daily", "results", "records",
            "history", "backtest_results", "portfolio_history", "data"
        ]
        for key in keys:
            value = data.get(key)
            if isinstance(value, list):
                rows = [x for x in value if _is_record(x)]
                if rows:
                    return rows

        # A few files wrap the actual result object one level deeper.
        for value in data.values():
            if isinstance(value, dict):
                for key in keys:
                    nested = value.get(key)
                    if isinstance(nested, list):
                        rows = [x for x in nested if _is_record(x)]
                        if rows:
                            return rows

        if _is_record(data):
            return [data]

    raise ValueError(
        f"Could not identify result records in '{file_path}'. "
        "Expected a daily record list or a wrapper containing daily_nav, "
        "results, records, history, or backtest_results."
    )


def load_json_records(file_path: str) -> List[dict]:
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Result file '{file_path}' not found.")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = _extract_records(data, file_path)

    cleaned = []
    for row in raw:
        date_value = row.get("date", row.get("datetime", row.get("timestamp", row.get("time"))))
        value = None
        for key in ("portfolio_value", "portfolioValue", "nav", "value", "equity", "portfolio_nav"):
            if key in row:
                value = row[key]
                break
        try:
            date = pd.Timestamp(date_value)
            value = float(value)
        except (TypeError, ValueError):
            continue
        if pd.isna(date) or not np.isfinite(value):
            continue
        r = dict(row)
        r["date"] = date
        r["portfolio_value"] = value
        cleaned.append(r)

    if not cleaned:
        raise ValueError(f"No valid daily records found in '{file_path}'.")

    df = pd.DataFrame(cleaned).sort_values("date").drop_duplicates("date", keep="last")
    return df.to_dict("records")


def load_json_series(file_path: Optional[str]):
    if not file_path or not os.path.exists(file_path):
        return None, None
    records = load_json_records(file_path)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    series = pd.to_numeric(df["portfolio_value"], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"No numeric portfolio values found in '{file_path}'.")
    return series, records


def extract_prices(records: List[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        prices = r.get("prices")
        if not isinstance(prices, dict):
            continue
        clean = {}
        for ticker, price in prices.items():
            try:
                p = float(price)
                if np.isfinite(p) and p > 0:
                    clean[str(ticker)] = p
            except (TypeError, ValueError):
                pass
        if clean:
            rows.append({"date": pd.Timestamp(r["date"]), **clean})
    if not rows:
        raise ValueError("Primary result file has no usable 'prices' dictionaries.")
    df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    return df.apply(pd.to_numeric, errors="coerce")


def monthly_rebalance_dates(index) -> set:
    idx = pd.DatetimeIndex(index).sort_values()
    if len(idx) == 0:
        return set()
    s = pd.Series(idx, index=idx)
    return {pd.Timestamp(x) for x in s.groupby([idx.year, idx.month]).first().tolist()}


def equal_weight_curve(price_df: pd.DataFrame, initial_capital: float) -> pd.Series:
    prices = price_df.dropna(how="any").copy()
    if prices.empty:
        raise ValueError("No common dates available for equal-weight benchmark.")
    dates = list(prices.index)
    rebalance_dates = monthly_rebalance_dates(prices.index)
    n = len(prices.columns)
    weights = np.ones(n) / n
    value = float(initial_capital)
    out = []

    for i, date in enumerate(dates):
        px = prices.iloc[i].values.astype(float)
        if i == 0:
            value *= (1.0 - FEE_RATE)  # initial establishment
            weights = np.ones(n) / n
        else:
            prev = prices.iloc[i - 1].values.astype(float)
            r = px / prev - 1.0
            value *= 1.0 + float(np.dot(weights, r))
            drifted = weights * (1.0 + r)
            denom = float(drifted.sum())
            if denom > 0:
                weights = drifted / denom
            if date in rebalance_dates:
                target = np.ones(n) / n
                turnover = float(np.abs(target - weights).sum())
                value *= 1.0 - FEE_RATE * turnover
                weights = target
        out.append(value)
    return pd.Series(out, index=prices.index, name="Equal-Weight (1/n)")


def risk_parity_curve(price_df: pd.DataFrame, initial_capital: float, lookback: int = 126) -> pd.Series:
    prices = price_df.dropna(how="any").copy()
    returns = prices.pct_change()
    dates = list(prices.index)
    rebalance_dates = monthly_rebalance_dates(prices.index)
    n = len(prices.columns)
    weights = np.ones(n) / n
    value = float(initial_capital) * (1.0 - FEE_RATE)
    out = [value]

    for i in range(1, len(dates)):
        px = prices.iloc[i].values.astype(float)
        prev = prices.iloc[i - 1].values.astype(float)
        r = px / prev - 1.0
        value *= 1.0 + float(np.dot(weights, r))
        drifted = weights * (1.0 + r)
        denom = float(drifted.sum())
        if denom > 0:
            weights = drifted / denom

        date = dates[i]
        if date in rebalance_dates:
            window = returns.iloc[max(0, i - lookback):i]
            vols = window.std(ddof=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
            inv = np.where(vols > 1e-8, 1.0 / vols, 0.0)
            target = inv / inv.sum() if inv.sum() > 0 else np.ones(n) / n
            turnover = float(np.abs(target - weights).sum())
            value *= 1.0 - FEE_RATE * turnover
            weights = target
        out.append(value)
    return pd.Series(out, index=prices.index, name="Risk Parity")


def normalize(s: pd.Series) -> pd.Series:
    s = pd.Series(s).copy()
    s.index = pd.to_datetime(s.index)
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.empty or abs(float(s.iloc[0])) < 1e-12:
        return s
    return s / float(s.iloc[0])


def align_systems(systems: dict) -> dict:
    clean = {k: v for k, v in systems.items() if v is not None and len(v) > 0}
    if not clean:
        return {}
    common = None
    for s in clean.values():
        idx = pd.DatetimeIndex(s.index)
        common = idx if common is None else common.intersection(idx)
    common = pd.DatetimeIndex(common).sort_values()
    return {k: s.reindex(common).ffill().dropna() for k, s in clean.items()}


def metrics(series: pd.Series, benchmark: Optional[pd.Series] = None) -> dict:
    s = pd.Series(series).dropna().sort_index()
    r = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return {"Total Return (%)": 0.0, "CAGR (%)": 0.0, "Ann. Volatility (%)": 0.0, "Max Drawdown (%)": 0.0, "Sharpe Ratio": 0.0, "Sortino Ratio": 0.0, "Calmar Ratio": 0.0, "Beta (Systematic Risk)": 0.0, "Alpha (% p.a.)": 0.0, "drawdown_series": s * 0}

    years = max((s.index[-1] - s.index[0]).days / 365.25, 1.0 / 252.0)
    total = float(s.iloc[-1] / s.iloc[0] - 1.0) if s.iloc[0] != 0 else 0.0
    cagr = float((s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0) if s.iloc[0] > 0 and s.iloc[-1] > 0 else 0.0
    vol = float(r.std(ddof=1) * math.sqrt(252.0)) if len(r) > 1 else 0.0
    downside = r[r < 0]
    down = float(math.sqrt(np.mean(downside ** 2)) * math.sqrt(252.0)) if len(downside) else 0.0
    sharpe = float((r.mean() * 252.0) / vol) if vol > 1e-12 else 0.0
    sortino = float(cagr / down) if down > 1e-12 else 0.0
    peak = s.cummax()
    dd = s / peak - 1.0
    mdd = abs(float(dd.min()))
    calmar = float(cagr / mdd) if mdd > 1e-12 else 0.0
    beta = alpha = 0.0

    if benchmark is not None:
        b = pd.Series(benchmark).sort_index()
        joined = pd.concat([s.rename("p"), b.rename("b")], axis=1).dropna()
        if len(joined) >= 3:
            pr = joined.p.pct_change()
            br = joined.b.pct_change()
            x = pd.concat([pr, br], axis=1).dropna()
            if len(x) >= 3 and x.iloc[:, 1].var(ddof=1) > 1e-15:
                beta = float(x.iloc[:, 1].cov(x.iloc[:, 0]) / x.iloc[:, 1].var(ddof=1))
                alpha = float((x.iloc[:, 0].mean() - beta * x.iloc[:, 1].mean()) * 252.0 * 100.0)

    return {"Total Return (%)": round(total * 100, 2), "CAGR (%)": round(cagr * 100, 2), "Ann. Volatility (%)": round(vol * 100, 2), "Max Drawdown (%)": round(mdd * 100, 2), "Sharpe Ratio": round(sharpe, 2), "Sortino Ratio": round(sortino, 2), "Calmar Ratio": round(calmar, 2), "Beta (Systematic Risk)": round(beta, 2), "Alpha (% p.a.)": round(alpha, 2), "drawdown_series": dd}


def generate_evaluation_report(harness_file="react_harness_results_compare.json", gpt_harness_file="react_gpt_harness_results_compare.json", no_harness_file="react_no_harness_results_compare.json", raw_llm_file="qwen_raw_results_compare.json", output_plot="results_compare.png"):
    requested = [
        ("ReAct + Risk Harness", harness_file, True),
        ("ReAct + GPT Harness", gpt_harness_file, False),
        ("Vanilla ReAct", no_harness_file, False),
        ("Raw Direct Qwen", raw_llm_file, False),
    ]
    systems = {}
    primary_records = None

    for name, path, required in requested:
        if not path or not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Primary file '{path}' not found.")
            print(f"WARNING: optional file '{path}' not found; skipping {name}.")
            continue
        try:
            series, records = load_json_series(path)
            systems[name] = series
            if required:
                primary_records = records
        except Exception as exc:
            if required:
                raise
            print(f"WARNING: could not load {name} from '{path}': {exc}")

    if not systems:
        raise ValueError("No usable model result JSON files were found.")
    if primary_records is None:
        primary_records = load_json_records(harness_file)

    price_df = extract_prices(primary_records)
    initial = float(systems["ReAct + Risk Harness"].iloc[0]) if "ReAct + Risk Harness" in systems else float(next(iter(systems.values())).iloc[0])

    try:
        systems["Equal-Weight (1/n)"] = equal_weight_curve(price_df, initial)
    except Exception as exc:
        print(f"WARNING: equal-weight benchmark unavailable: {exc}")
    try:
        systems["Risk Parity"] = risk_parity_curve(price_df, initial)
    except Exception as exc:
        print(f"WARNING: risk-parity benchmark unavailable: {exc}")

    systems = align_systems(systems)
    benchmark = systems.get("Equal-Weight (1/n)", next(iter(systems.values())))
    all_metrics = {name: metrics(series, benchmark) for name, series in systems.items()}

    print("\n" + "=" * 170)
    print("PORTBENCH / REACT SYSTEM COMPARISON")
    print("=" * 170)
    print("Model JSON costs: already embedded in portfolio_value; no second cost is applied.")
    print("Quant baselines: 15 bps turnover cost applied independently.")
    print("\n" + f"{'Metric':<28} | " + " | ".join(f"{n[:18]:<18}" for n in all_metrics))
    print("-" * 170)
    for key in ["Total Return (%)", "CAGR (%)", "Ann. Volatility (%)", "Max Drawdown (%)", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Beta (Systematic Risk)", "Alpha (% p.a.)"]:
        print(f"{key:<28} | " + " | ".join(f"{all_metrics[n][key]:<18}" for n in all_metrics))
    print("=" * 170 + "\n")

    styles = {
        "ReAct + Risk Harness": dict(color="#1f77b4", linestyle="-", linewidth=2.5),
        "ReAct + GPT Harness": dict(color="#9467bd", linestyle="-", linewidth=2.3),
        "Vanilla ReAct": dict(color="#2ca02c", linestyle="--", linewidth=1.7),
        "Raw Direct Qwen": dict(color="#d62728", linestyle="-.", linewidth=1.7),
        "Equal-Weight (1/n)": dict(color="#ff7f0e", linestyle=":", linewidth=1.8),
        "Risk Parity": dict(color="#17becf", linestyle="-", linewidth=1.5),
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    for name, series in systems.items():
        st = styles.get(name, dict(linestyle="-", linewidth=1.4))
        m = all_metrics[name]
        ax1.plot(normalize(series).index, normalize(series).values, label=f"{name} (Return {m['Total Return (%)']}%, Sharpe {m['Sharpe Ratio']})", **st)
        dd = m["drawdown_series"] * 100.0
        ax2.plot(dd.index, dd.values, label=name, **st)
        if "Harness" in name:
            ax2.fill_between(dd.index, dd.values, 0, color=st.get("color", "gray"), alpha=0.08)

    ax1.set_title("AI Trading Systems vs Quantitative Baselines", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Growth of $1.00")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8)
    ax2.set_title("Underwater Chart (Drawdown %)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    resolved = resolve_output_plot_path(output_plot)
    plt.savefig(resolved, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph saved to {resolved}")

    metrics_file = os.path.splitext(resolved)[0] + "_metrics.json"
    serializable = {name: {k: v for k, v in m.items() if k != "drawdown_series"} for name, m in all_metrics.items()}
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=4)
    print(f"Metrics saved to {metrics_file}")
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ReAct/Qwen portfolio result JSON files.")
    parser.add_argument("--harness-file", default="react_harness_results_compare.json")
    parser.add_argument("--gpt-harness-file", default="react_gpt_harness_results_compare.json")
    parser.add_argument("--no-harness-file", default="react_no_harness_results_compare.json")
    parser.add_argument("--raw-llm-file", default="qwen_raw_results_compare.json")
    parser.add_argument("--output", "-o", default="results_compare.png")
    args = parser.parse_args()
    generate_evaluation_report(args.harness_file, args.gpt_harness_file, args.no_harness_file, args.raw_llm_file, args.output)