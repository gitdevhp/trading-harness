import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import yfinance as yf
from openai import OpenAI


# ============================================================
# YFINANCE CONFIG
# ============================================================

os.environ["YFINANCE_CACHE_DIR"] = "/tmp/yf_cache"

try:
    yf.set_tz_cache_location("/tmp/yf_tz_cache")
except Exception:
    pass


# ============================================================
# OPENAI / QWEN CONFIG
# ============================================================

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

MODEL_NAME = "Qwen/Qwen3.6-35B-A3B-FP8"
# MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-AWQ"


# ============================================================
# BACKTEST CONFIG
# ============================================================

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ADBE", "QCOM", "TXN",
    "AMAT", "MU", "AMZN", "TSLA", "PEP", "COST", "HON", "GOOGL",
    "META", "NFLX",
]

FEE_RATE = 0.0015
INITIAL_CAPITAL = 1_000_000.0


# ============================================================
# GLOBAL STATE
# ============================================================

RAW_UNIVERSE = []
ANONYMOUS_MAP = {}
REVERSE_MAP = {}
ANONYMOUS_UNIVERSE = []
GLOBAL_DATA_CACHE = {}


# ============================================================
# UNIVERSE
# ============================================================

def setup_universe(tickers):
    global RAW_UNIVERSE, ANONYMOUS_MAP, REVERSE_MAP
    global ANONYMOUS_UNIVERSE, GLOBAL_DATA_CACHE

    RAW_UNIVERSE = [str(t).upper() for t in tickers]

    if len(set(RAW_UNIVERSE)) != len(RAW_UNIVERSE):
        raise ValueError("Duplicate tickers are not allowed.")

    ANONYMOUS_MAP = {
        ticker: f"ASSET_{chr(65 + i)}"
        for i, ticker in enumerate(RAW_UNIVERSE)
    }

    REVERSE_MAP = {
        v: k
        for k, v in ANONYMOUS_MAP.items()
    }

    ANONYMOUS_UNIVERSE = list(ANONYMOUS_MAP.values())

    GLOBAL_DATA_CACHE.clear()


# ============================================================
# ROBUST YFINANCE DATA GATHERING
# ============================================================

def _clean_download(df, ticker):
    """
    Normalize yfinance output into a simple DataFrame:

        index = timezone-naive DatetimeIndex
        column = Close

    Handles:
      - empty downloads
      - MultiIndex columns
      - different MultiIndex orientations
      - duplicate dates
      - timezone-aware indexes
      - non-numeric Close values
      - malformed downloads
    """

    if df is None or df.empty:
        raise ValueError(
            f"{ticker}: yfinance returned no rows."
        )

    if isinstance(df.columns, pd.MultiIndex):
        close_series = None

        for i, col in enumerate(df.columns):
            parts = [str(x) for x in col]

            if any(part.lower() == "close" for part in parts):
                try:
                    candidate = df.iloc[:, i]

                    if isinstance(candidate, pd.Series):
                        close_series = candidate
                        break

                except Exception:
                    continue

        if close_series is None:
            try:
                if "Close" in df.columns.get_level_values(0):
                    candidate = df["Close"]

                    if isinstance(candidate, pd.DataFrame):
                        if ticker in candidate.columns:
                            close_series = candidate[ticker]
                        elif len(candidate.columns) == 1:
                            close_series = candidate.iloc[:, 0]
                    else:
                        close_series = candidate
            except Exception:
                pass

        if close_series is None:
            try:
                if "Close" in df.columns.get_level_values(1):
                    candidate = df.xs("Close", axis=1, level=1)

                    if isinstance(candidate, pd.DataFrame):
                        if ticker in candidate.columns:
                            close_series = candidate[ticker]
                        elif len(candidate.columns) == 1:
                            close_series = candidate.iloc[:, 0]
                    else:
                        close_series = candidate
            except Exception:
                pass

        if close_series is None:
            raise ValueError(
                f"{ticker}: downloaded MultiIndex data does not contain a usable Close column."
            )

        df = pd.DataFrame({"Close": close_series})

    else:
        if "Close" not in df.columns:
            close_col = None

            for col in df.columns:
                if str(col).strip().lower() == "close":
                    close_col = col
                    break

            if close_col is None:
                raise ValueError(
                    f"{ticker}: downloaded data is missing Close."
                )

            df = df[[close_col]].rename(columns={close_col: "Close"})
        else:
            df = df[["Close"]]

    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    df = df[~df.index.isna()]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Close"])

    if df.empty:
        raise ValueError(
            f"{ticker}: no usable Close prices after cleaning."
        )

    return df


def _download_ticker(ticker, start_date, end_date):
    """
    Robust downloader.

    Attempts:
      1. yf.download with threads=False
      2. repeated yf.download retries
      3. yf.Ticker(...).history fallback

    This prevents one transient Yahoo Finance failure from
    immediately killing the entire backtest.
    """

    last_error = None

    for attempt in range(4):
        try:
            print(
                f"[DATA] Downloading {ticker} "
                f"(attempt {attempt + 1}/4)..."
            )

            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                actions=False,
                progress=False,
                threads=False,
                timeout=30,
            )

            cleaned = _clean_download(df, ticker)

            print(
                f"[DATA] {ticker}: "
                f"{len(cleaned)} usable rows."
            )

            return cleaned

        except Exception as exc:
            last_error = exc

            print(
                f"[DATA] {ticker}: download attempt "
                f"{attempt + 1} failed: {exc}"
            )

            if attempt < 3:
                time.sleep(2 ** attempt)

    for attempt in range(2):
        try:
            print(
                f"[DATA] Trying history fallback for {ticker} "
                f"(attempt {attempt + 1}/2)..."
            )

            ticker_obj = yf.Ticker(ticker)

            df = ticker_obj.history(
                start=start_date,
                end=end_date,
                auto_adjust=True,
                actions=False,
            )

            cleaned = _clean_download(df, ticker)

            print(
                f"[DATA] {ticker}: history fallback succeeded "
                f"with {len(cleaned)} rows."
            )

            return cleaned

        except Exception as exc:
            last_error = exc

            print(
                f"[DATA] {ticker}: history fallback attempt "
                f"{attempt + 1} failed: {exc}"
            )

            if attempt == 0:
                time.sleep(3)

    raise ValueError(
        f"{ticker}: unable to obtain usable market data "
        f"after all yfinance attempts. Last error: {last_error}"
    )


def prefetch_data(start_date, end_date):
    """
    Download all required market data before the backtest.

    Extra history is downloaded so that the 200-day SMA,
    120-day momentum, etc. are available on the first
    rebalance date.

    No ticker is silently dropped.
    """

    lookback_start = (
        pd.to_datetime(start_date) - pd.Timedelta(days=365)
    ).strftime("%Y-%m-%d")

    download_end = (
        pd.to_datetime(end_date) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    print(
        f"[DATA] Requested range: "
        f"{lookback_start} -> {download_end}"
    )

    failures = []

    for ticker in RAW_UNIVERSE:
        try:
            df = _download_ticker(
                ticker,
                lookback_start,
                download_end,
            )

            GLOBAL_DATA_CACHE[
                ANONYMOUS_MAP[ticker]
            ] = df

        except Exception as exc:
            failures.append(
                f"{ticker}: {exc}"
            )

    if failures:
        message = (
            "\n[DATA] The following tickers could not be loaded:\n"
            + "\n".join(f"  - {x}" for x in failures)
        )

        raise RuntimeError(message)

    print(
        f"[DATA] Successfully loaded "
        f"{len(GLOBAL_DATA_CACHE)}/{len(RAW_UNIVERSE)} assets."
    )


def get_common_trading_days(start_date, end_date):
    """
    Find trading dates shared by every asset.
    """

    sets = []

    for asset in ANONYMOUS_UNIVERSE:
        if asset not in GLOBAL_DATA_CACHE:
            raise ValueError(
                f"Missing cached data for {asset}."
            )

        idx = GLOBAL_DATA_CACHE[asset].index

        dates = {
            d.strftime("%Y-%m-%d")
            for d in idx
            if start_date <= d.strftime("%Y-%m-%d") <= end_date
        }

        if not dates:
            ticker = REVERSE_MAP.get(asset, asset)

            raise ValueError(
                f"{ticker}: no trading days inside "
                f"{start_date} -> {end_date}."
            )

        sets.append(dates)

    days = sorted(set.intersection(*sets))

    if not days:
        raise ValueError(
            "No common trading days across the universe."
        )

    print(
        f"[DATA] Common trading days: "
        f"{days[0]} -> {days[-1]} "
        f"({len(days)} days)"
    )

    return days


# ============================================================
# ALLOCATION UTILITIES
# ============================================================

def normalize_allocations(raw):
    clean = {
        a: max(0.0, float(raw.get(a, 0.0)))
        for a in ANONYMOUS_UNIVERSE
    }

    clean["CASH"] = max(
        0.0,
        float(raw.get("CASH", 0.0)),
    )

    total = sum(clean.values())

    if total <= 0:
        return {
            a: 0.0
            for a in ANONYMOUS_UNIVERSE
        } | {"CASH": 100.0}

    return {
        k: round(v / total * 100.0, 6)
        for k, v in clean.items()
    }


def parse_allocations(text):
    pairs = re.findall(
        r'["\']?([A-Za-z0-9_]+)["\']?\s*:\s*(-?\d+(?:\.\d+)?)',
        text,
    )

    raw = {
        k: float(v)
        for k, v in pairs
        if k in ANONYMOUS_UNIVERSE or k == "CASH"
    }

    return normalize_allocations(raw) if raw else None


# ============================================================
# MARKET SNAPSHOT
# ============================================================

def build_market_snapshot(current_date):
    rows = []

    for asset in ANONYMOUS_UNIVERSE:
        closes = (
            GLOBAL_DATA_CACHE[asset]
            .loc[:current_date]["Close"]
        )

        if closes.empty:
            raise ValueError(
                f"{REVERSE_MAP.get(asset, asset)}: "
                f"no prices available through {current_date}."
            )

        cp = float(closes.iloc[-1])

        sma20 = (
            float(closes.tail(20).mean())
            if len(closes) >= 20
            else cp
        )

        sma50 = (
            float(closes.tail(50).mean())
            if len(closes) >= 50
            else cp
        )

        sma200 = (
            float(closes.tail(200).mean())
            if len(closes) >= 200
            else cp
        )

        mom21 = (
            (cp / float(closes.iloc[-21]) - 1.0) * 100.0
            if len(closes) >= 21
            else 0.0
        )

        mom126 = (
            (cp / float(closes.iloc[-126]) - 1.0) * 100.0
            if len(closes) >= 126
            else 0.0
        )

        vol21 = (
            float(
                closes.tail(21)
                .pct_change()
                .std()
                * np.sqrt(252)
                * 100.0
            )
            if len(closes) >= 21
            else 0.0
        )

        rows.append(
            f"{asset}: Price=${cp:.2f} | "
            f"20d-SMA=${sma20:.2f} | "
            f"50d-SMA=${sma50:.2f} | "
            f"200d-SMA=${sma200:.2f} | "
            f"1M-Mom={mom21:.2f}% | "
            f"6M-Mom={mom126:.2f}% | "
            f"1M-Vol={vol21:.2f}%"
        )

    return "\n".join(rows)


# ============================================================
# PORTFOLIO STATE
# ============================================================

def make_portfolio_state(
    holdings,
    cash,
    prices,
    value,
):
    alloc = {
        a: holdings[a] * prices[a] / value * 100.0
        if value > 0 else 0.0
        for a in ANONYMOUS_UNIVERSE
    }

    alloc["CASH"] = (
        cash / value * 100.0
        if value > 0 else 0.0
    )

    return {
        "portfolio_value": round(value, 2),
        "cash_pct": round(alloc["CASH"], 6),
        "allocations_pct": alloc,
    }


# ============================================================
# EXECUTION
# ============================================================

def execute_rebalance(
    value,
    holdings,
    cash,
    prices,
    target_allocs,
):
    target_allocs = normalize_allocations(
        target_allocs
    )

    target_w = {
        k: v / 100.0
        for k, v in target_allocs.items()
    }

    current_w = {
        a: holdings[a] * prices[a] / value
        if value > 0 else 0.0
        for a in ANONYMOUS_UNIVERSE
    }

    turnover = sum(
        abs(
            target_w.get(a, 0.0)
            - current_w.get(a, 0.0)
        )
        for a in ANONYMOUS_UNIVERSE
    )

    transaction_cost = (
        value * turnover * FEE_RATE
    )

    execution_value = max(
        value - transaction_cost,
        0.0,
    )

    new_cash = (
        execution_value
        * target_w.get("CASH", 0.0)
    )

    new_holdings = {
        a:
        execution_value
        * target_w.get(a, 0.0)
        / prices[a]
        if prices[a] > 0
        else 0.0
        for a in ANONYMOUS_UNIVERSE
    }

    orders = []

    for a in ANONYMOUS_UNIVERSE:
        delta = (
            target_w.get(a, 0.0)
            - current_w.get(a, 0.0)
        )

        if abs(delta) < 1e-12:
            continue

        notional = abs(delta) * value

        orders.append(
            {
                "asset": REVERSE_MAP[a],
                "direction": (
                    "buy"
                    if delta > 0
                    else "sell"
                ),
                "delta_weight": round(delta, 8),
                "trade_value": round(notional, 2),
                "transaction_cost": round(
                    notional * FEE_RATE,
                    4,
                ),
            }
        )

    return (
        new_holdings,
        new_cash,
        turnover,
        transaction_cost,
        orders,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(daily_nav):
    nav = pd.Series(
        [
            float(x["portfolio_value"])
            for x in daily_nav
        ]
    )

    dates = pd.to_datetime(
        [
            x["date"]
            for x in daily_nav
        ]
    )

    r = (
        nav.pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    initial = float(nav.iloc[0])
    final = float(nav.iloc[-1])

    total_return = (
        final / initial - 1.0
        if initial > 0
        else 0.0
    )

    sd = (
        float(r.std(ddof=1))
        if len(r) > 1
        else 0.0
    )

    mean = (
        float(r.mean())
        if len(r)
        else 0.0
    )

    volatility = (
        sd * np.sqrt(252.0)
    )

    sharpe = (
        mean / sd * np.sqrt(252.0)
        if sd > 0
        else 0.0
    )

    neg = r[r < 0]

    dsd = (
        float(neg.std(ddof=1))
        if len(neg) > 1
        else 0.0
    )

    sortino = (
        mean / dsd * np.sqrt(252.0)
        if dsd > 0
        else 0.0
    )

    dd = (
        nav / nav.cummax()
        - 1.0
    )

    mdd = (
        float(dd.min())
        if len(dd)
        else 0.0
    )

    years = max(
        (
            dates[-1]
            - dates[0]
        ).days / 365.25,
        1.0 / 365.25,
    )

    cagr = (
        (final / initial) ** (1.0 / years) - 1.0
        if initial > 0
        else 0.0
    )

    calmar = (
        cagr / abs(mdd)
        if mdd < 0
        else 0.0
    )

    return {
        "final_nav": round(final, 2),
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown": round(mdd, 6),
        "calmar_ratio": round(calmar, 4),
        "volatility": round(volatility, 6),
    }


# ============================================================
# OUTPUT
# ============================================================

def write_result(
    output_file,
    variant,
    start_date,
    end_date,
    initial_capital,
    tickers,
    daily_nav,
    rebalances,
):
    metrics = calculate_metrics(daily_nav)

    summary = {
        "protocol_version": "fair-portbench-v1",
        "variant": variant,
        "provider": "local_vllm_openai_compatible",
        "model_name": MODEL_NAME,
        "data_source": "Yahoo Finance via yfinance",
        "data_adjustment": "auto_adjust=True",
        "future_data_allowed": False,
        "lookahead_bias": False,
        "information_cutoff": "rebalance_date_close",
        "universe": list(tickers),
        "start_date": start_date,
        "end_date": end_date,
        "initial_nav": round(initial_capital, 2),
        **metrics,
        "n_rebalances": len(rebalances),
        "total_turnover": round(
            sum(
                x["total_turnover"]
                for x in rebalances
            ),
            8,
        ),
        "total_transaction_cost": round(
            sum(
                x["total_transaction_cost"]
                for x in rebalances
            ),
            4,
        ),
        "transaction_cost_model": {
            "slippage_bps": 10.0,
            "commission_bps": 5.0,
            "total_bps": 15.0,
            "fee_rate": FEE_RATE,
            "cost_basis": "absolute_stock_weight_change",
        },
        "rebalance_frequency":
            "first_common_trading_day_each_new_calendar_month",
        "initial_portfolio": "equal_weight",
        "leverage": 1.0,
        "cash_allowed": True,
        "output_schema":
            "summary_rebalances_daily_nav_v1",
    }

    with open(output_file, "w") as f:
        json.dump(
            {
                "summary": summary,
                "rebalances": rebalances,
                "daily_nav": daily_nav,
            },
            f,
            indent=4,
            allow_nan=False,
        )

    print(
        f"{variant} -> {output_file}"
    )


# ============================================================
# GPT / INSTITUTIONAL HARNESS
# ============================================================

def _portfolio_vol(weights, returns_df):
    if (
        len(returns_df) < 10
        or weights.sum() <= 0
    ):
        return 0.15

    cov = (
        returns_df.cov().values
        * 252.0
    )

    cov = (
        0.75 * cov
        + 0.25 * np.diag(
            np.diag(cov)
        )
    )

    return float(
        np.sqrt(
            max(
                float(
                    weights.T
                    @ cov
                    @ weights
                ),
                1e-8,
            )
        )
    )


def apply_institutional_harness(
    raw_allocs,
    current_date,
):
    adjusted = {
        a: float(
            raw_allocs.get(a, 0.0)
        )
        for a in ANONYMOUS_UNIVERSE
    }

    for asset in ANONYMOUS_UNIVERSE:
        if adjusted[asset] <= 0:
            continue

        closes = (
            GLOBAL_DATA_CACHE[asset]
            .loc[:current_date]["Close"]
        )

        cp = float(
            closes.iloc[-1]
        )

        m20 = (
            cp / float(closes.iloc[-20]) - 1.0
            if len(closes) >= 20
            else 0.0
        )

        m60 = (
            cp / float(closes.iloc[-60]) - 1.0
            if len(closes) >= 60
            else 0.0
        )

        m120 = (
            cp / float(closes.iloc[-120]) - 1.0
            if len(closes) >= 120
            else 0.0
        )

        composite = (
            0.5 * m20
            + 0.3 * m60
            + 0.2 * m120
        )

        sma50 = (
            float(closes.tail(50).mean())
            if len(closes) >= 50
            else cp
        )

        sma200 = (
            float(closes.tail(200).mean())
            if len(closes) >= 200
            else cp
        )

        trend = (
            cp / sma50 - 1.0
        ) + (
            sma50 / sma200 - 1.0
        )

        score = float(
            np.clip(
                0.6 * composite
                + 0.4 * trend,
                -0.5,
                0.5,
            )
        )

        adjusted[asset] *= (
            1.0 + score
        )

    vec = np.array(
        [
            adjusted[a]
            for a in ANONYMOUS_UNIVERSE
        ],
        dtype=float,
    )

    if vec.sum() > 0:
        returns = {
            a:
            GLOBAL_DATA_CACHE[a]
            .loc[:current_date]["Close"]
            .pct_change()
            .dropna()
            for a in ANONYMOUS_UNIVERSE
        }

        ret_df = (
            pd.DataFrame(returns)
            .tail(126)
            .fillna(0.0)
        )

        vol = _portfolio_vol(
            vec / 100.0,
            ret_df,
        )

        if vol > 0.215:
            vec *= (
                0.18 / vol
            )

        adjusted = {
            a: float(vec[i])
            for i, a in enumerate(
                ANONYMOUS_UNIVERSE
            )
        }

    above_200 = 0

    for asset in ANONYMOUS_UNIVERSE:
        closes = (
            GLOBAL_DATA_CACHE[asset]
            .loc[:current_date]["Close"]
        )

        cp = float(
            closes.iloc[-1]
        )

        sma200 = (
            float(closes.tail(200).mean())
            if len(closes) >= 200
            else cp
        )

        above_200 += int(
            cp >= sma200
        )

    ratio = (
        above_200
        / len(ANONYMOUS_UNIVERSE)
    )

    multiplier = (
        1.0
        if ratio >= 0.60
        else 0.80
        if ratio >= 0.40
        else 0.50
        if ratio >= 0.20
        else 0.25
    )

    adjusted = {
        a: v * multiplier
        for a, v in adjusted.items()
    }

    cash = max(
        0.0,
        100.0 - sum(adjusted.values()),
    )

    return normalize_allocations(
        adjusted | {"CASH": cash}
    )


# ============================================================
# REACT DECISION
# ============================================================

def DECISION_FUNCTION(
    current_date,
    state,
    rebalance_days,
):
    snapshot = build_market_snapshot(
        current_date
    )

    def market(_=""):
        return snapshot

    def portfolio(_=""):
        return (
            f"Portfolio Value: "
            f"${state['portfolio_value']:,.2f}\n"
            f"Cash: "
            f"{state['cash_pct']:.2f}%\n"
            f"Allocations: "
            f"{state['allocations_pct']}"
        )

    tools = {
        "get_market_screener": market,
        "get_portfolio_status": portfolio,
    }

    # Same ReAct structure as the no-harness baseline.
    # The institutional harness is applied only AFTER
    # the ReAct model produces its raw allocation.
    system = f"""You are an autonomous ReAct Portfolio Manager on {current_date}.
Assets: {ANONYMOUS_UNIVERSE} + CASH
Decision horizon: {rebalance_days} trading days.

Available tools:
- get_market_screener[]
- get_portfolio_status[]

The tools expose the complete information available to you.
You have no access to ticker names, news, fundamentals, future data,
future prices, or any information not returned by a tool.

Rules:
- Long-only.
- No leverage.
- CASH allowed.
- Do not invent assets.
- Final action MUST be Target_Allocations.

ReAct format:
Thought: <brief reasoning>
Action: get_market_screener[]
Observation: <tool result>
Thought: <brief reasoning>
Action: Target_Allocations[{{"ASSET_A": 10, ..., "CASH": 0}}]

The Target_Allocations percentages must be non-negative and
will be normalized to 100%."""

    messages = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": (
                f"Set target allocations for the next "
                f"{rebalance_days} trading days."
            ),
        },
    ]

    decision = None

    for _ in range(4):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=800,
            stop=["Observation:"],
        )

        message = response.choices[0].message

        # Qwen3.6 may return content=None when the response is
        # placed in the reasoning field. Never call .strip()
        # directly on message.content.
        reply = (message.content or "").strip()

        if not reply:
            reply = (getattr(message, "reasoning", None) or "").strip()

        messages.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )

        match = re.search(
            r"Action:\s*([A-Za-z_]+)\[(.*?)\]",
            reply,
            re.DOTALL,
        )

        if not match:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Continue using the required ReAct format. "
                        "You must provide an Action."
                    ),
                }
            )
            continue

        action = match.group(1)
        arg = match.group(2).strip()

        if action == "Target_Allocations":
            decision = parse_allocations(
                arg
            )

            if decision is not None:
                break

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The Target_Allocations action was invalid. "
                        "Return valid JSON-style asset percentage "
                        "allocations using only the provided assets "
                        "and CASH."
                    ),
                }
            )
            continue

        if action in tools:
            observation = tools[action](arg)
        else:
            observation = "Tool not found."

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Observation: {observation}"
                ),
            }
        )

    raw = (
        decision
        or normalize_allocations(
            {
                a:
                100.0
                / len(ANONYMOUS_UNIVERSE)
                for a in ANONYMOUS_UNIVERSE
            }
            | {"CASH": 0.0}
        )
    )

    return apply_institutional_harness(
        raw,
        current_date,
    )


# ============================================================
# RUN
# ============================================================

VARIANT_NAME = "react_gpt_harness"
DEFAULT_OUTPUT = (
    "react_gpt_harness_results_compare.json"
)


def run_backtest(
    start_date,
    end_date,
    initial_capital,
    tickers,
    output_file,
    rebalance_days=20,
    warmup_days=0,
):
    if warmup_days != 0:
        raise ValueError(
            "Fair comparison requires --warmup-days 0."
        )

    if rebalance_days <= 0:
        raise ValueError(
            "--rebalance-days must be positive."
        )

    setup_universe(tickers)

    prefetch_data(
        start_date,
        end_date,
    )

    trading_days = get_common_trading_days(
        start_date,
        end_date,
    )

    first_prices = {
        a:
        float(
            GLOBAL_DATA_CACHE[a]
            .loc[trading_days[0], "Close"]
        )
        for a in ANONYMOUS_UNIVERSE
    }

    cash = 0.0

    holdings = {
        a:
        initial_capital
        / len(ANONYMOUS_UNIVERSE)
        / first_prices[a]
        for a in ANONYMOUS_UNIVERSE
    }

    target_allocs = normalize_allocations(
        {
            a:
            100.0
            / len(ANONYMOUS_UNIVERSE)
            for a in ANONYMOUS_UNIVERSE
        }
        | {"CASH": 0.0}
    )

    daily_nav = []
    rebalances = []

    for idx, current_date in enumerate(
        trading_days
    ):
        prices = {
            a:
            float(
                GLOBAL_DATA_CACHE[a]
                .loc[current_date, "Close"]
            )
            for a in ANONYMOUS_UNIVERSE
        }

        value = (
            cash
            + sum(
                holdings[a] * prices[a]
                for a in ANONYMOUS_UNIVERSE
            )
        )

        is_rebalance = (
            idx > 0
            and pd.to_datetime(
                current_date
            ).month
            != pd.to_datetime(
                trading_days[idx - 1]
            ).month
        )

        turnover = 0.0
        transaction_cost = 0.0
        orders = []

        if is_rebalance:
            state = make_portfolio_state(
                holdings,
                cash,
                prices,
                value,
            )

            target_allocs = normalize_allocations(
                DECISION_FUNCTION(
                    current_date,
                    state,
                    rebalance_days,
                )
            )

            (
                holdings,
                cash,
                turnover,
                transaction_cost,
                orders,
            ) = execute_rebalance(
                value,
                holdings,
                cash,
                prices,
                target_allocs,
            )

            post_value = (
                cash
                + sum(
                    holdings[a] * prices[a]
                    for a in ANONYMOUS_UNIVERSE
                )
            )

            rebalances.append(
                {
                    "date": current_date,
                    "portfolio_value_before":
                        round(value, 2),
                    "portfolio_value_after_trade":
                        round(post_value, 2),
                    "total_turnover":
                        round(turnover, 8),
                    "total_transaction_cost":
                        round(
                            transaction_cost,
                            4,
                        ),
                    "fee_rate": FEE_RATE,
                    "orders": orders,
                    "target_allocations":
                        {
                            REVERSE_MAP.get(k, k):
                            round(
                                float(v),
                                6,
                            )
                            for k, v in target_allocs.items()
                        },
                }
            )

        value = (
            cash
            + sum(
                holdings[a] * prices[a]
                for a in ANONYMOUS_UNIVERSE
            )
        )

        daily_nav.append(
            {
                "date": current_date,
                "prices":
                    {
                        REVERSE_MAP[a]:
                        round(float(p), 6)
                        for a, p in prices.items()
                    },
                "portfolio_value":
                    round(value, 2),
                "harnessed_allocations":
                    {
                        REVERSE_MAP.get(k, k):
                        round(float(v), 6)
                        for k, v
                        in target_allocs.items()
                    },
                "turnover":
                    round(turnover, 8),
                "transaction_cost":
                    round(transaction_cost, 4),
                "fee_rate": FEE_RATE,
            }
        )

    write_result(
        output_file,
        VARIANT_NAME,
        start_date,
        end_date,
        initial_capital,
        tickers,
        daily_nav,
        rebalances,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_UNIVERSE,
    )

    parser.add_argument(
        "--start",
        default="2024-01-01",
    )

    parser.add_argument(
        "--end",
        default="2024-12-31",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--initial-capital",
        type=float,
        default=INITIAL_CAPITAL,
    )

    parser.add_argument(
        "--rebalance-days",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--warmup-days",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    run_backtest(
        args.start,
        args.end,
        args.initial_capital,
        args.tickers,
        args.output,
        args.rebalance_days,
        args.warmup_days,
    )
