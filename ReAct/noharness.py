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
# ENVIRONMENT
# ============================================================

os.environ["YFINANCE_CACHE_DIR"] = "/tmp/yf_cache"

try:
    yf.set_tz_cache_location("/tmp/yf_tz_cache")
except Exception:
    pass


# ============================================================
# LOCAL QWEN
# ============================================================

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

MODEL_NAME = os.environ.get("REACT_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
# MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-AWQ"


# ============================================================
# CONFIG
# ============================================================

DEFAULT_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AVGO",
    "AMD",
    "ADBE",
    "QCOM",
    "TXN",
    "AMAT",
    "MU",
    "AMZN",
    "TSLA",
    "PEP",
    "COST",
    "HON",
    "GOOGL",
    "META",
    "NFLX",
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
    global RAW_UNIVERSE
    global ANONYMOUS_MAP
    global REVERSE_MAP
    global ANONYMOUS_UNIVERSE
    global GLOBAL_DATA_CACHE

    RAW_UNIVERSE = [
        str(t).upper()
        for t in tickers
    ]

    if len(set(RAW_UNIVERSE)) != len(RAW_UNIVERSE):
        raise ValueError(
            "Duplicate tickers are not allowed."
        )

    ANONYMOUS_MAP = {
        ticker: f"ASSET_{chr(65 + i)}"
        for i, ticker in enumerate(RAW_UNIVERSE)
    }

    REVERSE_MAP = {
        anonymous: ticker
        for ticker, anonymous in ANONYMOUS_MAP.items()
    }

    ANONYMOUS_UNIVERSE = list(
        ANONYMOUS_MAP.values()
    )

    GLOBAL_DATA_CACHE.clear()


# ============================================================
# YFINANCE
# ============================================================

def _clean_download(df):
    if df is None or df.empty:
        raise ValueError(
            "No usable market data returned."
        )

    if isinstance(df.columns, pd.MultiIndex):

        if "Close" in df.columns.get_level_values(0):

            df = df.xs(
                "Close",
                axis=1,
                level=0,
            )

            if isinstance(df, pd.DataFrame):

                if df.shape[1] == 1:
                    df = df.iloc[:, 0].to_frame("Close")

                else:
                    df = df.iloc[:, [0]].copy()
                    df.columns = ["Close"]

        else:

            if "Close" in df.columns.get_level_values(-1):

                df = df.xs(
                    "Close",
                    axis=1,
                    level=-1,
                )

                if isinstance(df, pd.DataFrame):

                    if df.shape[1] == 1:
                        df = df.iloc[:, 0].to_frame("Close")

                    else:
                        df = df.iloc[:, [0]].copy()
                        df.columns = ["Close"]

    else:

        if "Close" not in df.columns:
            raise ValueError(
                "Downloaded data is missing Close."
            )

        df = df[["Close"]].copy()

    if "Close" not in df.columns:
        raise ValueError(
            "Downloaded data is missing Close."
        )

    df = df.sort_index()

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    try:
        if getattr(
            df.index,
            "tz",
            None,
        ) is not None:

            df.index = df.index.tz_localize(None)

    except Exception:
        pass

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Close"]
    )

    if df.empty:
        raise ValueError(
            "No usable market data returned."
        )

    return df


def _download_ticker(
    ticker,
    start_date,
    end_date,
    retries=4,
):

    last_error = None

    for attempt in range(retries):

        try:

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

            cleaned = _clean_download(
                df
            )

            if not cleaned.empty:
                return cleaned

        except Exception as exc:
            last_error = exc

        if attempt < retries - 1:
            time.sleep(
                2 ** attempt
            )

    try:

        ticker_obj = yf.Ticker(
            ticker
        )

        df = ticker_obj.history(
            start=start_date,
            end=end_date,
            auto_adjust=True,
            actions=False,
        )

        cleaned = _clean_download(
            df
        )

        if not cleaned.empty:
            return cleaned

    except Exception as exc:
        last_error = exc

    raise ValueError(
        f"Unable to download usable market data for {ticker}. "
        f"Last error: {last_error}"
    )


def prefetch_data(
    start_date,
    end_date,
):

    download_start = (
        pd.to_datetime(
            start_date
        ).strftime("%Y-%m-%d")
    )

    download_end = (
        pd.to_datetime(
            end_date
        )
        + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    for ticker in RAW_UNIVERSE:

        print(
            f"Downloading {ticker}..."
        )

        df = _download_ticker(
            ticker,
            download_start,
            download_end,
        )

        GLOBAL_DATA_CACHE[
            ANONYMOUS_MAP[ticker]
        ] = df

        print(
            f"  {ticker}: "
            f"{len(df)} rows "
            f"({df.index.min().date()} -> "
            f"{df.index.max().date()})"
        )


# ============================================================
# TRADING DAYS
# ============================================================

def get_common_trading_days(
    start_date,
    end_date,
):

    sets = []

    for asset in ANONYMOUS_UNIVERSE:

        idx = GLOBAL_DATA_CACHE[
            asset
        ].index

        dates = {
            d.strftime("%Y-%m-%d")
            for d in idx
            if (
                start_date
                <= d.strftime("%Y-%m-%d")
                <= end_date
            )
        }

        sets.append(dates)

    if not sets:
        raise ValueError(
            "No market data available."
        )

    days = sorted(
        set.intersection(*sets)
    )

    if not days:
        raise ValueError(
            "No common trading days "
            "across the universe."
        )

    return days


# ============================================================
# ALLOCATION HANDLING
# ============================================================

def normalize_allocations(raw):

    clean = {
        asset: max(
            0.0,
            float(
                raw.get(
                    asset,
                    0.0,
                )
            ),
        )
        for asset in ANONYMOUS_UNIVERSE
    }

    clean["CASH"] = max(
        0.0,
        float(
            raw.get(
                "CASH",
                0.0,
            )
        ),
    )

    total = sum(
        clean.values()
    )

    if total <= 0:

        return {
            **{
                asset: 0.0
                for asset in ANONYMOUS_UNIVERSE
            },
            "CASH": 100.0,
        }

    return {
        key: round(
            value / total * 100.0,
            6,
        )
        for key, value
        in clean.items()
    }


def parse_allocations(text):

    pairs = re.findall(
        r'["\']?([A-Za-z0-9_]+)["\']?\s*:\s*(-?\d+(?:\.\d+)?)',
        text,
    )

    raw = {
        key: float(value)
        for key, value
        in pairs
        if (
            key in ANONYMOUS_UNIVERSE
            or key == "CASH"
        )
    }

    if not raw:
        return None

    return normalize_allocations(
        raw
    )


# ============================================================
# REACT TOOL: CURRENT PRICES ONLY
# ============================================================

def build_market_snapshot(
    current_date,
):
    """
    Current market prices only.

    No historical prices.
    No price arrays.
    No moving averages.
    No momentum.
    No volatility.
    No support/resistance.
    No channels.
    No Fibonacci.
    No RSI.
    No Bollinger Bands.
    No technical indicators.
    No rule-based signals.
    No harness logic.
    """

    current_dt = pd.to_datetime(
        current_date
    )

    lines = [
        f"Market data date: {current_date}",
        "Current prices:",
    ]

    for asset in ANONYMOUS_UNIVERSE:

        df = GLOBAL_DATA_CACHE[
            asset
        ]

        available = df.loc[
            df.index <= current_dt,
            "Close",
        ]

        if available.empty:

            lines.append(
                f"{asset}: no data available"
            )

            continue

        current_price = float(
            available.iloc[-1]
        )

        lines.append(
            f"{asset}: "
            f"${current_price:.6f}"
        )

    return "\n".join(lines)


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
        asset: (
            holdings[asset]
            * prices[asset]
            / value
            * 100.0
            if value > 0
            else 0.0
        )
        for asset in ANONYMOUS_UNIVERSE
    }

    alloc["CASH"] = (
        cash
        / value
        * 100.0
        if value > 0
        else 0.0
    )

    return {
        "portfolio_value": round(
            value,
            2,
        ),
        "cash_pct": round(
            alloc["CASH"],
            6,
        ),
        "allocations_pct": alloc,
    }


# ============================================================
# REACT DECISION
# ============================================================

def DECISION_FUNCTION(
    current_date,
    state,
    prices,
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

    system = f"""
You are an autonomous ReAct Portfolio Manager.

Date:
{current_date}

Assets:
{ANONYMOUS_UNIVERSE} + CASH

Decision horizon:
{rebalance_days} trading days.

You have two tools:

- get_market_screener[]
- get_portfolio_status[]

Use the tools to gather the information available to you
before making the portfolio decision.

The market tool provides only current asset prices.
No historical prices or technical indicators are provided.

Rules:

- Long-only.
- No leverage.
- CASH is allowed.
- Only use the provided assets.
- Do not invent assets.
- Do not identify the real ticker names behind ASSET_*.
- Do not use future information.
- Do not assume market information that was not provided.
- Make the decision yourself.
- The harness does not apply an investment strategy.
- The harness does not modify your allocation decision.
- Final action MUST be Target_Allocations.

Use this ReAct format:

Thought: ...
Action: get_market_screener[]
Observation: ...

Thought: ...
Action: get_portfolio_status[]
Observation: ...

Thought: ...
Action: Target_Allocations[{{"ASSET_A": 10, ..., "CASH": 0}}]
"""

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

    # --------------------------------------------------------
    # REACT LOOP
    # --------------------------------------------------------

    for _ in range(4):

        # Keep only the system prompt, original task,
        # and the latest two conversation messages.
        #
        # This prevents the conversation from growing
        # unnecessarily while preserving ReAct behavior.

        request_messages = [
            messages[0],
            messages[1],
        ]

        if len(messages) > 2:

            request_messages.extend(
                messages[-2:]
            )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=request_messages,
            temperature=0.0,
            max_tokens=800,
            stop=["Observation:"],
        )

        message = response.choices[0].message

        # Qwen3.6 can return content=None when the answer is placed
        # in the reasoning field. Avoid None.strip() crashes.
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
            r"Action:\s*(\w+)\[(.*?)\]",
            reply,
            re.DOTALL,
        )

        if not match:
            continue

        action = match.group(1)
        arg = match.group(2).strip()

        # ----------------------------------------------------
        # FINAL DECISION
        # ----------------------------------------------------

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
                        "Observation: Invalid allocation "
                        "format. Return Target_Allocations "
                        "with numeric allocations."
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # TOOL CALL
        # ----------------------------------------------------

        if action in tools:

            observation = tools[action](
                arg
            )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Observation: {observation}"
                    ),
                }
            )

        else:

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Observation: Tool not found."
                    ),
                }
            )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if decision is None:

        decision = normalize_allocations(
            {
                **{
                    asset:
                    100.0
                    / len(
                        ANONYMOUS_UNIVERSE
                    )
                    for asset
                    in ANONYMOUS_UNIVERSE
                },
                "CASH": 0.0,
            }
        )

    return decision


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
        key: value / 100.0
        for key, value
        in target_allocs.items()
    }

    current_w = {
        asset: (
            holdings[asset]
            * prices[asset]
            / value
            if value > 0
            else 0.0
        )
        for asset in ANONYMOUS_UNIVERSE
    }

    turnover = sum(
        abs(
            target_w.get(
                asset,
                0.0,
            )
            - current_w.get(
                asset,
                0.0,
            )
        )
        for asset
        in ANONYMOUS_UNIVERSE
    )

    transaction_cost = (
        value
        * turnover
        * FEE_RATE
    )

    execution_value = max(
        value
        - transaction_cost,
        0.0,
    )

    new_cash = (
        execution_value
        * target_w.get(
            "CASH",
            0.0,
        )
    )

    new_holdings = {
        asset: (
            execution_value
            * target_w.get(
                asset,
                0.0,
            )
            / prices[asset]
            if prices[asset] > 0
            else 0.0
        )
        for asset
        in ANONYMOUS_UNIVERSE
    }

    orders = []

    for asset in ANONYMOUS_UNIVERSE:

        delta = (
            target_w.get(
                asset,
                0.0,
            )
            - current_w.get(
                asset,
                0.0,
            )
        )

        if abs(delta) < 1e-12:
            continue

        notional = (
            abs(delta)
            * value
        )

        orders.append(
            {
                "asset": REVERSE_MAP[
                    asset
                ],
                "direction": (
                    "buy"
                    if delta > 0
                    else "sell"
                ),
                "delta_weight": round(
                    delta,
                    8,
                ),
                "trade_value": round(
                    notional,
                    2,
                ),
                "transaction_cost": round(
                    notional
                    * FEE_RATE,
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

def calculate_metrics(
    daily_nav,
):

    nav = pd.Series(
        [
            float(
                x["portfolio_value"]
            )
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
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    initial = float(
        nav.iloc[0]
    )

    final = float(
        nav.iloc[-1]
    )

    total_return = (
        final / initial - 1.0
        if initial > 0
        else 0.0
    )

    sd = (
        float(
            r.std(
                ddof=1
            )
        )
        if len(r) > 1
        else 0.0
    )

    mean = (
        float(
            r.mean()
        )
        if len(r)
        else 0.0
    )

    volatility = (
        sd
        * np.sqrt(252.0)
    )

    sharpe = (
        mean
        / sd
        * np.sqrt(252.0)
        if sd > 0
        else 0.0
    )

    neg = r[r < 0]

    dsd = (
        float(
            neg.std(
                ddof=1
            )
        )
        if len(neg) > 1
        else 0.0
    )

    sortino = (
        mean
        / dsd
        * np.sqrt(252.0)
        if dsd > 0
        else 0.0
    )

    dd = (
        nav
        / nav.cummax()
        - 1.0
    )

    mdd = (
        float(
            dd.min()
        )
        if len(dd)
        else 0.0
    )

    years = max(
        (
            dates[-1]
            - dates[0]
        ).days
        / 365.25,
        1.0 / 365.25,
    )

    cagr = (
        (
            final
            / initial
        )
        ** (
            1.0
            / years
        )
        - 1.0
        if initial > 0
        else 0.0
    )

    calmar = (
        cagr
        / abs(mdd)
        if mdd < 0
        else 0.0
    )

    return {
        "final_nav": round(
            final,
            2,
        ),
        "total_return": round(
            total_return,
            6,
        ),
        "cagr": round(
            cagr,
            6,
        ),
        "sharpe_ratio": round(
            sharpe,
            4,
        ),
        "sortino_ratio": round(
            sortino,
            4,
        ),
        "max_drawdown": round(
            mdd,
            6,
        ),
        "calmar_ratio": round(
            calmar,
            4,
        ),
        "volatility": round(
            volatility,
            6,
        ),
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

    metrics = calculate_metrics(
        daily_nav
    )

    summary = {
        "protocol_version": (
            "fair-portbench-v1"
        ),
        "variant": variant,
        "provider": (
            "local_vllm_openai_compatible"
        ),
        "model_name": MODEL_NAME,
        "data_source": (
            "Yahoo Finance via yfinance"
        ),
        "data_adjustment": (
            "auto_adjust=True"
        ),
        "future_data_allowed": False,
        "lookahead_bias": False,
        "information_cutoff": (
            "rebalance_date_close"
        ),
        "universe": list(tickers),
        "start_date": start_date,
        "end_date": end_date,
        "initial_nav": round(
            initial_capital,
            2,
        ),
        **metrics,
        "n_rebalances": len(
            rebalances
        ),
        "total_turnover": round(
            sum(
                x["total_turnover"]
                for x in rebalances
            ),
            8,
        ),
        "total_transaction_cost": round(
            sum(
                x[
                    "total_transaction_cost"
                ]
                for x in rebalances
            ),
            4,
        ),
        "transaction_cost_model": {
            "slippage_bps": 10.0,
            "commission_bps": 5.0,
            "total_bps": 15.0,
            "fee_rate": FEE_RATE,
            "cost_basis": (
                "absolute_stock_weight_change"
            ),
        },
        "rebalance_frequency": (
            "first_common_trading_day_each_new_calendar_month"
        ),
        "initial_portfolio": (
            "equal_weight"
        ),
        "leverage": 1.0,
        "cash_allowed": True,
        "output_schema": (
            "summary_rebalances_daily_nav_v1"
        ),
    }

    with open(
        output_file,
        "w",
    ) as f:

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
# BACKTEST
# ============================================================

VARIANT_NAME = (
    "react_no_harness"
)

DEFAULT_OUTPUT = (
    "react_no_harness_results_compare.json"
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
            "Fair comparison requires "
            "--warmup-days 0."
        )

    if rebalance_days <= 0:
        raise ValueError(
            "--rebalance-days must be positive."
        )

    setup_universe(
        tickers
    )

    prefetch_data(
        start_date,
        end_date,
    )

    trading_days = (
        get_common_trading_days(
            start_date,
            end_date,
        )
    )

    first_prices = {
        asset: float(
            GLOBAL_DATA_CACHE[
                asset
            ]
            .loc[
                trading_days[0],
                "Close",
            ]
        )
        for asset
        in ANONYMOUS_UNIVERSE
    }

    cash = 0.0

    holdings = {
        asset: (
            initial_capital
            / len(
                ANONYMOUS_UNIVERSE
            )
            / first_prices[
                asset
            ]
        )
        for asset
        in ANONYMOUS_UNIVERSE
    }

    target_allocs = (
        normalize_allocations(
            {
                **{
                    asset:
                    100.0
                    / len(
                        ANONYMOUS_UNIVERSE
                    )
                    for asset
                    in ANONYMOUS_UNIVERSE
                },
                "CASH": 0.0,
            }
        )
    )

    daily_nav = []
    rebalances = []

    for idx, current_date in enumerate(
        trading_days
    ):

        prices = {
            asset: float(
                GLOBAL_DATA_CACHE[
                    asset
                ]
                .loc[
                    current_date,
                    "Close",
                ]
            )
            for asset
            in ANONYMOUS_UNIVERSE
        }

        value = (
            cash
            + sum(
                holdings[asset]
                * prices[asset]
                for asset
                in ANONYMOUS_UNIVERSE
            )
        )

        is_rebalance = (
            idx > 0
            and pd.to_datetime(
                current_date
            ).month
            != pd.to_datetime(
                trading_days[
                    idx - 1
                ]
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

            target_allocs = (
                DECISION_FUNCTION(
                    current_date,
                    state,
                    prices,
                    rebalance_days,
                )
            )

            target_allocs = (
                normalize_allocations(
                    target_allocs
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
                    holdings[asset]
                    * prices[asset]
                    for asset
                    in ANONYMOUS_UNIVERSE
                )
            )

            rebalances.append(
                {
                    "date": current_date,
                    "portfolio_value_before": round(
                        value,
                        2,
                    ),
                    "portfolio_value_after_trade": round(
                        post_value,
                        2,
                    ),
                    "total_turnover": round(
                        turnover,
                        8,
                    ),
                    "total_transaction_cost": round(
                        transaction_cost,
                        4,
                    ),
                    "fee_rate": FEE_RATE,
                    "orders": orders,
                    "target_allocations": {
                        REVERSE_MAP.get(
                            key,
                            key,
                        ): round(
                            float(val),
                            6,
                        )
                        for key, val
                        in target_allocs.items()
                    },
                }
            )

        value = (
            cash
            + sum(
                holdings[asset]
                * prices[asset]
                for asset
                in ANONYMOUS_UNIVERSE
            )
        )

        daily_nav.append(
            {
                "date": current_date,
                "prices": {
                    REVERSE_MAP[
                        asset
                    ]: round(
                        float(price),
                        6,
                    )
                    for asset, price
                    in prices.items()
                },
                "portfolio_value": round(
                    value,
                    2,
                ),
                "harnessed_allocations": {
                    REVERSE_MAP.get(
                        key,
                        key,
                    ): round(
                        float(val),
                        6,
                    )
                    for key, val
                    in target_allocs.items()
                },
                "turnover": round(
                    turnover,
                    8,
                ),
                "transaction_cost": round(
                    transaction_cost,
                    4,
                ),
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
# CLI
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