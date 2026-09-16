import argparse
import json
import random
import re

import numpy as np
import pandas as pd
import yfinance as yf
from openai import OpenAI


# ============================================================
# YFINANCE CONFIGURATION
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

MODEL_NAME = "Qwen/Qwen3.6-35B-A3B-FP8"
# MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-AWQ"


# ============================================================
# BACKTEST CONFIGURATION
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
# UNIVERSE SETUP
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
        v: k
        for k, v in ANONYMOUS_MAP.items()
    }

    ANONYMOUS_UNIVERSE = list(
        ANONYMOUS_MAP.values()
    )

    GLOBAL_DATA_CACHE.clear()
    PRICE_INDEX_BASE.clear()
    DISPLAY_MAP.clear()
    DISPLAY_REVERSE.clear()

def reshuffle_display_map():
    """Shuffle which display label each internal asset gets this rebalance period."""
    global DISPLAY_MAP, DISPLAY_REVERSE
    n = len(ANONYMOUS_UNIVERSE)
    labels = [f"ASSET_{chr(65+i)}" for i in range(n)]
    random.shuffle(labels)
    DISPLAY_MAP = {internal: display for internal, display in zip(ANONYMOUS_UNIVERSE, labels)}
    DISPLAY_REVERSE = {v: k for k, v in DISPLAY_MAP.items()}


# ============================================================
# YFINANCE DATA CLEANING
# ============================================================

def _clean_download(df):
    if df is None:
        raise ValueError(
            "Downloaded data is None."
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.sort_index()

    df = df[
        ~df.index.duplicated(
            keep="first"
        )
    ]

    if "Close" not in df.columns:
        raise ValueError(
            "Downloaded data is missing Close."
        )

    df = df[
        ["Close"]
    ].dropna()

    if df.empty:
        raise ValueError(
            "No usable market data returned."
        )

    return df


# ============================================================
# YFINANCE DATA PREFETCH
# ============================================================

def prefetch_data(start_date, end_date):

    # --------------------------------------------------------
    # PURE PRICE-ONLY BASELINE
    #
    # Only download the actual backtest period.
    #
    # No historical lookback is downloaded for Qwen.
    # --------------------------------------------------------

    download_start = pd.to_datetime(
        start_date
    ).strftime("%Y-%m-%d")

    download_end = (
        pd.to_datetime(end_date)
        + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    for ticker in RAW_UNIVERSE:

        print(
            f"Downloading {ticker}: "
            f"{download_start} -> {download_end}"
        )

        df = None

        # ----------------------------------------------------
        # Primary download
        # ----------------------------------------------------

        try:
            df = yf.download(
                ticker,
                start=download_start,
                end=download_end,
                auto_adjust=True,
                actions=False,
                progress=False,
                threads=False,
            )

        except Exception as e:
            print(
                f"WARNING: yf.download failed "
                f"for {ticker}: {e}"
            )

        # ----------------------------------------------------
        # Fallback to Ticker.history()
        # ----------------------------------------------------

        if df is None or df.empty:

            print(
                f"Retrying {ticker} using "
                f"Ticker.history()..."
            )

            try:
                df = yf.Ticker(
                    ticker
                ).history(
                    start=download_start,
                    end=download_end,
                    auto_adjust=True,
                    actions=False,
                )

            except Exception as e:
                raise ValueError(
                    f"Could not download usable "
                    f"data for {ticker}: {e}"
                )

        # ----------------------------------------------------
        # Clean and validate
        # ----------------------------------------------------

        try:
            cleaned = _clean_download(df)

        except Exception as e:
            raise ValueError(
                f"No usable market data for "
                f"{ticker}: {e}"
            )

        GLOBAL_DATA_CACHE[
            ANONYMOUS_MAP[ticker]
        ] = cleaned

        print(
            f"Loaded {ticker}: "
            f"{len(cleaned)} rows"
        )


# ============================================================
# COMMON TRADING DAYS
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

        sets.append({
            d.strftime("%Y-%m-%d")
            for d in idx
            if (
                start_date
                <= d.strftime("%Y-%m-%d")
                <= end_date
            )
        })

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
# ALLOCATION NORMALIZATION
# ============================================================

def normalize_allocations(raw):

    clean = {
        a: max(
            0.0,
            float(
                raw.get(a, 0.0)
            ),
        )
        for a in ANONYMOUS_UNIVERSE
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
            a: 0.0
            for a in ANONYMOUS_UNIVERSE
        } | {
            "CASH": 100.0
        }

    return {
        k: round(
            v / total * 100.0,
            6,
        )
        for k, v in clean.items()
    }


# ============================================================
# QWEN OUTPUT PARSER
# ============================================================

def parse_allocations(text):

    pairs = re.findall(
        r"""["']?([A-Za-z0-9_]+)["']?\s*:\s*(-?\d+(?:\.\d+)?)""",
        text,
    )

    raw = {
        k: float(v)
        for k, v in pairs
        if (
            k in ANONYMOUS_UNIVERSE
            or k == "CASH"
        )
    }

    return (
        normalize_allocations(raw)
        if raw
        else None
    )


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
        a: (
            holdings[a]
            * prices[a]
            / value
            * 100.0
            if value > 0
            else 0.0
        )
        for a in ANONYMOUS_UNIVERSE
    }

    alloc["CASH"] = (
        cash / value * 100.0
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
# PURE QWEN PRICE-ONLY DECISION
# ============================================================

def DECISION_FUNCTION(
    current_date,
    state,
    prices,
    rebalance_days,
):

    # ========================================================
    # PURE BASELINE
    #
    # Qwen receives:
    #   - current date
    #   - decision horizon
    #   - anonymous asset names
    #   - current price of each asset
    #   - current portfolio state
    #
    # Qwen does NOT receive:
    #   - historical prices
    #   - price arrays
    #   - moving averages
    #   - momentum
    #   - volatility
    #   - support/resistance
    #   - channels
    #   - Fibonacci
    #   - RSI
    #   - Bollinger Bands
    #   - technical signals
    #   - rule-based signals
    #   - any harness analysis
    #
    # The model simply sees prices and makes the decision.
    # ========================================================

    price_lines = "\n".join(
        f"{asset}: ${prices[asset]:.6f}"
        for asset in ANONYMOUS_UNIVERSE
    )

    allocs = ", ".join(
        f"{k}: {v:.6f}%"
        for k, v
        in state[
            "allocations_pct"
        ].items()
    )

    prompt = f"""You are the sole autonomous portfolio decision-maker.

Date: {current_date}
Decision horizon: {rebalance_days} trading days.

Assets:
{", ".join(ANONYMOUS_UNIVERSE)}

Current market prices:
{price_lines}

Current portfolio:
Portfolio Value: ${state["portfolio_value"]:,.2f}
Cash: {state["cash_pct"]:.6f}%
Current Allocations: {allocs}

Choose target allocations for the next {rebalance_days} trading days.

You are given only the information in this prompt.
Make the investment decision yourself.

Do not assume or invent information that is not provided.

Constraints:
- Long-only.
- No leverage.
- CASH is allowed.
- Every allocation must be >= 0%.
- Allocations must sum to 100%.
- Do not identify the real ticker names behind ASSET_*.
- Do not use future information.

Return ONLY one JSON object containing every ASSET_* key and CASH.

Example:
{{"ASSET_A": 10, "ASSET_B": 20, "ASSET_C": 0, "CASH": 70}}

Do not include markdown.
Do not include explanations.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
        max_tokens=800,
    )

    message = response.choices[0].message

    # Qwen3.6 may return content=None when the response is placed
    # in the reasoning field. Prevent the None.strip() crash.
    content = (message.content or "").strip()

    # If no normal content was returned, try the reasoning field.
    if not content:
        content = (getattr(message, "reasoning", None) or "").strip()

    parsed = parse_allocations(content)

    if parsed is None:

        return normalize_allocations({
            a: (
                100.0
                / len(
                    ANONYMOUS_UNIVERSE
                )
            )
            for a in ANONYMOUS_UNIVERSE
        } | {
            "CASH": 0.0
        })

    return parsed


# ============================================================
# OUTPUT CONFIGURATION
# ============================================================

VARIANT_NAME = "raw_qwen"

DEFAULT_OUTPUT = (
    "qwen_raw_results_compare.json"
)


# ============================================================
# REBALANCE EXECUTION
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
        a: (
            holdings[a]
            * prices[a]
            / value
            if value > 0
            else 0.0
        )
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
        value
        * turnover
        * FEE_RATE
    )

    execution_value = max(
        value - transaction_cost,
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
        a: (
            execution_value
            * target_w.get(
                a,
                0.0,
            )
            / prices[a]
            if prices[a] > 0
            else 0.0
        )
        for a in ANONYMOUS_UNIVERSE
    }

    orders = []

    for a in ANONYMOUS_UNIVERSE:

        delta = (
            target_w.get(
                a,
                0.0,
            )
            - current_w.get(
                a,
                0.0,
            )
        )

        if abs(delta) < 1e-12:
            continue

        notional = (
            abs(delta)
            * value
        )

        orders.append({
            "asset": REVERSE_MAP[a],
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
                notional * FEE_RATE,
                4,
            ),
        })

    return (
        new_holdings,
        new_cash,
        turnover,
        transaction_cost,
        orders,
    )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(daily_nav):

    nav = pd.Series([
        float(
            x["portfolio_value"]
        )
        for x in daily_nav
    ])

    dates = pd.to_datetime([
        x["date"]
        for x in daily_nav
    ])

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
        sd * np.sqrt(252.0)
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
            final / initial
        )
        ** (
            1.0 / years
        )
        - 1.0
        if initial > 0
        else 0.0
    )

    calmar = (
        cagr / abs(mdd)
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
# RESULT WRITER
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
        a: float(
            GLOBAL_DATA_CACHE[a]
            .loc[
                trading_days[0],
                "Close",
            ]
        )
        for a in ANONYMOUS_UNIVERSE
    }

    cash = 0.0

    holdings = {
        a: (
            initial_capital
            / len(
                ANONYMOUS_UNIVERSE
            )
            / first_prices[a]
        )
        for a in ANONYMOUS_UNIVERSE
    }

    target_allocs = (
        normalize_allocations({
            a: (
                100.0
                / len(
                    ANONYMOUS_UNIVERSE
                )
            )
            for a in ANONYMOUS_UNIVERSE
        } | {
            "CASH": 0.0
        })
    )

    daily_nav = []
    rebalances = []

    for idx, current_date in enumerate(
        trading_days
    ):

        prices = {
            a: float(
                GLOBAL_DATA_CACHE[a]
                .loc[
                    current_date,
                    "Close",
                ]
            )
            for a in ANONYMOUS_UNIVERSE
        }

        value = (
            cash
            + sum(
                holdings[a]
                * prices[a]
                for a in ANONYMOUS_UNIVERSE
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
                    holdings[a]
                    * prices[a]
                    for a in ANONYMOUS_UNIVERSE
                )
            )

            rebalances.append({
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
                        k,
                        k,
                    ): round(
                        float(v),
                        6,
                    )
                    for k, v
                    in target_allocs.items()
                },
            })

        value = (
            cash
            + sum(
                holdings[a]
                * prices[a]
                for a in ANONYMOUS_UNIVERSE
            )
        )

        daily_nav.append({
            "date": current_date,
            "prices": {
                REVERSE_MAP[a]: round(
                    float(p),
                    6,
                )
                for a, p
                in prices.items()
            },
            "portfolio_value": round(
                value,
                2,
            ),
            "harnessed_allocations": {
                REVERSE_MAP.get(
                    k,
                    k,
                ): round(
                    float(v),
                    6,
                )
                for k, v
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
        })

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
# COMMAND LINE
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
