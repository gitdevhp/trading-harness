import argparse
import json
import os
import re
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from openai import OpenAI

os.environ["YFINANCE_CACHE_DIR"] = "/tmp/yf_cache"
yf.set_tz_cache_location("/tmp/yf_tz_cache")

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-AWQ"

DEFAULT_UNIVERSE = ["GLD", "LLY", "KRE", "TSLA", "GOOGL", "TLT", "XLU", "XLE", "XOM", "NVDA"]

# Global containers dynamically populated at runtime
RAW_UNIVERSE = []
ANONYMOUS_MAP = {}
REVERSE_MAP = {}
ANONYMOUS_UNIVERSE = []
GLOBAL_DATA_CACHE = {}
POSITION_PEAKS = {}

def setup_universe(tickers: list):
    """Dynamically sets up asset mapping and resets global caches."""
    global RAW_UNIVERSE, ANONYMOUS_MAP, REVERSE_MAP, ANONYMOUS_UNIVERSE, GLOBAL_DATA_CACHE, POSITION_PEAKS
    
    RAW_UNIVERSE = [t.upper() for t in tickers]
    ANONYMOUS_MAP = {ticker: f"ASSET_{chr(65+i)}" for i, ticker in enumerate(RAW_UNIVERSE)}
    REVERSE_MAP = {v: k for k, v in ANONYMOUS_MAP.items()}
    ANONYMOUS_UNIVERSE = list(ANONYMOUS_MAP.values())
    
    GLOBAL_DATA_CACHE.clear()
    POSITION_PEAKS.clear()

def prefetch_data(start_date: str, end_date: str):
    lookback_start = (pd.to_datetime(start_date) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    for ticker in RAW_UNIVERSE:
        df = yf.download(ticker, start=lookback_start, end=end_date, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.ffill()
        required_cols = [c for c in ("Open", "Close") if c in df.columns]
        if required_cols:
            df = df.dropna(subset=required_cols)
        GLOBAL_DATA_CACHE[ANONYMOUS_MAP[ticker]] = df



PORTFOLIO_PEAK_VALUE = 0.0

def calculate_technical_indicators(past_df: pd.DataFrame) -> dict:
    closes = past_df['Close'].astype(float)
    price = float(closes.iloc[-1])
    sma20 = float(closes.tail(20).mean()) if len(closes) >= 20 else price
    sma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else price
    sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else price
    def mom(n):
        if len(closes) <= n: return 0.0
        base = float(closes.iloc[-n-1])
        return ((price/base)-1)*100 if base else 0.0
    m20,m60,m120=mom(20),mom(60),mom(120)
    r=closes.pct_change().dropna().tail(60)
    vol=float(r.std(ddof=1)*np.sqrt(252)) if len(r)>=20 else 0.20
    neg=r[r<0]
    dvol=float(np.sqrt(np.mean(neg**2))*np.sqrt(252)) if len(neg)>=6 else max(vol*0.7,0.05)
    peak=float(closes.tail(252).max()) if len(closes)>=20 else price
    dd=price/peak-1 if peak>0 else 0.0
    momentum=float(np.clip(0.2*np.clip(m20/12,-1,1)+0.35*np.clip(m60/25,-1,1)+0.45*np.clip(m120/40,-1,1),-1,1))
    trend=float(np.clip(0.45*np.clip(((price/max(sma50,1e-9))-1)/0.15,-1,1)+0.55*np.clip(((sma50/max(sma200,1e-9))-1)/0.15,-1,1),-1,1))
    return {'price':round(price,2),'sma20':round(sma20,2),'sma50':round(sma50,2),'sma200':round(sma200,2),'momentum20_pct':round(m20,2),'momentum60_pct':round(m60,2),'momentum120_pct':round(m120,2),'annual_volatility':max(vol,0.04),'downside_volatility':max(dvol,0.025),'drawdown':float(dd),'momentum_score':momentum,'trend_score':trend}


def _portfolio_vol(targets: dict, current_date: str) -> float:
    frames=[]
    for a in ANONYMOUS_UNIVERSE:
        frames.append(GLOBAL_DATA_CACHE[a].loc[:current_date]['Close'].astype(float).pct_change().dropna().tail(126).rename(a))
    if not frames: return 0.20
    ret=pd.concat(frames,axis=1,join='inner').dropna()
    if len(ret)<40: return 0.20
    cov=ret.cov().values*252.0
    cov=0.75*cov+0.25*np.diag(np.diag(cov))
    w=np.array([max(0,float(targets.get(a,0)))/100 for a in ANONYMOUS_UNIVERSE])
    return float(np.sqrt(max(w@cov@w,1e-10)))


def _conviction_overlay(raw_allocations: dict,current_date: str) -> dict:
    tech={a:calculate_technical_indicators(GLOBAL_DATA_CACHE[a].loc[:current_date]) for a in ANONYMOUS_UNIVERSE}
    med=float(np.median([tech[a]['downside_volatility'] for a in ANONYMOUS_UNIVERSE])) if ANONYMOUS_UNIVERSE else 0.15
    med=max(med,0.05)
    out={}
    for a in ANONYMOUS_UNIVERSE:
        raw=max(0.0,float(raw_allocations.get(a,0.0)))
        if raw<=0: out[a]=0.0; continue
        t=tech[a]
        mult=1.0+0.10*t['momentum_score']+0.08*t['trend_score']
        mult*=float(np.clip(1.0-0.08*(t['downside_volatility']/med-1.0),0.84,1.08))
        if t['momentum_score']>0.45 and t['trend_score']>0.25 and t['drawdown']>-0.12: mult*=1.05
        if t['momentum_score']<-0.35 and t['trend_score']<-0.20: mult*=0.84
        elif t['momentum_score']<-0.15 and t['trend_score']<0: mult*=0.92
        out[a]=raw*float(np.clip(mult,0.75,1.25))
    return out


def _apply_risk_target(targets: dict,current_date: str) -> dict:
    vol=_portfolio_vol(targets,current_date)
    if vol<=0.215: return dict(targets)
    scale=float(np.clip(0.215/max(vol,0.05),0.80,1.0))
    return {a:max(0.0,float(targets.get(a,0.0)))*scale for a in ANONYMOUS_UNIVERSE}


def _apply_breadth_overlay(targets: dict,current_date: str) -> dict:
    tech={a:calculate_technical_indicators(GLOBAL_DATA_CACHE[a].loc[:current_date]) for a in ANONYMOUS_UNIVERSE}
    breadth=sum(tech[a]['price']>=tech[a]['sma200'] for a in ANONYMOUS_UNIVERSE)/max(len(ANONYMOUS_UNIVERSE),1)
    scale=1.0 if breadth>=0.55 else 0.97 if breadth>=0.40 else 0.90 if breadth>=0.25 else 0.80
    return {a:max(0.0,float(targets.get(a,0.0)))*scale for a in ANONYMOUS_UNIVERSE}


def _apply_conditional_stops(targets: dict,current_allocations: dict,current_date: str,prices: dict) -> dict:
    out=dict(targets)
    for a in ANONYMOUS_UNIVERSE:
        cw=float(current_allocations.get(a,0.0))
        if cw<=0.5: POSITION_PEAKS.pop(a,None); continue
        p=float(prices[a])
        POSITION_PEAKS[a]=max(p,float(POSITION_PEAKS.get(a,p)))
        dd=p/POSITION_PEAKS[a]-1 if POSITION_PEAKS[a]>0 else 0.0
        t=calculate_technical_indicators(GLOBAL_DATA_CACHE[a].loc[:current_date])
        stop=float(np.clip(1.35*t['annual_volatility'],0.10,0.22))
        breakdown=p<t['sma50'] and t['momentum60_pct']<-3 and t['trend_score']<-0.25
        severe=dd<=-max(0.22,stop*1.35)
        if dd<=-stop and (breakdown or severe): out[a]=0.0; POSITION_PEAKS.pop(a,None)
    return out


def _apply_portfolio_drawdown_guard(targets: dict,portfolio_value: float) -> dict:
    global PORTFOLIO_PEAK_VALUE
    PORTFOLIO_PEAK_VALUE=max(PORTFOLIO_PEAK_VALUE,portfolio_value)
    if PORTFOLIO_PEAK_VALUE<=0: return dict(targets)
    dd=portfolio_value/PORTFOLIO_PEAK_VALUE-1
    scale=1.0 if dd>-0.12 else 0.96 if dd>-0.18 else 0.88 if dd>-0.24 else 0.78 if dd>-0.30 else 0.65
    return {a:max(0.0,float(targets.get(a,0.0)))*scale for a in ANONYMOUS_UNIVERSE}


def apply_institutional_risk_harness(raw_allocations: dict,current_allocations: dict,current_date: str,holdings_prices: dict,portfolio_value: float=0.0,drift_threshold: float=2.5,max_asset_cap_pct: float=35.0) -> dict:
    # The LLM is the alpha engine. This harness is intentionally an overlay.
    targets=_conviction_overlay(raw_allocations,current_date)
    targets=_apply_risk_target(targets,current_date)
    targets=_apply_breadth_overlay(targets,current_date)
    targets=_apply_conditional_stops(targets,current_allocations,current_date,holdings_prices)
    if portfolio_value>0: targets=_apply_portfolio_drawdown_guard(targets,portfolio_value)
    for a in ANONYMOUS_UNIVERSE: targets[a]=min(max(0.0,float(targets.get(a,0.0))),max_asset_cap_pct)
    gross=sum(targets.get(a,0.0) for a in ANONYMOUS_UNIVERSE)
    if gross>100:
        f=100/gross
        for a in ANONYMOUS_UNIVERSE: targets[a]*=f
        gross=100.0
    targets['CASH']=round(max(0.0,100-gross),2)
    targets={k:round(float(v),2) for k,v in targets.items()}
    drift=max([abs(targets.get(a,0.0)-current_allocations.get(a,0.0)) for a in ANONYMOUS_UNIVERSE],default=0.0)
    forced=any(current_allocations.get(a,0.0)-targets.get(a,0.0)>=8 for a in ANONYMOUS_UNIVERSE)
    return current_allocations if drift<drift_threshold and not forced else targets

def get_market_screener(current_date: str) -> str:
    screener = []
    for asset in ANONYMOUS_UNIVERSE:
        closes = GLOBAL_DATA_CACHE[asset].loc[:current_date]["Close"]
        cp = float(closes.iloc[-1])
        sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else cp
        mom120 = ((cp - float(closes.iloc[-120])) / float(closes.iloc[-120])) * 100.0 if len(closes) >= 120 else 0.0
        screener.append(f"{asset}: Price=${cp:.2f} | 200d-SMA=${sma200:.2f} | 120d-Mom={mom120:.1f}%")
    return "\n".join(screener)

def run_react_agent(current_date: str, portfolio_state: dict) -> dict:
    def get_portfolio_status(arg: str = "") -> str:
        alloc_str = ", ".join([f"{k}: {v:.1f}%" for k, v in portfolio_state["allocations_pct"].items()])
        return f"Portfolio Value: ${portfolio_state['portfolio_value']:,.2f} | Cash: {portfolio_state['cash_pct']:.1f}%\nAllocations: {alloc_str}"

    def tool_screener(arg: str = "") -> str:
        return get_market_screener(current_date)

    available_tools = {"get_market_screener": tool_screener, "get_portfolio_status": get_portfolio_status}

    system_prompt = f"""You are an autonomous ReAct Portfolio Manager on {current_date}.
Assets: {ANONYMOUS_UNIVERSE} + CASH

Primary objective:
Maximize long-term compounded return while maintaining strong risk-adjusted performance.
Prioritize CAGR/profit first, then Sortino, Calmar, and Sharpe, while controlling maximum drawdown.
Do not default to cash or minimum-volatility portfolios.

Decision principles:
- Use strong conviction when price, momentum, trend, and portfolio diversification support it.
- Do not punish an asset merely because it is volatile; distinguish upside volatility from harmful downside risk.
- Favor asymmetric upside and improving trend/momentum.
- Avoid chasing assets solely because they recently rose sharply.
- Avoid buying an asset solely because it is down.
- Treat CASH as a tactical allocation, not a default safety allocation.
- Think about the whole portfolio: expected return, correlation, concentration, downside risk, and drawdown.
- When two opportunities are similar, prefer the one with better Sortino/Calmar and lower drawdown.

Tools:
- get_market_screener[]
- get_portfolio_status[]

Format:
Thought: <Reasoning step>
Action: <tool_name>[]
Observation: <tool response>
...
Thought: <Final allocation decision>
Action: Target_Allocations[{{\"ASSET_A\": 15, \"ASSET_B\": 15, ..., \"CASH\": 10}}]"""


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Date: {current_date}. Analyze market conditions and set target allocations."}
    ]

    raw_decision = {"CASH": 100.0}

    for step in range(4):
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=0.0, max_tokens=600, stop=["Observation:"]
        )
        reply = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": reply})

        action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", reply, re.DOTALL)
        if action_match:
            action_name, action_arg = action_match.group(1), action_match.group(2).strip()

            if action_name == "Target_Allocations":
                parsed = re.findall(r'["\']?([A-Za-z0-9_]+)["\']?\s*:\s*([\d\.-]+)', action_arg)
                if parsed:
                    raw_decision = {k: float(v) for k, v in parsed}
                break

            if action_name in available_tools:
                obs_text = f"Observation: {available_tools[action_name](action_arg)}"
            else:
                obs_text = f"Observation: Tool '{action_name}' not found."
            messages.append({"role": "user", "content": obs_text})

    total = sum(raw_decision.values())
    return {k: round((v / total * 100.0), 2) if total > 0 else 0.0 for k, v in raw_decision.items()}

def run_backtest(
    start_date: str = "2023-03-15",
    end_date: str = "2026-04-01",
    initial_capital: float = 100000.0,
    tickers: list = None,
    output_file: str = "react_harness_results.json"
):
    if tickers is None or len(tickers) == 0:
        tickers = DEFAULT_UNIVERSE

    global PORTFOLIO_PEAK_VALUE
    setup_universe(tickers)
    PORTFOLIO_PEAK_VALUE = float(initial_capital)
    prefetch_data(start_date, end_date)

    trading_days = [
        d.strftime("%Y-%m-%d")
        for d in GLOBAL_DATA_CACHE[ANONYMOUS_UNIVERSE[0]].index
        if d.strftime("%Y-%m-%d") >= start_date
    ]

    cash = float(initial_capital)
    holdings = {t: 0.0 for t in ANONYMOUS_UNIVERSE}
    backtest_results = []

    last_raw_allocs = {t: 0.0 for t in ANONYMOUS_UNIVERSE}
    last_raw_allocs["CASH"] = 100.0
    pending_harnessed_targets = {"CASH": 100.0}

    for idx, current_date in enumerate(trading_days):
        close_prices = {
            t: float(GLOBAL_DATA_CACHE[t].loc[current_date]["Close"])
            for t in ANONYMOUS_UNIVERSE
        }

        total_value = cash + sum(
            holdings[t] * close_prices[t]
            for t in ANONYMOUS_UNIVERSE
        )

        allocations_pct = {
            t: (
                (holdings[t] * close_prices[t]) / total_value * 100.0
                if total_value > 0 else 0.0
            )
            for t in ANONYMOUS_UNIVERSE
        }
        allocations_pct["CASH"] = (
            cash / total_value * 100.0
            if total_value > 0 else 100.0
        )

        # Generate the signal from the current day's CLOSE.
        # The signal is deliberately NOT executed at this same close.
        if idx % 5 == 0 or idx == 0:
            portfolio_state = {
                "cash": cash,
                "cash_pct": allocations_pct["CASH"],
                "portfolio_value": total_value,
                "allocations_pct": allocations_pct,
            }
            raw_target_allocs = run_react_agent(
                current_date,
                portfolio_state,
            )
            last_raw_allocs = raw_target_allocs
        else:
            raw_target_allocs = last_raw_allocs

        # Harness the signal using information available at today's close.
        # Execution happens only on the NEXT trading day's OPEN.
        # At every close we compute the target for the next session.
        # Keep it pending until the next OPEN.
        if idx % 5 == 0 or idx == 0:
            pending_harnessed_targets = apply_institutional_risk_harness(
                raw_target_allocs,
                allocations_pct,
                current_date,
                close_prices,
            )

        harnessed_targets = pending_harnessed_targets

        if idx > 0 and (idx - 1) % 5 == 0:
            execution_prices = {
                t: float(GLOBAL_DATA_CACHE[t].loc[current_date]["Open"])
                for t in ANONYMOUS_UNIVERSE
            }

            execution_value = cash + sum(
                holdings[t] * execution_prices[t]
                for t in ANONYMOUS_UNIVERSE
            )

            total_harnessed_sum = sum(harnessed_targets.values())

            if total_harnessed_sum > 0:
                norm_targets = {
                    k: v / total_harnessed_sum
                    for k, v in harnessed_targets.items()
                }
            else:
                norm_targets = {t: 0.0 for t in ANONYMOUS_UNIVERSE}
                norm_targets["CASH"] = 1.0

            cash = execution_value * norm_targets.get("CASH", 0.0)

            for t in ANONYMOUS_UNIVERSE:
                p = execution_prices[t]
                holdings[t] = (
                    execution_value * norm_targets.get(t, 0.0) / p
                    if p > 0 else 0.0
                )

        new_value = cash + sum(
            holdings[t] * close_prices[t]
            for t in ANONYMOUS_UNIVERSE
        )

        real_executed = {
            REVERSE_MAP.get(k, k): v
            for k, v in harnessed_targets.items()
        }
        real_prices = {
            REVERSE_MAP[k]: v
            for k, v in close_prices.items()
        }

        backtest_results.append({
            "date": current_date,
            "prices": real_prices,
            "portfolio_value": round(new_value, 2),
            "harnessed_allocations": real_executed
        })

    with open(output_file, "w") as f:
        json.dump(backtest_results, f, indent=4)

    print(
        f"ReAct + Risk Harness Backtest Complete ({len(tickers)} Assets) -> Saved to {output_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ReAct + Risk Harness Backtest with a custom portfolio.")
    parser.add_argument("--tickers", nargs="+", help="List of stock tickers (e.g. AAPL NVDA MSFT AMZN)", default=DEFAULT_UNIVERSE)
    parser.add_argument("--start", type=str, default="2023-03-15", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-04-01", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="react_harness_results.json", help="Output JSON filename")

    args = parser.parse_args()

    run_backtest(
        start_date=args.start,
        end_date=args.end,
        tickers=args.tickers,
        output_file=args.output
    )