"""Market data + screener. Two upgrades folded in here (they benefit
every system, not just the new engine) versus the original version:
  - prefetch() now retries failed downloads (yf.download, then a
    Ticker.history() fallback) with backoff and robust column/index
    cleaning, instead of a single unguarded yf.download call — lifted
    from your "fair portbench" script's _download_ticker/_clean_download,
    generalized to keep both Open and Close (engine.py's T+1-open
    execution needs Open; engine_monthly.py's same-close execution only
    needs Close, but both are cheap to keep).
  - get_market_screener() now includes 20d/50d SMA alongside the
    existing 200d-SMA/1M-Mom/6M-Mom/1M-Vol, matching that script's
    richer build_market_snapshot().
"""
import os
import time
import numpy as np
import pandas as pd
import yfinance as yf

os.environ.setdefault("YFINANCE_CACHE_DIR", "/tmp/yf_cache")
yf.set_tz_cache_location("/tmp/yf_tz_cache")


def _clean_download(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize a yfinance download into a DatetimeIndex frame with
    Open and Close columns, handling MultiIndex output, tz-aware
    indexes, duplicate dates, and empty/malformed data."""
    if df is None or df.empty:
        raise ValueError(f"{ticker}: yfinance returned no rows.")

    if isinstance(df.columns, pd.MultiIndex):
        cols = {}
        for wanted in ("Open", "Close"):
            series = None
            if wanted in df.columns.get_level_values(0):
                candidate = df[wanted]
                series = candidate.iloc[:, 0] if isinstance(candidate, pd.DataFrame) else candidate
            elif wanted in df.columns.get_level_values(-1):
                candidate = df.xs(wanted, axis=1, level=-1)
                series = candidate.iloc[:, 0] if isinstance(candidate, pd.DataFrame) else candidate
            if series is not None:
                cols[wanted] = series
        if not cols:
            raise ValueError(f"{ticker}: MultiIndex download has no usable Open/Close.")
        df = pd.DataFrame(cols)
    else:
        keep = [c for c in ("Open", "Close") if c in df.columns]
        if not keep:
            raise ValueError(f"{ticker}: downloaded data is missing Open/Close.")
        df = df[keep].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[c for c in ("Close",) if c in df.columns])
    df = df.ffill()

    if df.empty:
        raise ValueError(f"{ticker}: no usable prices after cleaning.")
    return df


def _download_ticker(ticker: str, start_date: str, end_date: str, attempts: int = 4) -> pd.DataFrame:
    last_error = None
    for attempt in range(attempts):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True,
                              actions=False, progress=False, threads=False, timeout=30)
            return _clean_download(df, ticker)
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)

    for attempt in range(2):
        try:
            df = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True, actions=False)
            return _clean_download(df, ticker)
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(3)

    raise ValueError(f"{ticker}: unable to obtain usable market data after all attempts. Last error: {last_error}")


class MarketUniverse:
    def __init__(self, tickers):
        self.raw_universe = [t.upper() for t in tickers]
        self.anon_map = {t: f"ASSET_{chr(65 + i)}" for i, t in enumerate(self.raw_universe)}
        self.reverse_map = {v: k for k, v in self.anon_map.items()}
        self.anon_universe = list(self.anon_map.values())
        self.data_cache = {}

    def prefetch(self, start_date: str, end_date: str):
        lookback_start = (pd.to_datetime(start_date) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        download_end = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        failures = []
        for ticker in self.raw_universe:
            try:
                self.data_cache[self.anon_map[ticker]] = _download_ticker(ticker, lookback_start, download_end)
            except Exception as exc:
                failures.append(f"{ticker}: {exc}")
        if failures:
            raise RuntimeError("Failed to load required assets:\n" + "\n".join(f"  - {f}" for f in failures))

    def all_dates(self):
        return [d.strftime("%Y-%m-%d") for d in self.data_cache[self.anon_universe[0]].index]

    def common_trading_days(self, start_date: str = None, end_date: str = None) -> list:
        """Dates present in EVERY asset's calendar, not just the first
        one — different tickers can have slightly different available
        dates (holidays, halts, late IPOs), and indexing into another
        asset's frame on a date it doesn't have raises a KeyError. Both
        engines use this instead of all_dates() for exactly that reason."""
        sets = []
        for asset in self.anon_universe:
            dates = {d.strftime("%Y-%m-%d") for d in self.data_cache[asset].index}
            if start_date:
                dates = {d for d in dates if d >= start_date}
            if end_date:
                dates = {d for d in dates if d <= end_date}
            if not dates:
                raise ValueError(f"{self.reverse_map.get(asset, asset)}: no usable trading dates in requested period.")
            sets.append(dates)
        days = sorted(set.intersection(*sets))
        if not days:
            raise ValueError("No common trading days across the universe.")
        return days

    def get_market_screener(self, current_date: str) -> str:
        lines = []
        for asset in self.anon_universe:
            closes = self.data_cache[asset].loc[:current_date]["Close"]
            if closes.empty:
                raise ValueError(f"{self.reverse_map.get(asset, asset)}: no prices available through {current_date}.")
            cp = float(closes.iloc[-1])
            sma20 = float(closes.tail(20).mean()) if len(closes) >= 20 else cp
            sma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else cp
            sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else cp
            mom21 = ((cp - float(closes.iloc[-21])) / float(closes.iloc[-21])) * 100.0 if len(closes) >= 21 else 0.0
            mom126 = ((cp - float(closes.iloc[-126])) / float(closes.iloc[-126])) * 100.0 if len(closes) >= 126 else 0.0
            vol21 = float(closes.tail(21).pct_change().std() * np.sqrt(252) * 100.0) if len(closes) >= 21 else 0.0
            lines.append(
                f"{asset}: Price=${cp:.2f} | 20d-SMA=${sma20:.2f} | 50d-SMA=${sma50:.2f} | 200d-SMA=${sma200:.2f} | "
                f"1M-Mom={mom21:.2f}% | 6M-Mom={mom126:.2f}% | 1M-Vol={vol21:.2f}%"
            )
        return "\n".join(lines)

    def close_prices(self, current_date: str) -> dict:
        return {t: float(self.data_cache[t].loc[current_date]["Close"]) for t in self.anon_universe}

    def open_prices(self, current_date: str) -> dict:
        return {t: float(self.data_cache[t].loc[current_date]["Open"]) for t in self.anon_universe}

