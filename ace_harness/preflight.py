"""Check that Yahoo price data covers a backtest window before spending GPU time on it.

    python -m ace_harness.preflight 2025-01-01 2025-12-31

Downloads every ticker of the six universes with the downloader and the 365-day
lookback the ace systems use (market.MarketUniverse.prefetch). No LLM calls.
Exits 1 if any ticker would break or shorten the run:
  * the download fails (the run would crash at its first cell);
  * the data starts late or ends early: every engine trades on the intersection
    of the assets' trading days, so one short ticker silently shortens the whole
    universe's backtest;
  * fewer than 200 rows before the start (the screener's 200-day SMA).
It also prints each universe's common trading days and every one-day move above
15% inside the window (an unadjusted split or spin-off shows up there).

With ACE_PRICE_CACHE set to a campaign's OUT_ROOT/price_cache it checks, and
stores, the exact frames the ace cells of that campaign will then read:
    ACE_PRICE_CACHE=$PWD/campaign_2024/price_cache python -m ace_harness.preflight 2024-01-01 2024-12-31
If that reports PROBLEMS, the stored frames include the bad ones: delete that
price_cache directory before submitting anything to the campaign.
"""
import argparse
import sys

import pandas as pd

from ace_harness import price_cache
from ace_harness.market import _download_ticker
from ace_harness.universes import UNIVERSES

LOOKBACK_DAYS = 365      # market.MarketUniverse.prefetch
MIN_LOOKBACK_ROWS = 200  # longest indicator window
SLACK_DAYS = 6           # first/last trading day vs requested dates
BIG_MOVE = 0.15


def _ace_download(ticker, start, end):
    """What MarketUniverse.prefetch reads: the campaign snapshot when ACE_PRICE_CACHE is set."""
    return price_cache.fetch("ace", ticker, start, end, lambda: _download_ticker(ticker, start, end))


def check(start, end, download=_ace_download, universes=UNIVERSES):
    t0, t1 = pd.to_datetime(start), pd.to_datetime(end)
    lookback_start = (t0 - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    download_end = (t1 + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    lines, problems, windows = [], [], {}
    for ticker in sorted({t for tickers in universes.values() for t in tickers}):
        try:
            close = download(ticker, lookback_start, download_end)["Close"]
        except Exception as exc:  # noqa: BLE001 - report every failure, then exit 1
            problems.append(f"{ticker}: download failed: {exc}")
            continue
        win = close[(close.index >= t0) & (close.index <= t1)]
        n_before = int((close.index < t0).sum())
        if win.empty:
            problems.append(f"{ticker}: no prices in {start}..{end}")
            continue
        windows[ticker] = set(win.index)
        moves = win.pct_change().dropna()
        big = [(d.strftime("%Y-%m-%d"), round(r * 100, 1)) for d, r in moves.items() if abs(r) > BIG_MOVE]
        lines.append(f"{ticker:<6} {win.index[0].date()}..{win.index[-1].date()} days={len(win):>3} "
                     f"lookback={n_before:>3}" + (f"  moves>{BIG_MOVE:.0%}: {big}" if big else ""))
        if (win.index[0] - t0).days > SLACK_DAYS:
            problems.append(f"{ticker}: starts {win.index[0].date()}, after {start}")
        if (t1 - win.index[-1]).days > SLACK_DAYS:
            problems.append(f"{ticker}: ends {win.index[-1].date()}, before {end}; every universe holding it "
                            "would stop there")
        if n_before < MIN_LOOKBACK_ROWS:
            problems.append(f"{ticker}: only {n_before} rows before {start} (< {MIN_LOOKBACK_ROWS})")
    lines.append("")
    for name, tickers in universes.items():
        if all(t in windows for t in tickers):
            common = sorted(set.intersection(*(windows[t] for t in tickers)))
            span = f"{common[0].date()}..{common[-1].date()}" if common else "none"
            lines.append(f"{name:<14} common trading days={len(common)} {span}")
        else:
            lines.append(f"{name:<14} missing {[t for t in tickers if t not in windows]}")
    return lines, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("start")
    parser.add_argument("end")
    args = parser.parse_args()
    lines, problems = check(args.start, args.end)
    print("\n".join(lines))
    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print(f"\nOK: every ticker covers {args.start}..{args.end} with enough lookback.")


if __name__ == "__main__":
    main()
