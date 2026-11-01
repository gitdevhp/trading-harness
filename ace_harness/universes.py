"""The six evaluation universes (identical to the shell drivers' lists).

    python -m ace_harness.universes sp30            # prints the tickers, space-separated
    python -m ace_harness.universes sp30 --seed 2   # the same tickers in a fixed shuffled order
    python -m ace_harness.universes --all           # prints the union of all six

Every system labels assets by their position in the ticker list (ASSET_A is the
first ticker, and the screener lists them in that order). Greedy decoding makes a
rerun on the same inputs bit-identical, so perturbation replicates (see
run_run1_refresh.sh, LABEL_SEED) reorder the tickers: the same information under
different anonymous labels and listing order. Seed 0 is the canonical order.
"""
import argparse
import random

UNIVERSES = {
    "tech18": "AAPL MSFT NVDA AVGO AMD ADBE QCOM TXN AMAT MU AMZN TSLA PEP COST HON GOOGL META NFLX".split(),
    "mag7": "AAPL MSFT NVDA AMZN GOOGL META TSLA".split(),
    "balanced15": "AAPL MSFT NVDA AMZN GOOGL META JPM JNJ XOM UNH HD WMT BAC PG CVX".split(),
    "sp30": ("AAPL MSFT NVDA AVGO AMD ADBE QCOM ORCL AMAT MU GOOGL META NFLX AMZN TSLA HD BKNG NKE WMT PG "
             "KO JNJ LLY UNH ABBV JPM V MA GS BAC").split(),
    "diversified40": ("AAPL MSFT NVDA AVGO AMD ADBE QCOM ORCL AMAT MU GOOGL META NFLX CMCSA AMZN TSLA HD BKNG "
                      "NKE MCD WMT PG KO JNJ LLY UNH ABBV JPM V MA GS BAC XOM CVX COP CAT HON RTX PEP DIS").split(),
    "volatile25": ("NVDA AMD AVGO QCOM MU AMAT MRVL CRWD PLTR DDOG TSLA META AMZN UBER SPOT ABNB NFLX LLY MRNA "
                   "BIIB VRTX ABBV AMGN REGN PFE").split(),
}


def all_tickers():
    return sorted({t for tickers in UNIVERSES.values() for t in tickers})


def ordered(name, seed=0):
    """The universe's tickers; seed > 0 gives a fixed permutation of them."""
    tickers = list(UNIVERSES[name])
    if seed:
        random.Random(f"{seed}:{name}").shuffle(tickers)
    return tickers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("universe", nargs="?", choices=sorted(UNIVERSES))
    parser.add_argument("--all", action="store_true", help="the union of all six (the default)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(" ".join(all_tickers() if args.all or not args.universe else ordered(args.universe, args.seed)))
