"""The six evaluation universes (identical to the shell drivers' lists).

    python -m ace_harness.universes sp30      # prints the tickers, space-separated
    python -m ace_harness.universes --all     # prints the union of all six
"""
import sys

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


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--all"
    print(" ".join(all_tickers() if arg == "--all" else UNIVERSES[arg]))
