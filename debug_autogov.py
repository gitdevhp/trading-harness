"""
Quick debug script — runs 3 rebalances of the AutoGovern fast loop and
prints every debater verdict, direction, should_refine, should_store, and
the feedback text.

Usage (vLLM must be running):
    python debug_autogov.py

Look at the [autogov] lines to understand WHY admitted=0:
  - verdict=revise, g_ref=True  → loop continues (solver gets feedback)
  - verdict=revise, g_ref=False → loop exits immediately, admitted=False
  - verdict=accept              → admitted=True, loop breaks
"""
import os, sys

# Point at the local vLLM if it's running
os.environ.setdefault("ACE_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
os.environ.setdefault("ACE_SOLVER_MODEL",      "Qwen/Qwen2.5-32B-Instruct-AWQ")
os.environ.setdefault("ACE_DEBATER_MODEL",     "Qwen/Qwen2.5-32B-Instruct-AWQ")
os.environ.setdefault("ACE_CONSOLIDATOR_MODEL","Qwen/Qwen2.5-32B-Instruct-AWQ")

from ace_harness.market import MarketUniverse
from ace_harness.memory import ExperienceMemory
from ace_harness.agents import Solver, Debater, Consolidator, RiskTuner
from ace_harness.risk_harness import ConvictionHarness
from ace_harness import harnesses
from ace_harness.engine_monthly import run_backtest

# Small, fast universe — 7 stocks, 3 rebalances only
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
START = "2024-01-01"
END   = "2024-04-01"   # ~3 monthly rebalances
OUTPUT = "/tmp/debug_autogov_results.json"

print("=" * 60)
print("AutoGovern debug run — mag7, 3 rebalances")
print("Watch the [autogov] lines for per-round verdict details")
print("=" * 60)

universe    = MarketUniverse(TICKERS)
universe.prefetch(START, END)

solver      = Solver(universe)
debater     = Debater()
consolidator = Consolidator()
memory      = ExperienceMemory()
memory_path = "/tmp/debug_autogov_playbook.txt"

risk_harness = ConvictionHarness(universe)

decision_fn = harnesses.make_autogover(
    universe, solver, debater, consolidator, memory, memory_path,
    max_rounds=3,
    risk_harness=risk_harness,
)

run_backtest(universe, decision_fn, START, END, 1_000_000, OUTPUT, rebalance_days=20)

print("\n" + "=" * 60)
print(f"Done — full result at {OUTPUT}")
print("=" * 60)
