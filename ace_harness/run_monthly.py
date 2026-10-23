"""
Runs the self-improvement structures against engine_monthly.py — your
"fair portbench" script's exact protocol (same-close execution,
calendar-month rebalance, no warmup, equal-weight start, 15bps fees,
$1M default capital).

The Solver here uses prompt_style="monthly": its system prompt is
byte-for-byte the same as your pasted DECISION_FUNCTION (same wording,
same Rules, same "ReAct format" header, same invalid-allocation nudge
text) — see agents.py's Solver._build_prompt(). Both `get_market_screener`
and `get_portfolio_status` are available by default, matching every
version of your script (they always exposed both tools).

Systems (paper name -> --systems value):
  HARNESS          baseline
  HARNESS+DEBATER  intra
  HARNESS+MEMORY   memory_only
  DUAL             dual            (fixed ReMo as originally run)
  REMO             remo            (faithful ReMo Algorithm 1: store only admitted tasks)
  ADAP             adap_v2         (AutoGovern, outcome-grounded; see governance.py)
  DEBATER+STOP     debater_stop    (ADAP's stops without memory)

Every run writes <stem>_tokens.json, <stem>_manifest.json and
<stem>_decisions.jsonl next to <stem>_results.json (see accounting.py).

The RiskTuner (an LLM that retunes the risk harness monthly) applies only to
memory systems and is ON by default for backward compatibility; pass
--no_adaptive_risk for the clean ablation in which every system shares the
same fixed risk harness.

Usage (from the directory containing ace_harness/):

    python -m ace_harness.run_monthly --systems adap_v2 --no_adaptive_risk \
        --tickers AAPL MSFT NVDA AVGO AMD ADBE QCOM TXN AMAT MU AMZN TSLA PEP COST HON GOOGL META NFLX \
        --start 2024-01-01 --end 2024-12-31 --output_dir ./results
"""
import argparse
import os

from ace_harness.market import MarketUniverse
from ace_harness.engine_monthly import run_backtest
from ace_harness.memory import ExperienceMemory
from ace_harness.agents import Solver, Debater, Consolidator, RiskTuner
from ace_harness.risk_harness import (
    GPTInstitutionalRiskHarness, SimpleTrailingRiskHarness, SimpleMomentumHarness, ConvictionHarness,
)
from ace_harness import accounting, governance, harnesses
from ace_harness.autogov_v2 import make_autogover_v2
from ace_harness.llm_client import reset_token_counts

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ADBE", "QCOM", "TXN",
    "AMAT", "MU", "AMZN", "TSLA", "PEP", "COST", "HON", "GOOGL", "META", "NFLX",
]

_MEMORY_SYSTEMS = ("inter", "dual", "memory_only", "dual_permanent", "adap_v2", "remo")
_V2_PRESETS = {"adap_v2": "v2", "debater_stop": "debater_stop", "remo": "remo"}
SYSTEMS = ["baseline", "intra", "inter", "dual", "dual_session", "memory_only", "dual_permanent",
           *_V2_PRESETS]

_RISK_HARNESS_CLASSES = {
    "gpt": GPTInstitutionalRiskHarness,
    "simple": SimpleTrailingRiskHarness,
    "momentum": SimpleMomentumHarness,
    "conviction": ConvictionHarness,
}


def build_universe(tickers, start, end):
    universe = MarketUniverse(tickers)
    universe.prefetch(start, end)
    return universe


def execute(universe, decision_fn, start, end, initial_capital, output_file, rebalance_days,
            system, args, risk_harness=None, extra_manifest=None, label=None):
    """Run one backtest with full accounting: decision log, manifest, tokens."""
    paths = accounting.sidecar_paths(output_file)
    decision_fn = accounting.wrap_with_decision_log(decision_fn, paths["decisions"], universe, risk_harness)
    accounting.write_json(paths["manifest"],
                          accounting.build_manifest(output_file, system, args, universe, extra_manifest))
    reset_token_counts()
    run_backtest(universe, decision_fn, start, end, initial_capital, output_file, rebalance_days)
    tok = accounting.write_tokens_sidecar(output_file, decision_fn.per_task)["totals"]
    accounting.finalize_manifest(paths["manifest"], tokens=tok)
    print(f"[{label or system}] done -> {output_file}")
    print(f"[tokens]   calls={tok['calls']}  prompt={tok['prompt']:,}  "
          f"completion={tok['completion']:,}  total={tok['total']:,}")
    return output_file


def run_system(system, tickers, start, end, output_dir, initial_capital=1_000_000.0,
                rebalance_days=20, max_rounds=3, use_risk_harness=True, risk_harness_type="conviction",
                adaptive_risk=True, include_screener_tool=True, fallback_mode="equal_weight",
                truncate_context=False, max_tokens=800, gov_overrides=None, gov_tag=None,
                resume_memory=False, debater_sizing_context=False):
    run_args = dict(locals())
    if gov_overrides and system not in _V2_PRESETS:
        raise ValueError("--gov applies only to " + ", ".join(_V2_PRESETS))
    universe = build_universe(tickers, start, end)
    solver = Solver(universe, include_screener_tool=include_screener_tool, fallback_mode=fallback_mode,
                     max_tokens=max_tokens, truncate_context=truncate_context, prompt_style="monthly")
    debater = Debater()

    is_adaptive = use_risk_harness and adaptive_risk and system in _MEMORY_SYSTEMS

    tag = f"monthly_{system}" + (f"_{gov_tag}" if gov_tag else "")
    if not include_screener_tool:
        tag += "_noscreener"
    if use_risk_harness:
        tag += f"_{risk_harness_type}riskharness"
        tag += "_adaptive" if is_adaptive else ""
    output_file = os.path.join(output_dir, f"{tag}_results.json")
    paths = accounting.sidecar_paths(output_file)
    memory_path = os.path.join(output_dir, f"{tag}_playbook.txt")

    risk_harness_cls = _RISK_HARNESS_CLASSES[risk_harness_type]
    risk_harness = risk_harness_cls(universe) if use_risk_harness else None
    risk_params_path = os.path.join(output_dir, f"{tag}_riskparams.json") if use_risk_harness else None
    risk_tuner = RiskTuner() if is_adaptive else None
    extra = {"tag": tag, "risk_tuner": is_adaptive, "debater_sizing_context": debater_sizing_context}

    def memory_for_run():
        memory, info = accounting.prepare_memory(memory_path, resume=resume_memory,
                                                 evidence_path=paths["evidence"])
        extra.update(info)
        return memory

    if system == "baseline":
        decision_fn = harnesses.make_baseline(universe, solver)
        if risk_harness is not None:
            decision_fn = harnesses.wrap_with_risk_harness(decision_fn, risk_harness)

    elif system == "intra":
        decision_fn = harnesses.make_intra_task(universe, solver, debater, max_rounds=max_rounds,
                                                sizing_context=debater_sizing_context)
        if risk_harness is not None:
            decision_fn = harnesses.wrap_with_risk_harness(decision_fn, risk_harness)

    elif system == "inter":
        decision_fn = harnesses.make_inter_task(
            universe, solver, debater, Consolidator(), memory_for_run(), memory_path,
            risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
        )

    elif system == "dual":
        decision_fn = harnesses.make_dual_timescale(
            universe, solver, debater, Consolidator(), memory_for_run(), memory_path, max_rounds=max_rounds,
            risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
            sizing_context=debater_sizing_context,
        )

    elif system == "dual_session":
        decision_fn = harnesses.make_dual_session(
            universe, solver, debater, Consolidator(), max_rounds=max_rounds, risk_harness=risk_harness,
        )

    elif system == "memory_only":
        decision_fn = harnesses.make_memory_only(
            universe, solver, debater, Consolidator(), memory_for_run(), memory_path,
            risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
        )

    elif system == "dual_permanent":
        decision_fn = harnesses.make_dual_permanent(
            universe, solver, debater, Consolidator(), memory_for_run(), memory_path,
            risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
        )

    elif system in _V2_PRESETS:
        cfg = governance.make_config(_V2_PRESETS[system], gov_overrides, max_rounds=max_rounds,
                                     debater_sizing_context=debater_sizing_context, use_risk_tuner=is_adaptive)
        memory = memory_for_run() if cfg.memory == "on" else ExperienceMemory()
        evidence = governance.EvidenceStore.load(paths["evidence"]) if extra.get("resumed_memory") else None
        extra.update(governance=governance.config_dict(cfg), governance_hash=governance.config_hash(cfg),
                     governance_preset=_V2_PRESETS[system])
        decision_fn = make_autogover_v2(
            universe, solver, debater, Consolidator(), memory, memory_path, cfg,
            evidence=evidence, evidence_path=paths["evidence"],
            risk_harness=risk_harness, risk_tuner=risk_tuner, risk_params_path=risk_params_path,
        )

    else:
        raise ValueError(f"Unknown system: {system}")

    return execute(universe, decision_fn, start, end, initial_capital, output_file, rebalance_days,
                   system, run_args, risk_harness=risk_harness, extra_manifest=extra, label=tag)


def main():
    parser = argparse.ArgumentParser(description="Run self-improvement backtests on the monthly-rebalance engine")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=["dual_permanent"])
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_UNIVERSE)
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--initial_capital", type=float, default=1_000_000.0)
    parser.add_argument("--rebalance-days", type=int, default=20, dest="rebalance_days",
                         help="only affects the Solver's 'decision horizon' prompt text — the engine's actual "
                              "rebalance trigger is always the first trading day of each new calendar month")
    parser.add_argument("--max_rounds", type=int, default=3,
                         help="max Solver<->Debater rounds per task (intra/dual/dual_session/adap_v2/remo/"
                              "debater_stop — dual_permanent always does one review + at most one revision)")
    parser.add_argument("--no_risk_harness", action="store_true", help="disable the risk-management layer")
    parser.add_argument("--risk_harness_type", choices=["gpt", "simple", "momentum", "conviction"],
                         default="conviction",
                         help="'conviction' (default) matches your latest script's apply_institutional_harness")
    parser.add_argument("--no_adaptive_risk", action="store_true",
                         help="disable the LLM RiskTuner for memory systems (the clean ablation: every system "
                              "then shares the same fixed risk harness)")
    parser.add_argument("--no_screener", action="store_true",
                         help="drop the get_market_screener tool — OFF by default because every version of "
                              "your script always exposes both get_market_screener and get_portfolio_status")
    parser.add_argument("--fallback_mode", choices=["cash", "equal_weight"], default="equal_weight",
                         help="what the Solver falls back to if parsing ever fails (matches this script's "
                              "equal-weight fallback by default)")
    parser.add_argument("--truncate_context", action="store_true",
                         help="resend only [system, user, last_2_messages] each turn instead of full history "
                              "(off by default, matching your latest script, which sends full history)")
    parser.add_argument("--max_tokens", type=int, default=800,
                         help="per-turn output budget for the Solver (matches yesharnessgpt.py's max_tokens=800)")
    parser.add_argument("--gov", action="append", default=[], metavar="KEY=VALUE",
                         help="override one ADAP-v2 GovernanceConfig field (repeatable), e.g. "
                              "--gov exposure_mode=all; requires --gov_tag so ablations never overwrite ADAP")
    parser.add_argument("--gov_tag", type=str, default=None, help="suffix for the output tag of a --gov ablation")
    parser.add_argument("--debater_sizing_context", action="store_true",
                         help="give the Debater the universe-relative sizing block — applies to every debate "
                              "system (intra, dual, adap_v2, remo, debater_stop) alike")
    parser.add_argument("--resume_memory", action="store_true",
                         help="load an existing playbook from --output_dir. Only valid to warm-start a NEW "
                              "period from a finished run; by default a stale playbook is moved aside")
    args = parser.parse_args()

    overrides = dict(governance.parse_override(o) for o in args.gov)
    if overrides and not args.gov_tag:
        parser.error("--gov requires --gov_tag")

    os.makedirs(args.output_dir, exist_ok=True)
    for system in args.systems:
        run_system(
            system, args.tickers, args.start, args.end, args.output_dir,
            initial_capital=args.initial_capital, rebalance_days=args.rebalance_days, max_rounds=args.max_rounds,
            use_risk_harness=not args.no_risk_harness, risk_harness_type=args.risk_harness_type,
            adaptive_risk=not args.no_adaptive_risk, include_screener_tool=not args.no_screener,
            fallback_mode=args.fallback_mode, truncate_context=args.truncate_context, max_tokens=args.max_tokens,
            gov_overrides=overrides, gov_tag=args.gov_tag, resume_memory=args.resume_memory,
            debater_sizing_context=args.debater_sizing_context,
        )


if __name__ == "__main__":
    main()
