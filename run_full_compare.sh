#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/full_compare_%j.out
#SBATCH --error=log/full_compare_%j.out
#SBATCH --job-name=FullCompare
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=mhong

set -euo pipefail

# ============================================================
# FULL COMPARATIVE RUN
#
# Single submission that fills every cell of the research table:
#
#   Systems (× 2 models each):
#     QWEN                  — qwen_port.py          (raw LLM, no ReAct)
#     QWEN+REACT            — noharness.py           (ReAct loop, no harness)
#     HARNESS               — yesharnessgpt.py       (ReAct + ConvictionHarness)
#     HARNESS+DEBATER       — run_monthly.py intra    (Debater, no Memory)
#     HARNESS+MEMORY        — run_monthly.py memory_only
#     HARNESS+MEMORY+DEBATER— run_monthly.py dual_permanent
#
#   Baselines (model-independent, built from embedded prices):
#     Equal-Weight | Risk-Parity | 60/40
#     Min-Variance | Cov-Risk-Parity | Black-Litterman
#
#   Models:
#     qwen25 — Qwen2.5-32B-Instruct-AWQ  (TP=1, 1×A100)
#     qwen36 — Qwen3.6-35B-A3B-FP8       (TP=2, 2×A100)
#
#   Universes (6):
#     tech18       — original 18 tech/mega-cap stocks
#     mag7         — Magnificent 7 (concentrated momentum)
#     balanced15   — cross-sector: tech + finance + health + consumer
#     sp30         — 30 S&P 500 stocks across all major GICS sectors
#     diversified40— 40 stocks: max sector breadth
#     volatile25   — 25 high-vol semis/cloud/biotech
#
# Outputs:
#   ReAct/results/           — QWEN / QWEN+REACT / HARNESS result JSONs
#   ace_results/<tag>/       — HARNESS+DEBATER/MEMORY/DUAL result JSONs
#   summary/full_comparison.json   — every cell in JSON
#   summary/full_comparison.txt    — ASCII metrics table grouped by universe
# ============================================================


# ============================================================
# SETUP
# ============================================================

mkdir -p log

module load gcc/11.3.0
module load miniforge

REACT_DIR="/users/2/cai00317/trading-harness/ReAct"
ROOT_DIR="/users/2/cai00317/trading-harness"

cd "$REACT_DIR"
source venv/bin/activate

mkdir -p results

export HF_HOME="$HOME/hf_cache"
export VLLM_USE_FLASHINFER_SAMPLER=0


# ============================================================
# PORTFOLIO UNIVERSES
# ============================================================

# ── From run_react_cross_compare.sh ────────────────────────
# Original 18 tech/mega-cap stocks used in baseline experiments
UNIVERSE_TECH18="AAPL MSFT NVDA AVGO AMD ADBE QCOM TXN AMAT MU AMZN TSLA PEP COST HON GOOGL META NFLX"

# Magnificent 7 — concentrated, 2024 momentum leaders
UNIVERSE_MAG7="AAPL MSFT NVDA AMZN GOOGL META TSLA"

# Balanced 15 — cross-sector: tech + financials + healthcare + consumer
UNIVERSE_BALANCED15="AAPL MSFT NVDA AMZN GOOGL META JPM JNJ XOM UNH HD WMT BAC PG CVX"

# ── From run_broad_scale_compare.sh ────────────────────────
# 30 S&P 500 stocks spanning all major GICS sectors
UNIVERSE_SP30="AAPL MSFT NVDA AVGO AMD ADBE QCOM ORCL AMAT MU GOOGL META NFLX AMZN TSLA HD BKNG NKE WMT PG KO JNJ LLY UNH ABBV JPM V MA GS BAC"

# 40 stocks — maximum sector breadth
UNIVERSE_DIVERSIFIED40="AAPL MSFT NVDA AVGO AMD ADBE QCOM ORCL AMAT MU GOOGL META NFLX CMCSA AMZN TSLA HD BKNG NKE MCD WMT PG KO JNJ LLY UNH ABBV JPM V MA GS BAC XOM CVX COP CAT HON RTX PEP DIS"

# 25 high-volatility, high-beta names: semis, cloud/SaaS, biotech
UNIVERSE_VOLATILE25="NVDA AMD AVGO QCOM MU AMAT MRVL CRWD PLTR DDOG TSLA META AMZN UBER SPOT ABNB NFLX LLY MRNA BIIB VRTX ABBV AMGN REGN PFE"

UNIVERSE_TAGS=("tech18" "mag7" "balanced15" "sp30" "diversified40" "volatile25")


# ============================================================
# BACKTEST CONFIGURATION
# ============================================================

INITIAL_CAPITAL=1000000
REBALANCE_DAYS=20
START_DATE="2024-01-01"
END_DATE="2024-12-31"

VLLM_PORT=8000
VLLM_TIMEOUT=3600
VLLM_PID=""


# ============================================================
# vLLM LIFECYCLE HELPERS
# ============================================================

clear_port() {
    local pids
    pids=$(lsof -ti :"${VLLM_PORT}" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "WARNING: Port ${VLLM_PORT} in use (${pids}). Killing..."
        # shellcheck disable=SC2086
        kill -TERM $pids 2>/dev/null || true
        sleep 5
        pids=$(lsof -ti :"${VLLM_PORT}" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            # shellcheck disable=SC2086
            kill -9 $pids 2>/dev/null || true
            sleep 3
        fi
    fi
}

start_vllm() {
    clear_port
    echo "Starting vLLM..."
    python -m vllm.entrypoints.openai.api_server "$@" &
    VLLM_PID=$!
    echo "PID: ${VLLM_PID}"
    local start_time elapsed
    start_time=$(date +%s)
    while ! curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM died during startup."
            exit 1
        fi
        elapsed=$(( $(date +%s) - start_time ))
        if [ "$elapsed" -ge "$VLLM_TIMEOUT" ]; then
            echo "ERROR: vLLM startup timed out after ${elapsed}s."
            exit 1
        fi
        echo "Waiting... ${elapsed}s"
        sleep 5
    done
    echo "vLLM ready."
    local served_model
    served_model=$(curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" | \
        python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "unknown")
    if [ "$served_model" != "$REACT_MODEL" ]; then
        echo "ERROR: Expected '${REACT_MODEL}' but serving '${served_model}'. Aborting."
        stop_vllm
        exit 1
    fi
    echo "Verified: serving ${served_model}"
    echo ""
    nvidia-smi
    echo ""
}

stop_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM (PID ${VLLM_PID})..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
    local pids
    pids=$(lsof -ti :"${VLLM_PORT}" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Killing stale process(es) on port ${VLLM_PORT}: ${pids}"
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
        sleep 3
    fi
    echo "vLLM stopped."
}

cleanup() { echo ""; stop_vllm; }
trap cleanup EXIT INT TERM

get_tickers() {
    case "$1" in
        tech18)        echo "$UNIVERSE_TECH18" ;;
        mag7)          echo "$UNIVERSE_MAG7" ;;
        balanced15)    echo "$UNIVERSE_BALANCED15" ;;
        sp30)          echo "$UNIVERSE_SP30" ;;
        diversified40) echo "$UNIVERSE_DIVERSIFIED40" ;;
        volatile25)    echo "$UNIVERSE_VOLATILE25" ;;
        *) echo "ERROR: unknown universe tag '$1'" >&2; exit 1 ;;
    esac
}


# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

echo "=========================================="
echo "FULL COMPARATIVE RUN"
echo "=========================================="
echo "Models:     qwen25 (Qwen2.5-32B-AWQ)  |  qwen36 (Qwen3.6-35B-FP8)"
echo "Universes:  ${UNIVERSE_TAGS[*]}"
echo "Systems:    QWEN  QWEN+REACT  HARNESS  HARNESS+DEBATER  HARNESS+MEMORY  HARNESS+MEMORY+DEBATER"
echo "Baselines:  EW  Risk-Parity  60/40  Min-Var  Cov-RP  Black-Litterman  (from embedded prices)"
echo "Period:     ${START_DATE} -> ${END_DATE}"
echo "Capital:    \$${INITIAL_CAPITAL}  |  Fees: 15 bps  |  Rebalance: monthly"
echo "=========================================="
echo ""


# ============================================================
# MAIN LOOP — model × universe
# ============================================================

for MODEL_VERSION in qwen25 qwen36; do

    if [ "$MODEL_VERSION" = "qwen25" ]; then
        MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"
        MODEL_TAG="25"
        VLLM_ARGS=(
            --model "$MODEL"
            --host 127.0.0.1
            --port "$VLLM_PORT"
            --tensor-parallel-size 1
            --max-model-len 32768
            --gpu-memory-utilization 0.90
            --enable-chunked-prefill
            --enable-auto-tool-choice
            --tool-call-parser hermes
        )
    else
        MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
        MODEL_TAG="36"
        VLLM_ARGS=(
            --model "$MODEL"
            --host 127.0.0.1
            --port "$VLLM_PORT"
            --tensor-parallel-size 2
            --max-model-len 16384
            --gpu-memory-utilization 0.90
            --enable-chunked-prefill
            --language-model-only
            --reasoning-parser qwen3
            --enable-auto-tool-choice
            --tool-call-parser qwen3_coder
            --generation-config vllm
        )
    fi

    export REACT_MODEL="$MODEL"
    # Wire all ACE roles to the same vLLM server and model
    export ACE_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
    export ACE_SOLVER_MODEL="$MODEL"
    export ACE_DEBATER_MODEL="$MODEL"
    export ACE_CONSOLIDATOR_MODEL="$MODEL"

    echo "=========================================="
    echo "MODEL: ${MODEL}  (tag: qwen${MODEL_TAG})"
    echo "=========================================="

    start_vllm "${VLLM_ARGS[@]}"

    for UNIVERSE_TAG in "${UNIVERSE_TAGS[@]}"; do
        TICKERS=$(get_tickers "$UNIVERSE_TAG")
        read -ra TICKER_ARRAY <<< "$TICKERS"

        # ReAct output files
        RAW_OUT="results/qwen_raw_${MODEL_TAG}_${UNIVERSE_TAG}.json"
        NOHARN_OUT="results/react_no_harness_${MODEL_TAG}_${UNIVERSE_TAG}.json"
        GPT_OUT="results/react_gpt_harness_${MODEL_TAG}_${UNIVERSE_TAG}.json"

        # ACE output directory — fresh for each (model, universe) so memory never leaks
        ACE_DIR="${ROOT_DIR}/ace_results/qwen${MODEL_TAG}_${UNIVERSE_TAG}"
        rm -rf "$ACE_DIR"
        mkdir -p "$ACE_DIR"

        echo ""
        echo "=========================================="
        echo "Universe: ${UNIVERSE_TAG}  (${#TICKER_ARRAY[@]} tickers)  |  qwen${MODEL_TAG}"
        echo "Tickers:  ${TICKERS}"
        echo "=========================================="

        # ----------------------------------------------------------
        # Step 1/6: QWEN — raw LLM, no ReAct, no harness
        # ----------------------------------------------------------
        echo ""
        echo "--- 1/6: QWEN (raw) ---"
        python qwen_port.py \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
            --initial-capital  "$INITIAL_CAPITAL" \
            --rebalance-days   "$REBALANCE_DAYS" \
            --warmup-days      0 \
            --output  "$RAW_OUT"
        echo "Done -> ${RAW_OUT}"

        # ----------------------------------------------------------
        # Step 2/6: QWEN+REACT — ReAct loop, no harness
        # ----------------------------------------------------------
        echo ""
        echo "--- 2/6: QWEN+REACT (no harness) ---"
        python noharness.py \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
            --initial-capital  "$INITIAL_CAPITAL" \
            --rebalance-days   "$REBALANCE_DAYS" \
            --warmup-days      0 \
            --output  "$NOHARN_OUT"
        echo "Done -> ${NOHARN_OUT}"

        # ----------------------------------------------------------
        # Step 3/6: HARNESS — ReAct + ConvictionHarness
        # ----------------------------------------------------------
        echo ""
        echo "--- 3/6: HARNESS (ReAct + ConvictionHarness) ---"
        python yesharnessgpt.py \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
            --initial-capital  "$INITIAL_CAPITAL" \
            --rebalance-days   "$REBALANCE_DAYS" \
            --warmup-days      0 \
            --output  "$GPT_OUT"
        echo "Done -> ${GPT_OUT}"

        # ----------------------------------------------------------
        # Steps 4-6: ACE harness — Debater-only, Memory-only, Dual
        #
        # intra         = HARNESS+DEBATER       (Debater, no persistent Memory)
        # memory_only   = HARNESS+MEMORY        (Memory + reflection, no intra-task Debater)
        # dual_permanent= HARNESS+MEMORY+DEBATER (Memory + one Debater round per task)
        #
        # All three share the same ConvictionHarness as yesharnessgpt.py
        # and use engine_monthly.py's same-close, calendar-month protocol.
        # ----------------------------------------------------------
        echo ""
        echo "--- 4-6/6: ACE harness (HARNESS+DEBATER / HARNESS+MEMORY / HARNESS+MEMORY+DEBATER) ---"
        cd "$ROOT_DIR"
        python -m ace_harness.run_monthly \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
            --systems intra memory_only dual_permanent \
            --output_dir "$ACE_DIR" \
            --risk_harness_type conviction \
            --initial_capital "$INITIAL_CAPITAL" \
            --rebalance-days  "$REBALANCE_DAYS"
        cd "$REACT_DIR"
        echo "Done -> ${ACE_DIR}/"

        echo ""
        echo "Universe ${UNIVERSE_TAG} / qwen${MODEL_TAG} complete."
    done

    stop_vllm

done


# ============================================================
# PLOTTING — per (model, universe)
# ============================================================

echo ""
echo "=========================================="
echo "Generating per-universe plots..."
echo "=========================================="

for MODEL_TAG in 25 36; do
    for UNIVERSE_TAG in "${UNIVERSE_TAGS[@]}"; do
        TICKERS=$(get_tickers "$UNIVERSE_TAG")
        read -ra TICKER_ARRAY <<< "$TICKERS"

        RAW_OUT="results/qwen_raw_${MODEL_TAG}_${UNIVERSE_TAG}.json"
        NOHARN_OUT="results/react_no_harness_${MODEL_TAG}_${UNIVERSE_TAG}.json"
        GPT_OUT="results/react_gpt_harness_${MODEL_TAG}_${UNIVERSE_TAG}.json"
        REACT_PLOT="results/compare_react_${MODEL_TAG}_${UNIVERSE_TAG}.png"
        ACE_DIR="${ROOT_DIR}/ace_results/qwen${MODEL_TAG}_${UNIVERSE_TAG}"

        # 3-way ReAct comparison chart
        echo "ReAct plot: qwen${MODEL_TAG}/${UNIVERSE_TAG} -> ${REACT_PLOT}"
        python plotport.py \
            --gpt-harness-file "$GPT_OUT" \
            --no-harness-file  "$NOHARN_OUT" \
            --raw-llm-file     "$RAW_OUT" \
            --output           "$REACT_PLOT" \
        || echo "WARNING: ReAct plot failed for qwen${MODEL_TAG}/${UNIVERSE_TAG}"

        # ACE comparison chart + baselines (baselines auto-built from embedded prices)
        echo "ACE plot:   qwen${MODEL_TAG}/${UNIVERSE_TAG} -> ${ACE_DIR}/ace_comparison.png"
        cd "$ROOT_DIR"
        python -m ace_harness.plot_results \
            "${ACE_DIR}"/*.json \
            --output_dir "$ACE_DIR" \
            --tickers "${TICKER_ARRAY[@]}" \
            --start "$START_DATE" \
            --end   "$END_DATE" \
            --initial_capital "$INITIAL_CAPITAL" \
        || echo "WARNING: ACE plot failed for qwen${MODEL_TAG}/${UNIVERSE_TAG}"

        # ACE text metrics table
        python -m ace_harness.compare_results "${ACE_DIR}"/*.json \
        || echo "WARNING: compare_results failed for qwen${MODEL_TAG}/${UNIVERSE_TAG}"
        cd "$REACT_DIR"
    done
done


# ============================================================
# FINAL AGGREGATE SUMMARY
#
# Reads every result file (ReAct + ACE), computes baselines
# from embedded prices, and writes:
#   summary/full_comparison.json  — all (model, universe, system) rows
#   summary/full_comparison.txt   — ASCII table grouped by universe
# ============================================================

echo ""
echo "=========================================="
echo "Building full aggregate summary..."
echo "=========================================="

mkdir -p "${ROOT_DIR}/summary"

cd "$ROOT_DIR"
python -m ace_harness.summarize_full_run \
    --react_dir     "${REACT_DIR}/results" \
    --ace_base      "${ROOT_DIR}/ace_results" \
    --output_dir    "${ROOT_DIR}/summary" \
    --model_tags    25 36 \
    --universe_tags tech18 mag7 balanced15 sp30 diversified40 volatile25

echo ""
echo "Full summary -> ${ROOT_DIR}/summary/"


# ============================================================
# FINAL FILE LISTING
# ============================================================

echo ""
echo "=========================================="
echo "ALL RUNS COMPLETE"
echo "=========================================="
echo "Period:  ${START_DATE} -> ${END_DATE}"
echo "Capital: \$${INITIAL_CAPITAL}  |  Fees: 15 bps"
echo ""
echo "ReAct results (ReAct/results/):"
ls -lh "${REACT_DIR}/results/"*.json 2>/dev/null | awk '{print "  "$NF, $5}' \
    || echo "  (none)"
echo ""
echo "ACE result directories (ace_results/):"
ls -lhd "${ROOT_DIR}/ace_results"/qwen*/ 2>/dev/null | awk '{print "  "$NF}' \
    || echo "  (none)"
echo ""
echo "Summary files:"
ls -lh "${ROOT_DIR}/summary/" 2>/dev/null | awk '{print "  "$NF, $5}' \
    || echo "  (none)"
echo "=========================================="
