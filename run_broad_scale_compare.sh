#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/broad_scale_compare_%j.out
#SBATCH --error=log/broad_scale_compare_%j.out
#SBATCH --job-name=BroadScale_Compare
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=interactive-gpu

set -euo pipefail

# ============================================================
# BROAD-SCALE COMPARISON
#
# Runs ALL five comparison systems across broad, diverse universes
# designed to stress-test simpler approaches and give the ACE
# dual system maximum opportunity to show learned improvement:
#
# Systems (in order of increasing sophistication):
#   1. Raw Qwen          — qwen_port.py      (no ReAct, no harness)
#   2. ReAct             — noharness.py      (ReAct loop, no harness)
#   3. HARNESS           — yesharnessgpt.py  (ReAct + ConvictionHarness)
#   4. HARNESS+MEMORY    — run_monthly.py --systems memory_only
#   5. HARNESS+DEBATER   — run_monthly.py --systems dual_permanent
#
# Models:
#   qwen25  — Qwen2.5-32B-Instruct-AWQ   (1 × A40, TP=1)
#   qwen36  — Qwen3.6-35B-A3B-FP8        (2 × A40, TP=2)
#
# Universes (broader than baseline, chosen to stress simpler systems):
#   sp30         — 30 S&P 500 stocks across all major GICS sectors;
#                  includes defensive (staples, healthcare) + cyclical
#                  sectors where momentum rules often misfire
#   diversified40 — 40 stocks; adds Energy, Industrials, extended Comm
#                  and Consumer — maximum sector breadth, divergent 2024
#                  regimes across sectors test cross-sector learning
#   volatile25   — 25 high-vol names (semis, cloud, biotech); volatile
#                  signals make position-sizing decisions harder and give
#                  the Debater the most to work with
# ============================================================


# ============================================================
# BASIC SETUP
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

# 30 S&P 500 stocks spanning all major GICS sectors.
# Mixed tech + defensive + financial + healthcare — the sectors with
# divergent 2024 dynamics stress-test harness rules that assume uniform
# momentum signals.
UNIVERSE_SP30="AAPL MSFT NVDA AVGO AMD ADBE QCOM ORCL AMAT MU GOOGL META NFLX AMZN TSLA HD BKNG NKE WMT PG KO JNJ LLY UNH ABBV JPM V MA GS BAC"

# 40 stocks — adds Energy, Industrials, extended Communication and Consumer.
# Maximum sector breadth: tech vs energy vs utilities vs defensives all in
# the same portfolio forces the model to learn sector rotation, not just
# rank momentum leaders within a single sector.
UNIVERSE_DIVERSIFIED40="AAPL MSFT NVDA AVGO AMD ADBE QCOM ORCL AMAT MU GOOGL META NFLX CMCSA AMZN TSLA HD BKNG NKE MCD WMT PG KO JNJ LLY UNH ABBV JPM V MA GS BAC XOM CVX COP CAT HON RTX PEP DIS"

# 25 high-volatility, high-beta names: semis, cloud/SaaS, biotech.
# High per-asset vol means the ConvictionHarness position-sizing rules
# fire more often. The Debater's signal-checking and the Memory's learned
# sizing rules have the most impact here — simpler approaches that rely on
# static rules will misfire most on this universe.
UNIVERSE_VOLATILE25="NVDA AMD AVGO QCOM MU AMAT MRVL CRWD PLTR DDOG TSLA META AMZN UBER SPOT ABNB NFLX LLY MRNA BIIB VRTX ABBV AMGN REGN PFE"

UNIVERSE_TAGS=("sp30" "diversified40" "volatile25")


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
        echo "WARNING: Port ${VLLM_PORT} in use by PID(s) ${pids}. Killing..."
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
        echo "ERROR: Expected model '${REACT_MODEL}' but port ${VLLM_PORT} is serving '${served_model}'."
        echo "A stale vLLM process may still be running. Aborting."
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

cleanup() {
    echo ""
    stop_vllm
}
trap cleanup EXIT INT TERM

get_universe_tickers() {
    case "$1" in
        sp30)          echo "$UNIVERSE_SP30" ;;
        diversified40) echo "$UNIVERSE_DIVERSIFIED40" ;;
        volatile25)    echo "$UNIVERSE_VOLATILE25" ;;
        *) echo "ERROR: unknown universe tag '$1'" >&2; exit 1 ;;
    esac
}


# ============================================================
# PRINT CONFIGURATION
# ============================================================

echo "=========================================="
echo "BROAD-SCALE COMPARISON"
echo "=========================================="
echo "Models:      qwen25 (Qwen2.5-32B-AWQ)  |  qwen36 (Qwen3.6-35B-FP8)"
echo "Universes:   ${UNIVERSE_TAGS[*]}"
echo "Systems:     raw_qwen  react  harness  harness+memory  harness+debater"
echo "Start:       ${START_DATE}"
echo "End:         ${END_DATE}"
echo "Capital:     \$${INITIAL_CAPITAL}"
echo "Transaction: 15 bps"
echo "Output:      ReAct -> ReAct/results/  |  ACE -> ace_results/"
echo "=========================================="
echo ""


# ============================================================
# MAIN LOOP — models then universes
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

    # Wire the ACE harness to the same vLLM server and model.
    # All three roles (Solver, Debater, Consolidator) run on the same model
    # — overriding individual roles requires ACE_DEBATER_MODEL etc.
    export ACE_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
    export ACE_SOLVER_MODEL="$MODEL"
    export ACE_DEBATER_MODEL="$MODEL"
    export ACE_CONSOLIDATOR_MODEL="$MODEL"

    echo "=========================================="
    echo "MODEL: ${MODEL}  (tag: qwen${MODEL_TAG})"
    echo "=========================================="

    start_vllm "${VLLM_ARGS[@]}"

    for UNIVERSE_TAG in "${UNIVERSE_TAGS[@]}"; do
        TICKERS=$(get_universe_tickers "$UNIVERSE_TAG")
        read -ra TICKER_ARRAY <<< "$TICKERS"

        # Output paths — ReAct scripts write to ReAct/results/
        RAW_OUT="results/qwen_raw_${MODEL_TAG}_${UNIVERSE_TAG}.json"
        NOHARN_OUT="results/react_no_harness_${MODEL_TAG}_${UNIVERSE_TAG}.json"
        GPT_OUT="results/react_gpt_harness_${MODEL_TAG}_${UNIVERSE_TAG}.json"

        # ACE harness outputs — always in a fresh directory so memory
        # never leaks between runs.
        ACE_DIR="${ROOT_DIR}/ace_results/qwen${MODEL_TAG}_${UNIVERSE_TAG}"
        rm -rf "$ACE_DIR"
        mkdir -p "$ACE_DIR"

        echo ""
        echo "=========================================="
        echo "Universe: ${UNIVERSE_TAG}  |  Model: qwen${MODEL_TAG}"
        echo "Tickers (${#TICKER_ARRAY[@]}):  ${TICKERS}"
        echo "=========================================="

        # ----------------------------------------------------------
        # Step 1/5: Raw Qwen (no ReAct, no harness)
        # ----------------------------------------------------------
        echo ""
        echo "--- Step 1/5: Raw Qwen ---"
        python qwen_port.py \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
            --initial-capital  "$INITIAL_CAPITAL" \
            --rebalance-days   "$REBALANCE_DAYS" \
            --warmup-days      0 \
            --output  "$RAW_OUT"
        echo "Raw Qwen done -> ${RAW_OUT}"

        # ----------------------------------------------------------
        # Step 2/5: ReAct — no harness
        # ----------------------------------------------------------
        echo ""
        echo "--- Step 2/5: ReAct (no harness) ---"
        python noharness.py \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
            --initial-capital  "$INITIAL_CAPITAL" \
            --rebalance-days   "$REBALANCE_DAYS" \
            --warmup-days      0 \
            --output  "$NOHARN_OUT"
        echo "ReAct No Harness done -> ${NOHARN_OUT}"

        # ----------------------------------------------------------
        # Step 3/5: ReAct + ConvictionHarness (HARNESS baseline)
        # ----------------------------------------------------------
        echo ""
        echo "--- Step 3/5: ReAct + Harness ---"
        python yesharnessgpt.py \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
            --initial-capital  "$INITIAL_CAPITAL" \
            --rebalance-days   "$REBALANCE_DAYS" \
            --warmup-days      0 \
            --output  "$GPT_OUT"
        echo "ReAct + Harness done -> ${GPT_OUT}"

        # ----------------------------------------------------------
        # Steps 4-5: ACE harness systems (HARNESS+MEMORY and DUAL)
        #
        # run_monthly.py uses:
        #   - engine_monthly.py (same-close, calendar-month, 15bps, equal-weight)
        #   - ConvictionHarness (exact match to yesharnessgpt.py's harness)
        #   - Solver with prompt_style="monthly" (same wording as DECISION_FUNCTION)
        #   - max_tokens=800 (matches yesharnessgpt.py)
        #
        # baseline    = same logic as yesharnessgpt.py but no Memory/Debater
        # memory_only = adds persistent Experience Memory + RiskTuner
        # dual_permanent = adds one Debater round per task on top of memory
        # ----------------------------------------------------------
        echo ""
        echo "--- Steps 4-5: ACE harness (baseline / memory_only / dual_permanent) ---"
        cd "$ROOT_DIR"
        python -m ace_harness.run_monthly \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
            --systems baseline memory_only dual_permanent \
            --output_dir "$ACE_DIR" \
            --risk_harness_type conviction \
            --initial_capital "$INITIAL_CAPITAL" \
            --rebalance-days   "$REBALANCE_DAYS"
        cd "$REACT_DIR"
        echo "ACE harness done -> ${ACE_DIR}/"

        echo ""
        echo "Universe ${UNIVERSE_TAG} / qwen${MODEL_TAG} complete."
        echo ""
    done

    stop_vllm

done


# ============================================================
# PLOTTING
# ============================================================

echo ""
echo "=========================================="
echo "Generating comparison plots..."
echo "=========================================="

for MODEL_TAG in 25 36; do
    for UNIVERSE_TAG in "${UNIVERSE_TAGS[@]}"; do
        TICKERS=$(get_universe_tickers "$UNIVERSE_TAG")
        read -ra TICKER_ARRAY <<< "$TICKERS"

        # --- ReAct 3-way comparison (raw vs react vs harness) ---
        RAW_OUT="results/qwen_raw_${MODEL_TAG}_${UNIVERSE_TAG}.json"
        NOHARN_OUT="results/react_no_harness_${MODEL_TAG}_${UNIVERSE_TAG}.json"
        GPT_OUT="results/react_gpt_harness_${MODEL_TAG}_${UNIVERSE_TAG}.json"
        REACT_PLOT="results/compare_react_${MODEL_TAG}_${UNIVERSE_TAG}.png"

        echo "ReAct plot: qwen${MODEL_TAG} / ${UNIVERSE_TAG} -> ${REACT_PLOT}"
        python plotport.py \
            --gpt-harness-file "$GPT_OUT" \
            --no-harness-file  "$NOHARN_OUT" \
            --raw-llm-file     "$RAW_OUT" \
            --output           "$REACT_PLOT" \
        || echo "WARNING: ReAct plot failed for qwen${MODEL_TAG}/${UNIVERSE_TAG}"

        # --- ACE 3-system comparison + quantitative baselines ---
        ACE_DIR="${ROOT_DIR}/ace_results/qwen${MODEL_TAG}_${UNIVERSE_TAG}"
        echo "ACE plot:  qwen${MODEL_TAG} / ${UNIVERSE_TAG} -> ${ACE_DIR}/ace_metrics.png"
        cd "$ROOT_DIR"
        python -m ace_harness.plot_results \
            "${ACE_DIR}"/*.json \
            --output_dir "$ACE_DIR" \
            --tickers "${TICKER_ARRAY[@]}" \
            --start "$START_DATE" \
            --end   "$END_DATE" \
            --initial_capital "$INITIAL_CAPITAL" \
        || echo "WARNING: ACE plot failed for qwen${MODEL_TAG}/${UNIVERSE_TAG}"

        # --- ACE text metrics table ---
        echo "ACE metrics table:"
        python -m ace_harness.compare_results "${ACE_DIR}"/*.json \
        || echo "WARNING: ACE metrics failed for qwen${MODEL_TAG}/${UNIVERSE_TAG}"
        cd "$REACT_DIR"
    done
done


# ============================================================
# SUMMARY
# ============================================================

echo ""
echo "=========================================="
echo "ALL RUNS COMPLETE"
echo "=========================================="
echo "Period:   ${START_DATE} -> ${END_DATE}"
echo "Capital:  \$${INITIAL_CAPITAL}"
echo ""
echo "ReAct result files (ReAct/results/):"
ls -lh results/qwen_raw_*.json results/react_*.json results/compare_react_*.png 2>/dev/null \
    || echo "  (none found)"
echo ""
echo "ACE harness result directories (ace_results/):"
ls -lhd "${ROOT_DIR}/ace_results"/qwen*/ 2>/dev/null \
    || echo "  (none found)"
echo "=========================================="
