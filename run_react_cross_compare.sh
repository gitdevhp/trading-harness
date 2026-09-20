#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/react_cross_compare_%j.out
#SBATCH --error=log/react_cross_compare_%j.out
#SBATCH --job-name=ReAct_CrossCompare
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=interactive-gpu

set -euo pipefail

# ============================================================
# CROSS-COMPARISON: both Qwen models × all 6 portfolio universes
#
# Runs qwen_port.py, noharness.py, yesharnessgpt.py for every
# (model, universe) combination, then plots each pair.
#
# Models:
#   qwen25  — Qwen2.5-32B-Instruct-AWQ   (1 × A40, TP=1)
#   qwen36  — Qwen3.6-35B-A3B-FP8        (2 × A40, TP=2)
#
# Universes:
#   tech18        — original 18 tech/mega-cap stocks
#   mag7          — Magnificent 7 (concentrated momentum)
#   balanced15    — diversified across tech/finance/health/consumer
#   sp30          — 30 S&P 500 stocks across all major GICS sectors
#   diversified40 — 40 stocks; max sector breadth (tech/energy/industrials/consumer)
#   volatile25    — 25 high-vol names (semis, cloud, biotech)
# ============================================================


# ============================================================
# BASIC SETUP
# ============================================================

mkdir -p log

module load gcc/11.3.0
module load miniforge

cd /users/2/cai00317/trading-harness/ReAct
source venv/bin/activate

mkdir -p results

export HF_HOME="$HOME/hf_cache"
export VLLM_USE_FLASHINFER_SAMPLER=0


# ============================================================
# PORTFOLIO UNIVERSES
# ============================================================

# Original 18 tech/mega-cap stocks used in baseline experiments
UNIVERSE_TECH18="AAPL MSFT NVDA AVGO AMD ADBE QCOM TXN AMAT MU AMZN TSLA PEP COST HON GOOGL META NFLX"

# Magnificent 7 — highly concentrated, 2024 momentum leaders
UNIVERSE_MAG7="AAPL MSFT NVDA AMZN GOOGL META TSLA"

# Balanced 15 — cross-sector: tech + financials + healthcare + consumer
UNIVERSE_BALANCED15="AAPL MSFT NVDA AMZN GOOGL META JPM JNJ XOM UNH HD WMT BAC PG CVX"

# 30 S&P 500 stocks spanning all major GICS sectors
UNIVERSE_SP30="AAPL MSFT NVDA AVGO AMD ADBE QCOM ORCL AMAT MU GOOGL META NFLX AMZN TSLA HD BKNG NKE WMT PG KO JNJ LLY UNH ABBV JPM V MA GS BAC"

# 40 stocks — maximum sector breadth: tech + energy + industrials + consumer
UNIVERSE_DIVERSIFIED40="AAPL MSFT NVDA AVGO AMD ADBE QCOM ORCL AMAT MU GOOGL META NFLX CMCSA AMZN TSLA HD BKNG NKE MCD WMT PG KO JNJ LLY UNH ABBV JPM V MA GS BAC XOM CVX COP CAT HON RTX PEP DIS"

# 25 high-volatility names: semis, cloud/SaaS, biotech
UNIVERSE_VOLATILE25="NVDA AMD AVGO QCOM MU AMAT MRVL CRWD PLTR DDOG TSLA META AMZN UBER SPOT ABNB NFLX LLY MRNA BIIB VRTX ABBV AMGN REGN PFE"

UNIVERSE_TAGS=("tech18" "mag7" "balanced15" "sp30" "diversified40" "volatile25")


# ============================================================
# BACKTEST CONFIGURATION
# ============================================================

INITIAL_CAPITAL=1000000
REBALANCE_DAYS=20
WARMUP_DAYS=0
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

    # Verify the correct model is actually being served before running any scripts.
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
    # Kill any remaining process still holding the port.
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
# PRINT CONFIGURATION
# ============================================================

echo "=========================================="
echo "ReAct CROSS-COMPARISON"
echo "=========================================="
echo "Models:         qwen25 (Qwen2.5-32B-AWQ)  |  qwen36 (Qwen3.6-35B-FP8)"
echo "Universes (6):  ${UNIVERSE_TAGS[*]}"
echo "Scripts:        qwen_port  noharness  yesharnessgpt"
echo "Start:          ${START_DATE}"
echo "End:            ${END_DATE}"
echo "Capital:        \$${INITIAL_CAPITAL}"
echo "Rebalance days: ${REBALANCE_DAYS}"
echo "Warmup days:    ${WARMUP_DAYS}"
echo "Transaction:    15 bps"
echo "Output dir:     results/"
echo "=========================================="
echo ""


# ============================================================
# MAIN LOOP — iterate over models, then universes
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

    echo "=========================================="
    echo "MODEL: ${MODEL}  (tag: qwen${MODEL_TAG})"
    echo "=========================================="

    start_vllm "${VLLM_ARGS[@]}"

    for UNIVERSE_TAG in "${UNIVERSE_TAGS[@]}"; do
        TICKERS=$(get_universe_tickers "$UNIVERSE_TAG")
        read -ra TICKER_ARRAY <<< "$TICKERS"

        RAW_OUT="results/qwen_raw_${MODEL_TAG}_${UNIVERSE_TAG}.json"
        NOHARN_OUT="results/react_no_harness_${MODEL_TAG}_${UNIVERSE_TAG}.json"
        GPT_OUT="results/react_gpt_harness_${MODEL_TAG}_${UNIVERSE_TAG}.json"

        echo ""
        echo "==========================================";
        echo "Universe: ${UNIVERSE_TAG}  |  Model: qwen${MODEL_TAG}"
        echo "Tickers:  ${TICKERS}"
        echo "=========================================="

        echo ""
        echo "--- Step 1/3: Raw Qwen ---"
        python qwen_port.py \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
            --initial-capital  "$INITIAL_CAPITAL" \
            --rebalance-days   "$REBALANCE_DAYS" \
            --warmup-days      "$WARMUP_DAYS" \
            --output  "$RAW_OUT"
        echo "Raw Qwen done → ${RAW_OUT}"

        echo ""
        echo "--- Step 2/3: ReAct No Harness ---"
        python noharness.py \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
            --initial-capital  "$INITIAL_CAPITAL" \
            --rebalance-days   "$REBALANCE_DAYS" \
            --warmup-days      "$WARMUP_DAYS" \
            --output  "$NOHARN_OUT"
        echo "ReAct No Harness done → ${NOHARN_OUT}"

        echo ""
        echo "--- Step 3/3: ReAct + GPT Harness ---"
        python yesharnessgpt.py \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
            --initial-capital  "$INITIAL_CAPITAL" \
            --rebalance-days   "$REBALANCE_DAYS" \
            --warmup-days      "$WARMUP_DAYS" \
            --output  "$GPT_OUT"
        echo "ReAct + GPT Harness done → ${GPT_OUT}"

        echo ""
        echo "Universe ${UNIVERSE_TAG} / qwen${MODEL_TAG} complete."
    done

    stop_vllm

done


# ============================================================
# PLOTTING — one comparison chart per (model, universe)
# ============================================================

echo ""
echo "=========================================="
echo "Generating comparison plots..."
echo "=========================================="

for MODEL_TAG in 25 36; do
    for UNIVERSE_TAG in "${UNIVERSE_TAGS[@]}"; do
        RAW_OUT="results/qwen_raw_${MODEL_TAG}_${UNIVERSE_TAG}.json"
        NOHARN_OUT="results/react_no_harness_${MODEL_TAG}_${UNIVERSE_TAG}.json"
        GPT_OUT="results/react_gpt_harness_${MODEL_TAG}_${UNIVERSE_TAG}.json"
        PLOT_OUT="results/compare_${MODEL_TAG}_${UNIVERSE_TAG}.png"

        echo "Plotting qwen${MODEL_TAG} / ${UNIVERSE_TAG} → ${PLOT_OUT}"
        python plotport.py \
            --gpt-harness-file "$GPT_OUT" \
            --no-harness-file  "$NOHARN_OUT" \
            --raw-llm-file     "$RAW_OUT" \
            --output           "$PLOT_OUT" \
        || echo "WARNING: plot failed for qwen${MODEL_TAG}/${UNIVERSE_TAG}"
    done
done


# ============================================================
# SUMMARY
# ============================================================

echo ""
echo "=========================================="
echo "ALL RUNS COMPLETE"
echo "=========================================="
echo "Capital:    \$${INITIAL_CAPITAL}"
echo "Period:     ${START_DATE} → ${END_DATE}"
echo ""
echo "Result files:"
ls -lh results/ 2>/dev/null || echo "  (results directory empty)"
echo "=========================================="
