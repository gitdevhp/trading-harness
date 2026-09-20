#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/adamo_harness_%j.out
#SBATCH --error=log/adamo_harness_%j.out
#SBATCH --job-name=AdaMoHarness
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=mhong

set -euo pipefail

# ============================================================
# HARNESS + AdaReMo RUN
# Runs the conviction harness + adaptive reward model only —
# no memory, no debater. Clean ablation against the standard
# harness baseline.
#
# RUN_TAG — names the output directory: adamo_results_<tag>/
# ============================================================
RUN_TAG="adamo_v1"   # <-- change between submissions


# ============================================================
# SETUP
# ============================================================

mkdir -p log

module load gcc/11.3.0
module load miniforge

ROOT_DIR="/users/2/cai00317/trading-harness"
REACT_DIR="/users/2/cai00317/trading-harness/ReAct"

cd "$REACT_DIR"
source venv/bin/activate

export HF_HOME="$HOME/hf_cache"
export VLLM_USE_FLASHINFER_SAMPLER=0


# ============================================================
# PORTFOLIO UNIVERSES
# ============================================================

UNIVERSE_TECH18="AAPL MSFT NVDA AVGO AMD ADBE QCOM TXN AMAT MU AMZN TSLA PEP COST HON GOOGL META NFLX"
UNIVERSE_MAG7="AAPL MSFT NVDA AMZN GOOGL META TSLA"
UNIVERSE_BALANCED15="AAPL MSFT NVDA AMZN GOOGL META JPM JNJ XOM UNH HD WMT BAC PG CVX"
UNIVERSE_SP30="AAPL MSFT NVDA AVGO AMD ADBE QCOM ORCL AMAT MU GOOGL META NFLX AMZN TSLA HD BKNG NKE WMT PG KO JNJ LLY UNH ABBV JPM V MA GS BAC"
UNIVERSE_DIVERSIFIED40="AAPL MSFT NVDA AVGO AMD ADBE QCOM ORCL AMAT MU GOOGL META NFLX CMCSA AMZN TSLA HD BKNG NKE MCD WMT PG KO JNJ LLY UNH ABBV JPM V MA GS BAC XOM CVX COP CAT HON RTX PEP DIS"
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
        kill -TERM $pids 2>/dev/null || true
        sleep 5
        pids=$(lsof -ti :"${VLLM_PORT}" 2>/dev/null || true)
        if [ -n "$pids" ]; then
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
    if [ "$served_model" != "$EXPECT_MODEL" ]; then
        echo "ERROR: Expected '${EXPECT_MODEL}' but serving '${served_model}'. Aborting."
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
# SUMMARY
# ============================================================

echo "=========================================="
echo "HARNESS + AdaReMo RUN  [tag: ${RUN_TAG}]"
echo "=========================================="
echo "System:     Conviction harness + AdaReMo (no memory, no debater)"
echo "Models:     qwen25 (Qwen2.5-32B-AWQ)  |  qwen36 (Qwen3.6-35B-A3B-FP8)"
echo "Universes:  ${UNIVERSE_TAGS[*]}"
echo "Period:     ${START_DATE} -> ${END_DATE}"
echo "Capital:    \$${INITIAL_CAPITAL}  |  Fees: 15 bps  |  Rebalance: monthly"
echo "Output:     ${ROOT_DIR}/adamo_results_${RUN_TAG}/"
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
        )
    fi

    export EXPECT_MODEL="$MODEL"
    export ACE_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
    export ACE_SOLVER_MODEL="$MODEL"

    echo "=========================================="
    echo "MODEL: ${MODEL}  (tag: qwen${MODEL_TAG})"
    echo "=========================================="

    start_vllm "${VLLM_ARGS[@]}"

    for UNIVERSE_TAG in "${UNIVERSE_TAGS[@]}"; do
        TICKERS=$(get_tickers "$UNIVERSE_TAG")
        read -ra TICKER_ARRAY <<< "$TICKERS"

        OUT_DIR="${ROOT_DIR}/adamo_results_${RUN_TAG}/qwen${MODEL_TAG}_${UNIVERSE_TAG}"
        rm -rf "$OUT_DIR"
        mkdir -p "$OUT_DIR"

        echo ""
        echo "=========================================="
        echo "Universe: ${UNIVERSE_TAG}  (${#TICKER_ARRAY[@]} tickers)  |  qwen${MODEL_TAG}"
        echo "Tickers:  ${TICKERS}"
        echo "=========================================="

        cd "$ROOT_DIR"
        python -m ace_harness.run_monthly_baseline_adamo \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
            --output_dir "$OUT_DIR" \
            --risk_harness_type conviction \
            --initial_capital "$INITIAL_CAPITAL" \
            --rebalance-days  "$REBALANCE_DAYS" \
            --adamo_min_samples 2
        cd "$REACT_DIR"

        echo "Done -> ${OUT_DIR}/"
    done

    stop_vllm

done


# ============================================================
# FINAL LISTING
# ============================================================

echo ""
echo "=========================================="
echo "HARNESS + AdaReMo COMPLETE  [tag: ${RUN_TAG}]"
echo "=========================================="
echo "Period:  ${START_DATE} -> ${END_DATE}"
echo "Capital: \$${INITIAL_CAPITAL}  |  Fees: 15 bps"
echo ""
echo "Result directories:"
ls -lhd "${ROOT_DIR}/adamo_results_${RUN_TAG}"/qwen*/ 2>/dev/null | awk '{print "  "$NF}' \
    || echo "  (none)"
echo "=========================================="
