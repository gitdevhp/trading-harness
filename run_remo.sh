#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/remo_%j.out
#SBATCH --error=log/remo_%j.out
#SBATCH --job-name=ReMo
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=mhong

set -euo pipefail

# ============================================================
# HARNESS + ReMo RUN
#
# Runs HARNESS+REMO (plain reward model, raw returns) for both
# models across all universes and produces per-universe plots.
#
#   System: HARNESS+REMO — run_monthly_baseline_remo
#
#   Models:
#     qwen25 — Qwen2.5-32B-Instruct-AWQ  (TP=1, 1×A100)
#     qwen36 — Qwen3.6-35B-A3B           (TP=4, 4×A100)
#
#   RUN_TAG — names remo_results_<tag>/ directory
# ============================================================
RUN_TAG="remo_v1"   # <-- change between submissions


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

export HF_HOME="$HOME/hf_cache"
export VLLM_USE_FLASHINFER_SAMPLER=0
# Qwen3.6 torch.compile + warmup can exceed vLLM's default 600s engine-ready timeout.
export VLLM_ENGINE_READY_TIMEOUT_S=1800


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
    local pids waited
    pids=$(lsof -ti :"${VLLM_PORT}" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "WARNING: Port ${VLLM_PORT} in use (${pids}). Killing..."
        # shellcheck disable=SC2086
        kill -TERM $pids 2>/dev/null || true
        sleep 10
        pids=$(lsof -ti :"${VLLM_PORT}" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            # shellcheck disable=SC2086
            kill -9 $pids 2>/dev/null || true
            sleep 5
        fi
    fi
    # Wait until port is actually free before returning.
    waited=0
    while lsof -ti :"${VLLM_PORT}" > /dev/null 2>&1; do
        sleep 2
        waited=$(( waited + 2 ))
        if [ "$waited" -ge 30 ]; then
            echo "WARNING: Port ${VLLM_PORT} still occupied after ${waited}s — forcing kill."
            lsof -ti :"${VLLM_PORT}" | xargs kill -9 2>/dev/null || true
            sleep 3
            break
        fi
    done
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
echo "HARNESS + ReMo RUN  [tag: ${RUN_TAG}]"
echo "=========================================="
echo "System:     HARNESS+REMO (plain reward model, raw returns)"
echo "Models:     qwen25 (Qwen2.5-32B-AWQ)  |  qwen36 (Qwen3.6-35B-A3B)"
echo "Universes:  ${UNIVERSE_TAGS[*]}"
echo "Period:     ${START_DATE} -> ${END_DATE}"
echo "Capital:    \$${INITIAL_CAPITAL}  |  Fees: 15 bps  |  Rebalance: monthly"
echo "Results:    ${ROOT_DIR}/remo_results_${RUN_TAG}/"
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
        MODEL="Qwen/Qwen3.6-35B-A3B"
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
    export ACE_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
    export ACE_SOLVER_MODEL="$MODEL"

    echo "=========================================="
    echo "MODEL: ${MODEL}  (tag: qwen${MODEL_TAG})"
    echo "=========================================="

    start_vllm "${VLLM_ARGS[@]}"

    for UNIVERSE_TAG in "${UNIVERSE_TAGS[@]}"; do
        TICKERS=$(get_tickers "$UNIVERSE_TAG")
        read -ra TICKER_ARRAY <<< "$TICKERS"

        REMO_DIR="${ROOT_DIR}/remo_results_${RUN_TAG}/qwen${MODEL_TAG}_${UNIVERSE_TAG}"
        rm -rf "$REMO_DIR"
        mkdir -p "$REMO_DIR"

        echo ""
        echo "=========================================="
        echo "Universe: ${UNIVERSE_TAG}  (${#TICKER_ARRAY[@]} tickers)  |  qwen${MODEL_TAG}"
        echo "Tickers:  ${TICKERS}"
        echo "=========================================="

        cd "$ROOT_DIR"
        echo "--- HARNESS+REMO (reward signal on top of plain harness) ---"
        python -m ace_harness.run_monthly_baseline_remo \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
            --output_dir "$REMO_DIR" \
            --risk_harness_type conviction \
            --initial_capital "$INITIAL_CAPITAL" \
            --rebalance-days  "$REBALANCE_DAYS" \
            --remo_min_samples 2
        cd "$REACT_DIR"
        echo "Done -> ${REMO_DIR}/monthly_baseline_remo_*"

        echo ""
        echo "Universe ${UNIVERSE_TAG} / qwen${MODEL_TAG} complete."
    done

    stop_vllm

done


# ============================================================
# PLOTTING
# ============================================================

echo ""
echo "=========================================="
echo "Generating per-universe plots..."
echo "=========================================="

for MODEL_TAG in 25 36; do
    for UNIVERSE_TAG in "${UNIVERSE_TAGS[@]}"; do
        TICKERS=$(get_tickers "$UNIVERSE_TAG")
        read -ra TICKER_ARRAY <<< "$TICKERS"

        REMO_DIR="${ROOT_DIR}/remo_results_${RUN_TAG}/qwen${MODEL_TAG}_${UNIVERSE_TAG}"

        echo "Plot: qwen${MODEL_TAG}/${UNIVERSE_TAG} -> ${REMO_DIR}/ace_comparison.png"
        cd "$ROOT_DIR"
        python -m ace_harness.plot_results \
            "${REMO_DIR}"/*.json \
            --output_dir "$REMO_DIR" \
            --tickers "${TICKER_ARRAY[@]}" \
            --start "$START_DATE" \
            --end   "$END_DATE" \
            --initial_capital "$INITIAL_CAPITAL" \
        || echo "WARNING: plot failed for qwen${MODEL_TAG}/${UNIVERSE_TAG}"

        python -m ace_harness.compare_results "${REMO_DIR}"/*.json \
        || echo "WARNING: compare_results failed for qwen${MODEL_TAG}/${UNIVERSE_TAG}"
        cd "$REACT_DIR"
    done
done


# ============================================================
# FINAL FILE LISTING
# ============================================================

echo ""
echo "=========================================="
echo "HARNESS + ReMo RUN COMPLETE  [tag: ${RUN_TAG}]"
echo "=========================================="
echo "Period:  ${START_DATE} -> ${END_DATE}"
echo "Capital: \$${INITIAL_CAPITAL}  |  Fees: 15 bps"
echo ""
echo "Result directories (remo_results_${RUN_TAG}/):"
ls -lhd "${ROOT_DIR}/remo_results_${RUN_TAG}"/qwen*/ 2>/dev/null | awk '{print "  "$NF}' \
    || echo "  (none)"
echo "=========================================="
