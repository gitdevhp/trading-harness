#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/ace_only_%j.out
#SBATCH --error=log/ace_only_%j.out
#SBATCH --job-name=AceOnly
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=mhong

set -euo pipefail

# ============================================================
# ACE-ONLY RUN — re-runs only the three ACE systems so you can
# iterate on memory / Debater / dual without re-running the
# ReAct baselines (which are slow and already stable).
#
# Two tags:
#   ACE_RUN_TAG    — names the new ace_results_<tag>/ and
#                    summary_<tag>/ directories for THIS run
#   REACT_RUN_TAG  — points at the existing ReAct results
#                    directory to use in the final summary
#                    (set this to whichever previous run tag
#                    has good ReAct results, e.g. "2" or "fixed")
# ============================================================
ACE_RUN_TAG="ace_fixed"   # <-- change between ACE-only submissions
REACT_RUN_TAG="2"         # <-- tag of the existing ReAct results to reuse


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

# Existing ReAct results — read-only, not regenerated
REACT_RESULTS_DIR="${REACT_DIR}/results_${REACT_RUN_TAG}"
if [ ! -d "$REACT_RESULTS_DIR" ]; then
    echo "ERROR: REACT_RESULTS_DIR does not exist: ${REACT_RESULTS_DIR}"
    echo "Set REACT_RUN_TAG to a tag that has completed ReAct results."
    exit 1
fi

export HF_HOME="$HOME/hf_cache"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1


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
echo "ACE-ONLY RUN  [ace: ${ACE_RUN_TAG}  react: ${REACT_RUN_TAG}]"
echo "=========================================="
echo "Systems:    HARNESS+MEMORY+DEBATER (dual_permanent)  |  +AdaReMo"
echo "Models:     qwen25 (Qwen2.5-32B-AWQ)  |  qwen36 (Qwen3.6-35B-A3B-FP8)"
echo "Universes:  ${UNIVERSE_TAGS[*]}"
echo "Period:     ${START_DATE} -> ${END_DATE}"
echo "Capital:    \$${INITIAL_CAPITAL}  |  Fees: 15 bps  |  Rebalance: monthly"
echo "ReAct in:   ${REACT_RESULTS_DIR}/  (existing, not re-run)"
echo "ACE out:    ${ROOT_DIR}/ace_results_${ACE_RUN_TAG}/"
echo "Summary:    ${ROOT_DIR}/summary_${ACE_RUN_TAG}/"
echo "=========================================="
echo ""


# ============================================================
# MAIN LOOP — model × universe (ACE only)
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

    export REACT_MODEL="$MODEL"
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

        ACE_DIR="${ROOT_DIR}/ace_results_${ACE_RUN_TAG}/qwen${MODEL_TAG}_${UNIVERSE_TAG}"
        rm -rf "$ACE_DIR"
        mkdir -p "$ACE_DIR"

        echo ""
        echo "=========================================="
        echo "Universe: ${UNIVERSE_TAG}  (${#TICKER_ARRAY[@]} tickers)  |  qwen${MODEL_TAG}"
        echo "Tickers:  ${TICKERS}"
        echo "=========================================="

        echo ""
        echo "--- ACE: HARNESS+MEMORY+DEBATER (dual_permanent) ---"
        cd "$ROOT_DIR"
        python -m ace_harness.run_monthly \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
            --systems dual_permanent \
            --output_dir "$ACE_DIR" \
            --risk_harness_type conviction \
            --initial_capital "$INITIAL_CAPITAL" \
            --rebalance-days  "$REBALANCE_DAYS"

        echo ""
        echo "--- ACE+AdaReMo: HARNESS+MEMORY+DEBATER+REWARD_MODEL ---"
        python -m ace_harness.run_monthly_adamo \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
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
# PLOTTING — ACE only
# ============================================================

echo ""
echo "=========================================="
echo "Generating ACE plots..."
echo "=========================================="

for MODEL_TAG in 25 36; do
    for UNIVERSE_TAG in "${UNIVERSE_TAGS[@]}"; do
        TICKERS=$(get_tickers "$UNIVERSE_TAG")
        read -ra TICKER_ARRAY <<< "$TICKERS"

        ACE_DIR="${ROOT_DIR}/ace_results_${ACE_RUN_TAG}/qwen${MODEL_TAG}_${UNIVERSE_TAG}"

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

        python -m ace_harness.compare_results "${ACE_DIR}"/*.json \
        || echo "WARNING: compare_results failed for qwen${MODEL_TAG}/${UNIVERSE_TAG}"
        cd "$REACT_DIR"
    done
done


# ============================================================
# FINAL AGGREGATE SUMMARY
# Reads new ACE results + existing ReAct results from REACT_RUN_TAG
# ============================================================

echo ""
echo "=========================================="
echo "Building full aggregate summary..."
echo "=========================================="

SUMMARY_DIR="${ROOT_DIR}/summary_${ACE_RUN_TAG}"
mkdir -p "$SUMMARY_DIR"

cd "$ROOT_DIR"
python -m ace_harness.summarize_full_run \
    --react_dir     "${REACT_RESULTS_DIR}" \
    --ace_base      "${ROOT_DIR}/ace_results_${ACE_RUN_TAG}" \
    --output_dir    "$SUMMARY_DIR" \
    --model_tags    25 36 \
    --universe_tags tech18 mag7 balanced15 sp30 diversified40 volatile25

echo ""
echo "Full summary -> ${SUMMARY_DIR}/"


# ============================================================
# FINAL FILE LISTING
# ============================================================

echo ""
echo "=========================================="
echo "ACE-ONLY RUN COMPLETE  [ace: ${ACE_RUN_TAG}  react: ${REACT_RUN_TAG}]"
echo "=========================================="
echo "Period:  ${START_DATE} -> ${END_DATE}"
echo "Capital: \$${INITIAL_CAPITAL}  |  Fees: 15 bps"
echo ""
echo "ACE result directories (ace_results_${ACE_RUN_TAG}/):"
ls -lhd "${ROOT_DIR}/ace_results_${ACE_RUN_TAG}"/qwen*/ 2>/dev/null | awk '{print "  "$NF}' \
    || echo "  (none)"
echo ""
echo "Summary files (${SUMMARY_DIR}/):"
ls -lh "${SUMMARY_DIR}/" 2>/dev/null | awk '{print "  "$NF, $5}' \
    || echo "  (none)"
echo "=========================================="
