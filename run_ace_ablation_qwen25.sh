#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/ace_ablation_q25_%j.out
#SBATCH --error=log/ace_ablation_q25_%j.out
#SBATCH --job-name=ACE_Ablation_Q25
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=interactive-gpu

# Runs the three ACE component systems for qwen25 across all 6 universes:
#   intra       = HARNESS+DEBATER only (no memory)
#   memory_only = HARNESS+MEMORY  only (no intra-task debater)
#   dual_permanent = HARNESS+MEMORY+DEBATER (full ACE)
#
# Each universe gets its own output subdirectory under ace_ablation_<JOB_ID>/.
# After all runs, plot_results and compare_results are called per universe.

set -euo pipefail

mkdir -p /users/2/cai00317/log

module load gcc/11.3.0
module load miniforge

ROOT_DIR="/users/2/cai00317/trading-harness"
cd "$ROOT_DIR"
source "${ROOT_DIR}/ReAct/venv/bin/activate"

export HF_HOME="$HOME/hf_cache"
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"
VLLM_PORT=8000

export ACE_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export ACE_SOLVER_MODEL="$MODEL"
export ACE_DEBATER_MODEL="$MODEL"
export ACE_CONSOLIDATOR_MODEL="$MODEL"

INITIAL_CAPITAL=1000000
START_DATE="2024-01-01"
END_DATE="2024-12-31"
REBALANCE_DAYS=1
RUN_TAG="ablation_q25_${SLURM_JOB_ID}"
OUT_ROOT="${ROOT_DIR}/ace_ablation_${SLURM_JOB_ID}"

# ── Universe definitions ──────────────────────────────────────
UNIVERSE_TECH18="AAPL MSFT NVDA AVGO AMD ADBE QCOM TXN AMAT MU AMZN TSLA PEP COST HON GOOGL META NFLX"
UNIVERSE_MAG7="AAPL MSFT NVDA AMZN GOOGL META TSLA"
UNIVERSE_BALANCED15="AAPL MSFT NVDA AMZN GOOGL META JPM JNJ XOM UNH HD WMT BAC PG CVX"
UNIVERSE_SP30="AAPL MSFT NVDA AVGO AMD ADBE QCOM ORCL AMAT MU GOOGL META NFLX AMZN TSLA HD BKNG NKE WMT PG KO JNJ LLY UNH ABBV JPM V MA GS BAC"
UNIVERSE_DIVERSIFIED40="AAPL MSFT NVDA AVGO AMD ADBE QCOM ORCL AMAT MU GOOGL META NFLX CMCSA AMZN TSLA HD BKNG NKE MCD WMT PG KO JNJ LLY UNH ABBV JPM V MA GS BAC XOM CVX COP CAT HON RTX PEP DIS"
UNIVERSE_VOLATILE25="NVDA AMD AVGO QCOM MU AMAT MRVL CRWD PLTR DDOG TSLA META AMZN UBER SPOT ABNB NFLX LLY MRNA BIIB VRTX ABBV AMGN REGN PFE"
UNIVERSE_TAGS=("tech18" "mag7" "balanced15" "sp30" "diversified40" "volatile25")

get_tickers() {
    case "$1" in
        tech18)        echo "$UNIVERSE_TECH18" ;;
        mag7)          echo "$UNIVERSE_MAG7" ;;
        balanced15)    echo "$UNIVERSE_BALANCED15" ;;
        sp30)          echo "$UNIVERSE_SP30" ;;
        diversified40) echo "$UNIVERSE_DIVERSIFIED40" ;;
        volatile25)    echo "$UNIVERSE_VOLATILE25" ;;
        *) echo "ERROR: unknown universe '$1'" >&2; exit 1 ;;
    esac
}

mkdir -p "$OUT_ROOT"

echo "=========================================="
echo "ACE ABLATION — qwen25 only"
echo "=========================================="
echo "Job:        ${SLURM_JOB_ID}"
echo "Model:      ${MODEL}"
echo "Systems:    intra (DEBATER) | memory_only (MEMORY) | dual_permanent (ACE)"
echo "Universes:  ${UNIVERSE_TAGS[*]}"
echo "Period:     ${START_DATE} -> ${END_DATE}"
echo "Capital:    \$${INITIAL_CAPITAL}"
echo "Output:     ${OUT_ROOT}/"
echo "=========================================="

# ── Start vLLM ───────────────────────────────────────────────
echo "Starting vLLM server..."
vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser hermes &

VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

TIMEOUT=1800
START_T=$(date +%s)
until curl -fsS http://127.0.0.1:${VLLM_PORT}/v1/models >/dev/null 2>&1; do
    ! kill -0 "$VLLM_PID" 2>/dev/null && echo "ERROR: vLLM died" && exit 1
    (( $(date +%s) - START_T >= TIMEOUT )) && echo "ERROR: vLLM timeout" && exit 1
    echo "Server loading... $(( $(date +%s) - START_T ))s"
    sleep 5
done
echo "vLLM online."

# ── Main loop ─────────────────────────────────────────────────
for UNIVERSE_TAG in "${UNIVERSE_TAGS[@]}"; do
    TICKERS=$(get_tickers "$UNIVERSE_TAG")
    read -ra TICKER_ARRAY <<< "$TICKERS"
    UNI_DIR="${OUT_ROOT}/${UNIVERSE_TAG}"
    mkdir -p "$UNI_DIR"

    echo ""
    echo "=========================================="
    echo "Universe: ${UNIVERSE_TAG}  (${#TICKER_ARRAY[@]} stocks)"
    echo "=========================================="

    # ── HARNESS+DEBATER only (intra) ─────────────────────────
    echo ""
    echo "--- System: HARNESS+DEBATER (intra) ---"
    DEBATER_DIR="${UNI_DIR}/debater"
    if compgen -G "${DEBATER_DIR}/*_results.json" >/dev/null 2>&1; then
        echo "SKIP: ${DEBATER_DIR}/ already complete."
    else
        rm -rf "$DEBATER_DIR" && mkdir -p "$DEBATER_DIR"
        python -m ace_harness.run_monthly \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
            --systems intra \
            --output_dir "$DEBATER_DIR" \
            --risk_harness_type conviction \
            --initial_capital "$INITIAL_CAPITAL" \
            --rebalance-days  "$REBALANCE_DAYS"
        echo "Done -> ${DEBATER_DIR}/"
    fi

    # ── HARNESS+MEMORY only (memory_only / inter) ─────────────
    echo ""
    echo "--- System: HARNESS+MEMORY (memory_only) ---"
    MEMORY_DIR="${UNI_DIR}/memory"
    if compgen -G "${MEMORY_DIR}/*_results.json" >/dev/null 2>&1; then
        echo "SKIP: ${MEMORY_DIR}/ already complete."
    else
        rm -rf "$MEMORY_DIR" && mkdir -p "$MEMORY_DIR"
        python -m ace_harness.run_monthly \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
            --systems memory_only \
            --output_dir "$MEMORY_DIR" \
            --risk_harness_type conviction \
            --initial_capital "$INITIAL_CAPITAL" \
            --rebalance-days  "$REBALANCE_DAYS"
        echo "Done -> ${MEMORY_DIR}/"
    fi

    # ── Full ACE: HARNESS+MEMORY+DEBATER (dual_permanent) ─────
    echo ""
    echo "--- System: ACE / HARNESS+MEMORY+DEBATER (dual_permanent) ---"
    ACE_DIR="${UNI_DIR}/ace"
    if compgen -G "${ACE_DIR}/*_results.json" >/dev/null 2>&1; then
        echo "SKIP: ${ACE_DIR}/ already complete."
    else
        rm -rf "$ACE_DIR" && mkdir -p "$ACE_DIR"
        python -m ace_harness.run_monthly \
            --tickers "${TICKER_ARRAY[@]}" \
            --start   "$START_DATE" \
            --end     "$END_DATE" \
            --systems dual_permanent \
            --output_dir "$ACE_DIR" \
            --risk_harness_type conviction \
            --initial_capital "$INITIAL_CAPITAL" \
            --rebalance-days  "$REBALANCE_DAYS"
        echo "Done -> ${ACE_DIR}/"
    fi

    # ── Metrics + plot per universe ────────────────────────────
    echo ""
    echo "--- Metrics: ${UNIVERSE_TAG} ---"
    ALL_RESULTS=()
    for SUBDIR in debater memory ace; do
        while IFS= read -r -d '' f; do ALL_RESULTS+=("$f"); done \
            < <(find "${UNI_DIR}/${SUBDIR}" -name "*_results.json" -print0 2>/dev/null)
    done

    if [ ${#ALL_RESULTS[@]} -gt 0 ]; then
        python -m ace_harness.compare_results "${ALL_RESULTS[@]}" || true
        python -m ace_harness.plot_results \
            "${ALL_RESULTS[@]}" \
            --output_dir "$UNI_DIR" \
            --tickers "${TICKER_ARRAY[@]}" \
            --start "$START_DATE" \
            --end   "$END_DATE" \
            --initial_capital "$INITIAL_CAPITAL" || true
        echo "Plot -> ${UNI_DIR}/ace_comparison.png"
    fi

    echo "Universe ${UNIVERSE_TAG} complete."
done

# ── Regime analysis ───────────────────────────────────────────
echo ""
echo "=========================================="
echo "Running regime concentration analysis..."
echo "=========================================="
python -m ace_harness.analyze_regime --output_dir "${OUT_ROOT}/regime_analysis" || true

echo ""
echo "=========================================="
echo "ACE ABLATION COMPLETE"
echo "Results: ${OUT_ROOT}/"
echo "=========================================="
