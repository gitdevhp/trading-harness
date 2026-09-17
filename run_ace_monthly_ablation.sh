#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/ace_monthly_%j.out
#SBATCH --error=log/ace_monthly_%j.out
#SBATCH --job-name=ACE_Monthly
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=interactive-gpu

# Runs all ACE ablation systems in one job:
#   baseline, intra, memory_only, dual_permanent
# then calls compare_results and plot_results for a full metrics report.

set -euo pipefail

mkdir -p /users/2/cai00317/log

module load gcc/11.3.0
module load miniforge

cd /users/2/cai00317/trading-harness
source /users/2/cai00317/trading-harness/ReAct/venv/bin/activate

export HF_HOME="$HOME/hf_cache"
export VLLM_USE_FLASHINFER_SAMPLER=0
export ACE_LLM_BASE_URL="http://127.0.0.1:8000/v1"
export ACE_SOLVER_MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"
export ACE_DEBATER_MODEL="$ACE_SOLVER_MODEL"
export ACE_CONSOLIDATOR_MODEL="$ACE_SOLVER_MODEL"

INITIAL_CAPITAL=1000000
START_DATE="2024-01-01"
END_DATE="2024-12-31"
OUTPUT_DIR="/users/2/cai00317/trading-harness/ace_harness/results/monthly_${SLURM_JOB_ID}"

TICKERS=(
    AAPL MSFT NVDA AVGO AMD ADBE QCOM TXN AMAT MU
    AMZN TSLA PEP COST HON GOOGL META NFLX
)

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "ACE MONTHLY FULL ABLATION"
echo "=========================================="
echo "Job:                ${SLURM_JOB_ID}"
echo "Systems:            baseline, intra, memory_only, dual_permanent"
echo "Initial capital:    \$${INITIAL_CAPITAL}"
echo "Start:              ${START_DATE}"
echo "End:                ${END_DATE}"
echo "Rebalance:          first trading day of each month"
echo "Risk harness:       conviction (adaptive for memory systems)"
echo "Output directory:   ${OUTPUT_DIR}"
echo "=========================================="

# ──────────────────────────────────────────────
# Start vLLM
# ──────────────────────────────────────────────

echo "Starting vLLM server..."
vllm serve "$ACE_SOLVER_MODEL" \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser hermes &

VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Waiting for vLLM server (PID: ${VLLM_PID})..."
START_TIME=$(date +%s)
VLLM_TIMEOUT=1800

while ! curl -fsS http://127.0.0.1:8000/v1/models >/dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server died during startup."
        exit 1
    fi
    elapsed=$(( $(date +%s) - START_TIME ))
    if (( elapsed >= VLLM_TIMEOUT )); then
        echo "ERROR: vLLM startup timed out after ${elapsed}s."
        exit 1
    fi
    echo "Server loading... ${elapsed}s"
    sleep 5
done

echo "vLLM server is online."

# ──────────────────────────────────────────────
# Run all systems
# ──────────────────────────────────────────────

python -m ace_harness.run_monthly \
    --systems baseline intra memory_only dual_permanent \
    --tickers "${TICKERS[@]}" \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --output_dir "$OUTPUT_DIR" \
    --initial_capital "$INITIAL_CAPITAL" \
    --risk_harness_type conviction \
    --fallback_mode equal_weight \
    --max_tokens 600

# ──────────────────────────────────────────────
# Compare and plot results
# ──────────────────────────────────────────────

echo ""
echo "=========================================="
echo "METRICS COMPARISON"
echo "=========================================="

python -m ace_harness.compare_results "$OUTPUT_DIR"/*_results.json

echo ""
echo "=========================================="
echo "GENERATING PLOTS"
echo "=========================================="

python -m ace_harness.plot_results \
    "$OUTPUT_DIR"/*_results.json \
    --output_dir "$OUTPUT_DIR" \
    --tickers "${TICKERS[@]}" \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --initial_capital "$INITIAL_CAPITAL"

echo ""
echo "=========================================="
echo "ACE MONTHLY ABLATION COMPLETE"
echo "Results:   ${OUTPUT_DIR}"
echo "Chart:     ${OUTPUT_DIR}/ace_comparison.png"
echo "Metrics:   ${OUTPUT_DIR}/comparison_metrics.json"
echo "=========================================="
