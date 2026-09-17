#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/reactharness_%j.out
#SBATCH --error=log/reactharness_%j.out
#SBATCH --job-name=ReAct_Qwen
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=interactive-gpu

set -euo pipefail

# ============================================================
# MODEL SELECTION — choose ONE:
#
#   MODEL_VERSION="qwen25"   (Qwen2.5-32B-Instruct-AWQ,  1 x A40)
#   MODEL_VERSION="qwen36"   (Qwen3.6-35B-A3B-FP8,       2 x A40)
# ============================================================

MODEL_VERSION="qwen36"


# ============================================================
# BASIC SETUP
# ============================================================

mkdir -p log

module load gcc/11.3.0
module load miniforge

cd /users/2/cai00317/trading-harness/ReAct
source venv/bin/activate

export HF_HOME="$HOME/hf_cache"
export VLLM_USE_FLASHINFER_SAMPLER=0


# ============================================================
# MODEL-SPECIFIC VLLM CONFIGURATION
# ============================================================

if [ "$MODEL_VERSION" = "qwen25" ]; then

    MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"
    GPU_COUNT=1

    VLLM_ARGS=(
        --model "$MODEL"
        --host 127.0.0.1
        --port 8000
        --tensor-parallel-size 1
        --max-model-len 32768
        --gpu-memory-utilization 0.90
        --enable-chunked-prefill
        --enable-auto-tool-choice
        --tool-call-parser hermes
    )

    MODEL_TAG="25"

elif [ "$MODEL_VERSION" = "qwen36" ]; then

    MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
    GPU_COUNT=2

    VLLM_ARGS=(
        --model "$MODEL"
        --host 127.0.0.1
        --port 8000
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

    MODEL_TAG="36"

else
    echo "ERROR: MODEL_VERSION must be 'qwen25' or 'qwen36'"
    exit 1
fi

# Export so Python scripts pick up the correct model name.
export REACT_MODEL="$MODEL"


# ============================================================
# BACKTEST CONFIGURATION
# ============================================================

INITIAL_CAPITAL=1000000
REBALANCE_DAYS=20
WARMUP_DAYS=0

START_DATE="2024-01-01"
END_DATE="2024-12-31"


# ============================================================
# OUTPUT FILES
# ============================================================

RAW_OUTPUT="qwen_raw_results_compare_${MODEL_TAG}.json"
NO_HARNESS_OUTPUT="react_no_harness_results_compare_${MODEL_TAG}.json"
GEMINI_OUTPUT="react_harness_results_compare_${MODEL_TAG}.json"
GPT_OUTPUT="react_gpt_harness_results_compare_${MODEL_TAG}.json"
PLOT_OUTPUT="results_compare_${MODEL_TAG}.png"


# ============================================================
# PRINT CONFIGURATION
# ============================================================

echo "=========================================="
echo "ReAct MODEL COMPARISON"
echo "=========================================="
echo "Model version:      ${MODEL_VERSION}"
echo "Model:              ${MODEL}"
echo "GPU count:          ${GPU_COUNT} x A40"
echo "Initial capital:    \$${INITIAL_CAPITAL}"
echo "Universe:           18 stocks"
echo "Start:              ${START_DATE}"
echo "End:                ${END_DATE}"
echo "Rebalancing:        Monthly"
echo "Rebalance days:     ${REBALANCE_DAYS}"
echo "Warmup:             ${WARMUP_DAYS}"
echo "Transaction cost:   15 bps"
echo "REACT_MODEL:        ${REACT_MODEL}"
echo "=========================================="
echo ""


# ============================================================
# START VLLM
# ============================================================

echo "Starting vLLM..."
echo "Model: ${MODEL}"

python -m vllm.entrypoints.openai.api_server \
    "${VLLM_ARGS[@]}" &

VLLM_PID=$!


# ============================================================
# CLEANUP
# ============================================================

cleanup() {
    echo ""
    echo "Stopping vLLM server..."
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM


# ============================================================
# WAIT FOR VLLM
# ============================================================

echo "Waiting for vLLM server..."
echo "PID: ${VLLM_PID}"

START_TIME=$(date +%s)
VLLM_TIMEOUT=3600

while ! curl -fsS http://127.0.0.1:8000/v1/models > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo ""
        echo "ERROR: vLLM died during startup."
        exit 1
    fi
    ELAPSED=$(( $(date +%s) - START_TIME ))
    if [ "$ELAPSED" -ge "$VLLM_TIMEOUT" ]; then
        echo "ERROR: vLLM startup timed out after ${ELAPSED}s."
        exit 1
    fi
    echo "Waiting... ${ELAPSED}s"
    sleep 5
done

echo ""
echo "vLLM is ready."
echo ""


# ============================================================
# VERIFY MODEL
# ============================================================

echo "=========================================="
echo "Verifying served model"
echo "=========================================="

curl -fsS http://127.0.0.1:8000/v1/models
echo ""
echo ""

nvidia-smi
echo ""


# ============================================================
# STEP 1 — RAW QWEN
# ============================================================

echo "=========================================="
echo "Step 1/5: Running Raw Qwen Baseline"
echo "=========================================="

python qwen_port.py

echo ""
echo "Raw Qwen complete."
echo ""


# ============================================================
# STEP 2 — REACT NO HARNESS
# ============================================================

echo "=========================================="
echo "Step 2/5: Running ReAct (No Harness)"
echo "=========================================="

python noharness.py

echo ""
echo "ReAct No Harness complete."
echo ""


# ============================================================
# STEP 3 — REACT + GEMINI HARNESS
# ============================================================

echo "=========================================="
echo "Step 3/5: Running ReAct + Gemini Harness"
echo "=========================================="

python yesharness.py

echo ""
echo "ReAct + Gemini Harness complete."
echo ""


# ============================================================
# STEP 4 — REACT + GPT HARNESS
# ============================================================

echo "=========================================="
echo "Step 4/5: Running ReAct + GPT Harness"
echo "=========================================="

python yesharnessgpt.py

echo ""
echo "ReAct + GPT Harness complete."
echo ""


# ============================================================
# STEP 5 — PLOT / RESULTS
# ============================================================

echo "=========================================="
echo "Step 5/5: Generating Results"
echo "=========================================="

python plotport.py

echo ""
echo ""


# ============================================================
# COMPLETE
# ============================================================

echo "=========================================="
echo "ALL BACKTESTS COMPLETED"
echo "=========================================="

echo "Model:              ${MODEL}"
echo "Initial capital:    \$${INITIAL_CAPITAL}"
echo "Universe:           18 stocks"
echo "Start:              ${START_DATE}"
echo "End:                ${END_DATE}"
echo "Rebalancing:        Monthly"
echo ""

echo "Output files:"
echo "  ${RAW_OUTPUT}"
echo "  ${NO_HARNESS_OUTPUT}"
echo "  ${GEMINI_OUTPUT}"
echo "  ${GPT_OUTPUT}"
echo "  ${PLOT_OUTPUT}"

echo "=========================================="
