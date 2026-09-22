#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --mem=32gb
#SBATCH --output=log/debug_autogov_%j.out
#SBATCH --error=log/debug_autogov_%j.out
#SBATCH --job-name=ACE_DBG
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=interactive-gpu

# Quick AutoGovern debugger — 3 rebalances on mag7.
# Prints per-round debater verdict/direction/g_ref/g_sto/feedback so you
# can see exactly why admitted=0.
#
# Usage:
#   sbatch run_debug_autogov.sh
#   # or locally (vLLM already running):
#   bash run_debug_autogov.sh
#
# Optional overrides:
#   TICKERS="AAPL MSFT NVDA AMZN GOOGL META TSLA"  (default: mag7)
#   START_DATE="2024-01-01"
#   END_DATE="2024-04-01"    (short window = fast, ~3 rebalances)
#   MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"

set -euo pipefail

mkdir -p /users/2/cai00317/log

module load gcc/11.3.0
module load miniforge

ROOT_DIR="/users/2/cai00317/trading-harness"
cd "$ROOT_DIR"
source "${ROOT_DIR}/ReAct/venv/bin/activate"

export HF_HOME="$HOME/hf_cache"
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
VLLM_PORT=8000

export ACE_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export ACE_SOLVER_MODEL="$MODEL"
export ACE_DEBATER_MODEL="$MODEL"
export ACE_CONSOLIDATOR_MODEL="$MODEL"

START_DATE="${START_DATE:-2024-01-01}"
END_DATE="${END_DATE:-2024-04-01}"

echo "=========================================="
echo "AutoGovern Debugger"
echo "=========================================="
echo "Model:   ${MODEL}"
echo "Period:  ${START_DATE} -> ${END_DATE}  (~3 rebalances)"
echo "Watch:   [autogov] lines for per-round verdict details"
echo "=========================================="

# ── Start vLLM ────────────────────────────────────────────────────────────────
VLLM_PID=""

cleanup() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM (PID: ${VLLM_PID})..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Only start vLLM if it isn't already running
if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM already running — skipping startup."
else
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

    TIMEOUT=1800
    START_T=$(date +%s)
    until curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
        ! kill -0 "$VLLM_PID" 2>/dev/null && echo "ERROR: vLLM died" && exit 1
        elapsed=$(( $(date +%s) - START_T ))
        (( elapsed >= TIMEOUT )) && echo "ERROR: vLLM startup timed out after ${elapsed}s" && exit 1
        echo "  Server loading... ${elapsed}s"
        sleep 5
    done
    echo "vLLM online."
fi

# ── Run debugger ──────────────────────────────────────────────────────────────
echo ""
echo "--- Running debug_autogov.py ---"
START_DATE="$START_DATE" END_DATE="$END_DATE" python debug_autogov.py

echo ""
echo "=========================================="
echo "DONE — check [autogov] lines above"
echo ""
echo "What to look for:"
echo "  verdict=revise, g_ref=False  → debater gave up on round 1 (most common cause of 0% admitted)"
echo "  verdict=revise, g_ref=True   → solver got feedback but debater never conceded in 3 rounds"
echo "  verdict=accept               → working correctly"
echo "=========================================="
