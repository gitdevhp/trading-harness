#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/adaptive_%j.out
#SBATCH --error=log/adaptive_%j.out
#SBATCH --job-name=ACE_Adaptive
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=interactive-gpu

# Runs the adaptive routing system (HARNESS / DEBATER / MEMORY / ACE),
# then calls plot_adaptive for the three-panel visualization + summary.

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

# ── Configurable parameters ────────────────────────────────────────────────────
INITIAL_CAPITAL=1000000
START_DATE="${START_DATE:-2024-01-01}"
END_DATE="${END_DATE:-2024-12-31}"
MAX_TOKENS="${MAX_TOKENS:-800}"
REBALANCE_DAYS="${REBALANCE_DAYS:-20}"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/users/2/cai00317/trading-harness/ace_harness/results/adaptive_${JOB_ID}}"

TICKERS=(
    AAPL MSFT NVDA AVGO AMD ADBE QCOM TXN AMAT MU
    AMZN TSLA PEP COST HON GOOGL META NFLX
)

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "ACE ADAPTIVE ROUTING BACKTEST"
echo "=========================================="
echo "Job:             ${JOB_ID}"
echo "Routing:         breadth<0.40→HARNESS | N≥30,breadth≥0.55→DEBATER"
echo "                 N<30,breadth≥0.55→ACE | else→MEMORY"
echo "Initial capital: \$${INITIAL_CAPITAL}"
echo "Start:           ${START_DATE}"
echo "End:             ${END_DATE}"
echo "Rebalance days:  ${REBALANCE_DAYS}"
echo "Max tokens:      ${MAX_TOKENS}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "=========================================="

# ── Determine what work is needed ─────────────────────────────────────────────

ADAPTIVE_JSON="$OUTPUT_DIR/monthly_adaptive_convictionriskharness_adaptive_results.json"

NEED_LLM=false
if [[ ! -f "$ADAPTIVE_JSON" ]]; then
    NEED_LLM=true
fi
for SYSTEM in intra memory_only dual_permanent; do
    BASELINE_JSON="$OUTPUT_DIR/monthly_${SYSTEM}_convictionriskharness_adaptive_results.json"
    if [[ ! -f "$BASELINE_JSON" ]]; then
        NEED_LLM=true
    fi
done

# ── Start vLLM (only if there is LLM work to do) ──────────────────────────────

VLLM_PID=""

cleanup() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID: ${VLLM_PID})..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "$NEED_LLM" == "true" ]]; then
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
        echo "  Server loading... ${elapsed}s"
        sleep 5
    done

    echo "vLLM server is online."
else
    echo "[resume] All result files already exist — skipping vLLM startup."
fi

# ── Run adaptive backtest ──────────────────────────────────────────────────────

if [[ -f "$ADAPTIVE_JSON" ]]; then
    echo "[skip] Adaptive results already exist: $ADAPTIVE_JSON"
else
    echo ""
    echo "=========================================="
    echo "RUNNING ADAPTIVE SYSTEM"
    echo "=========================================="

    python -m ace_harness.run_monthly_adaptive \
        --tickers "${TICKERS[@]}" \
        --start "$START_DATE" \
        --end "$END_DATE" \
        --output_dir "$OUTPUT_DIR" \
        --initial_capital "$INITIAL_CAPITAL" \
        --rebalance-days "$REBALANCE_DAYS" \
        --max_tokens "$MAX_TOKENS" \
        --fallback_mode equal_weight
fi

if [[ ! -f "$ADAPTIVE_JSON" ]]; then
    echo "ERROR: adaptive results file not found: $ADAPTIVE_JSON"
    exit 1
fi

# ── Run baselines for comparison (optional; skip if results already exist) ─────

echo ""
echo "=========================================="
echo "RUNNING BASELINES FOR COMPARISON"
echo "=========================================="

for SYSTEM in intra memory_only dual_permanent; do
    BASELINE_JSON="$OUTPUT_DIR/monthly_${SYSTEM}_convictionriskharness_adaptive_results.json"
    if [[ -f "$BASELINE_JSON" ]]; then
        echo "  [skip] $SYSTEM — results already exist"
    else
        echo "  Running $SYSTEM..."
        python -m ace_harness.run_monthly \
            --systems "$SYSTEM" \
            --tickers "${TICKERS[@]}" \
            --start "$START_DATE" \
            --end "$END_DATE" \
            --output_dir "$OUTPUT_DIR" \
            --initial_capital "$INITIAL_CAPITAL" \
            --risk_harness_type conviction \
            --fallback_mode equal_weight \
            --max_tokens "$MAX_TOKENS"
    fi
done

# ── Generate visualization + summary ──────────────────────────────────────────

echo ""
echo "=========================================="
echo "GENERATING ADAPTIVE PLOT + SUMMARY"
echo "=========================================="

BASELINE_ARGS=()
for SYSTEM in intra memory_only dual_permanent; do
    BASELINE_JSON="$OUTPUT_DIR/monthly_${SYSTEM}_convictionriskharness_adaptive_results.json"
    if [[ -f "$BASELINE_JSON" ]]; then
        BASELINE_ARGS+=("$BASELINE_JSON")
    fi
done

if [[ ${#BASELINE_ARGS[@]} -gt 0 ]]; then
    python -m ace_harness.plot_adaptive \
        --adaptive_json "$ADAPTIVE_JSON" \
        --baseline_jsons "${BASELINE_ARGS[@]}" \
        --output_png "$OUTPUT_DIR/adaptive_routing.png"
else
    python -m ace_harness.plot_adaptive \
        --adaptive_json "$ADAPTIVE_JSON" \
        --output_png "$OUTPUT_DIR/adaptive_routing.png"
fi

# ── Print metrics summary ──────────────────────────────────────────────────────

echo ""
echo "=========================================="
echo "METRICS SUMMARY"
echo "=========================================="

python - <<'PYEOF'
import json, os, sys

output_dir = os.environ.get("OUTPUT_DIR")
files = []
for name, label in [
    ("monthly_adaptive_convictionriskharness_adaptive_results.json", "ADAPTIVE"),
    ("monthly_intra_convictionriskharness_adaptive_results.json",    "DEBATER"),
    ("monthly_memory_only_convictionriskharness_adaptive_results.json", "MEMORY"),
    ("monthly_dual_permanent_convictionriskharness_adaptive_results.json", "ACE"),
]:
    path = os.path.join(output_dir, name)
    if os.path.exists(path):
        files.append((label, path))

if not files:
    print("No result files found.")
    sys.exit(0)

header = f"{'System':<12} {'Total Ret':>10} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8}"
print(header)
print("-" * len(header))

for label, path in files:
    with open(path) as f:
        data = json.load(f)
    history = data if isinstance(data, list) else data.get("history", [])
    if not history:
        continue
    vals = [r["portfolio_value"] for r in history if "portfolio_value" in r]
    cap = history[0].get("portfolio_value", 1_000_000)
    total_ret = (vals[-1] / cap - 1) * 100 if vals else float("nan")

    rets = []
    for i in range(1, len(vals)):
        rets.append(vals[i] / vals[i-1] - 1)

    import statistics, math
    sharpe = float("nan")
    if len(rets) > 1:
        mean_r = statistics.mean(rets)
        std_r = statistics.stdev(rets)
        if std_r > 0:
            sharpe = (mean_r / std_r) * math.sqrt(12)

    maxdd = float("nan")
    if vals:
        peak = vals[0]
        maxdd = 0.0
        for v in vals:
            peak = max(peak, v)
            dd = (v - peak) / peak
            maxdd = min(maxdd, dd)
        maxdd *= 100

    calmar = float("nan")
    if maxdd < 0 and not math.isnan(sharpe):
        ann_ret = total_ret
        calmar = ann_ret / abs(maxdd)

    print(f"{label:<12} {total_ret:>9.2f}% {sharpe:>8.2f} {maxdd:>7.2f}% {calmar:>8.2f}")

print("")
PYEOF

echo "=========================================="
echo "ADAPTIVE BACKTEST COMPLETE"
echo "Results:   ${OUTPUT_DIR}"
echo "Chart:     ${OUTPUT_DIR}/adaptive_routing.png"
echo "=========================================="
