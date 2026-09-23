#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/adaptive_only_%j.out
#SBATCH --error=log/adaptive_only_%j.out
#SBATCH --job-name=ACE_ADAPTIVE
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=interactive-gpu

# Runs ONLY the adaptive (AutoGovern) system for all 6 universes, then compares
# its results against existing memory / debater / dual results from a prior run.
#
# Required env var:
#   PREV_RUN_DIR   path to the prior ace_mda_* directory that contains
#                  {universe}/memory/, {universe}/debater/, {universe}/dual/
#
# Optional env vars (all have defaults):
#   OUT_ROOT       where to write new adaptive results
#                  default: ${PREV_RUN_DIR}_adaptive_fix_${JOB_ID}
#   START_DATE     default: 2024-01-01
#   END_DATE       default: 2024-12-31
#   INITIAL_CAPITAL  default: 1000000
#   MAX_TOKENS     default: 800
#
# Output structure (under OUT_ROOT):
#   {universe}/adaptive/   ← new AutoGovern results
#
# The compare step reads memory/debater/dual from PREV_RUN_DIR and adaptive
# from OUT_ROOT, so you get a clean 4-way comparison in one table.

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

# ── Parameters ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL=1000000
START_DATE="${START_DATE:-2024-01-01}"
END_DATE="${END_DATE:-2024-12-31}"
MAX_TOKENS="${MAX_TOKENS:-800}"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${PREV_RUN_DIR:-}" ]]; then
    echo "ERROR: PREV_RUN_DIR must be set to the prior ace_mda_* directory" >&2
    echo "  e.g.  PREV_RUN_DIR=/users/2/cai00317/trading-harness/ace_mda_1706716 sbatch run_adaptive_only.sh" >&2
    exit 1
fi

OUT_ROOT="${OUT_ROOT:-${PREV_RUN_DIR}_adaptive_fix_${JOB_ID}}"

# ── Universe definitions ───────────────────────────────────────────────────────
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
echo "ACE — ADAPTIVE ONLY (r1-fallback fix)"
echo "=========================================="
echo "Job:          ${JOB_ID}"
echo "Model:        ${MODEL}"
echo "System:       adaptive / AutoGovern (Algorithm 2 + r1 fallback)"
echo "Universes:    ${UNIVERSE_TAGS[*]}"
echo "Period:       ${START_DATE} -> ${END_DATE}"
echo "Capital:      \$${INITIAL_CAPITAL}"
echo "Max tokens:   ${MAX_TOKENS}"
echo "Baselines:    ${PREV_RUN_DIR}/"
echo "Output:       ${OUT_ROOT}/"
echo "=========================================="

# ── Check which universes still need running ───────────────────────────────────
NEED_LLM=false
for TAG in "${UNIVERSE_TAGS[@]}"; do
    ADAPTIVE_JSON="${OUT_ROOT}/${TAG}/adaptive/monthly_adaptive_autogover_results.json"
    if [[ ! -f "$ADAPTIVE_JSON" ]]; then
        NEED_LLM=true
        break
    fi
done

# ── Start vLLM (only if needed) ───────────────────────────────────────────────
VLLM_PID=""

cleanup() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM (PID: ${VLLM_PID})..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "$NEED_LLM" == "true" ]]; then
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
    until curl -fsS http://127.0.0.1:${VLLM_PORT}/v1/models >/dev/null 2>&1; do
        ! kill -0 "$VLLM_PID" 2>/dev/null && echo "ERROR: vLLM died" && exit 1
        elapsed=$(( $(date +%s) - START_T ))
        (( elapsed >= TIMEOUT )) && echo "ERROR: vLLM startup timed out after ${elapsed}s" && exit 1
        echo "  Server loading... ${elapsed}s"
        sleep 5
    done
    echo "vLLM online."
else
    echo "[resume] All adaptive results already exist — skipping vLLM startup."
fi

# ── Main loop: one universe at a time ─────────────────────────────────────────
for UNIVERSE_TAG in "${UNIVERSE_TAGS[@]}"; do
    TICKERS=$(get_tickers "$UNIVERSE_TAG")
    read -ra TICKER_ARRAY <<< "$TICKERS"

    echo ""
    echo "=========================================="
    echo "Universe: ${UNIVERSE_TAG}  (${#TICKER_ARRAY[@]} stocks)"
    echo "=========================================="

    # ── ADAPTIVE ──────────────────────────────────────────────────────────────
    ADAPTIVE_DIR="${OUT_ROOT}/${UNIVERSE_TAG}/adaptive"
    ADAPTIVE_JSON="${ADAPTIVE_DIR}/monthly_adaptive_autogover_results.json"

    if [[ -f "$ADAPTIVE_JSON" ]]; then
        echo "  [skip] ADAPTIVE — results already exist"
    else
        echo ""
        echo "--- Running ADAPTIVE (r1 fallback fix) ---"
        mkdir -p "$ADAPTIVE_DIR"
        python -m ace_harness.run_monthly_adaptive \
            --tickers "${TICKER_ARRAY[@]}" \
            --start "$START_DATE" \
            --end   "$END_DATE" \
            --output_dir "$ADAPTIVE_DIR" \
            --initial_capital "$INITIAL_CAPITAL" \
            --rebalance-days 20 \
            --fallback_mode equal_weight \
            --max_tokens "$MAX_TOKENS"
        echo "Done -> ${ADAPTIVE_DIR}/"
    fi

    # ── Per-universe compare: adaptive (new) vs baselines (prev run) ──────────
    echo ""
    echo "--- Metrics + compare: ${UNIVERSE_TAG} ---"

    ALL_RESULTS=()
    # Baselines from the prior run
    for SUBDIR in memory debater dual; do
        while IFS= read -r -d '' f; do
            ALL_RESULTS+=("$f")
        done < <(find "${PREV_RUN_DIR}/${UNIVERSE_TAG}/${SUBDIR}" -name "*_results.json" -print0 2>/dev/null)
    done
    # New adaptive result
    [[ -f "$ADAPTIVE_JSON" ]] && ALL_RESULTS+=("$ADAPTIVE_JSON")

    if [[ ${#ALL_RESULTS[@]} -gt 0 ]]; then
        python -m ace_harness.compare_results "${ALL_RESULTS[@]}" || true
    fi

    # plot_adaptive with prior baselines
    if [[ -f "$ADAPTIVE_JSON" ]]; then
        BASELINE_ARGS=()
        MEM_JSON=$(find "${PREV_RUN_DIR}/${UNIVERSE_TAG}/memory"   -name "*_results.json" 2>/dev/null | head -1)
        DEB_JSON=$(find "${PREV_RUN_DIR}/${UNIVERSE_TAG}/debater"  -name "*_results.json" 2>/dev/null | head -1)
        DUAL_JSON=$(find "${PREV_RUN_DIR}/${UNIVERSE_TAG}/dual"    -name "*_results.json" 2>/dev/null | head -1)
        [[ -n "$MEM_JSON" ]]  && BASELINE_ARGS+=("$MEM_JSON")
        [[ -n "$DEB_JSON" ]]  && BASELINE_ARGS+=("$DEB_JSON")
        [[ -n "$DUAL_JSON" ]] && BASELINE_ARGS+=("$DUAL_JSON")

        if [[ ${#BASELINE_ARGS[@]} -gt 0 ]]; then
            python -m ace_harness.plot_adaptive \
                --adaptive_json "$ADAPTIVE_JSON" \
                --baseline_jsons "${BASELINE_ARGS[@]}" \
                --output_png "${ADAPTIVE_DIR}/adaptive_routing.png" || true
        else
            python -m ace_harness.plot_adaptive \
                --adaptive_json "$ADAPTIVE_JSON" \
                --output_png "${ADAPTIVE_DIR}/adaptive_routing.png" || true
        fi
        echo "Plot -> ${ADAPTIVE_DIR}/adaptive_routing.png"
    fi

    echo "Universe ${UNIVERSE_TAG} complete."
done

# ── Cross-universe summary: new adaptive vs prior baselines ───────────────────
echo ""
echo "=========================================="
echo "CROSS-UNIVERSE METRICS SUMMARY"
echo "=========================================="

python - <<PYEOF
import glob, json, math, os, statistics

prev_run_dir = os.environ.get("PREV_RUN_DIR", "")
out_root     = "${OUT_ROOT}"
universes    = ["tech18", "mag7", "balanced15", "sp30", "diversified40", "volatile25"]

# (base_dir, subdir, label)
systems = [
    (prev_run_dir, "memory",   "MEMORY"),
    (prev_run_dir, "debater",  "DEBATER"),
    (prev_run_dir, "dual",     "DUAL"),
    (out_root,     "adaptive", "ADAPTIVE"),
]

def load_result(base, subdir, uni):
    matches = glob.glob(os.path.join(base, uni, subdir, "*_results.json"))
    if not matches:
        return None, None
    with open(matches[0]) as f:
        return matches[0], json.load(f)

def perf_metrics(data):
    history = data if isinstance(data, list) else data.get("history", [])
    vals = [r["portfolio_value"] for r in history if "portfolio_value" in r]
    if not vals:
        return None
    cap = vals[0]
    total_ret = (vals[-1] / cap - 1) * 100
    rets = [vals[i] / vals[i-1] - 1 for i in range(1, len(vals))]
    sharpe = float("nan")
    if len(rets) > 1:
        m, s = statistics.mean(rets), statistics.stdev(rets)
        if s > 0:
            sharpe = (m / s) * math.sqrt(12)
    peak, maxdd = vals[0], 0.0
    for v in vals:
        peak = max(peak, v)
        maxdd = min(maxdd, (v - peak) / peak)
    maxdd *= 100
    calmar = total_ret / abs(maxdd) if maxdd < 0 else float("nan")
    return total_ret, sharpe, maxdd, calmar

hdr = f"{'Universe':<15} {'System':<10} {'TotalRet':>9} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7}"
print(hdr)
print("-" * len(hdr))
for uni in universes:
    for base, subdir, label in systems:
        _, data = load_result(base, subdir, uni)
        if data is None:
            continue
        m = perf_metrics(data)
        if m:
            tr, sh, dd, ca = m
            print(f"{uni:<15} {label:<10} {tr:>8.2f}% {sh:>7.2f} {dd:>7.2f}% {ca:>7.2f}")
    print()

print()
print("TOKEN USAGE PER SYSTEM x UNIVERSE")
thdr = f"{'Universe':<15} {'System':<10} {'Calls':>7} {'Prompt':>12} {'Completion':>12} {'Total':>12}"
print(thdr)
print("-" * len(thdr))

grand = {label: {"calls": 0, "prompt": 0, "completion": 0, "total": 0} for _, _, label in systems}
for uni in universes:
    for base, subdir, label in systems:
        _, data = load_result(base, subdir, uni)
        if data is None:
            continue
        tok = data.get("token_usage") if isinstance(data, dict) else None
        if not tok:
            continue
        calls = tok.get("calls", 0)
        prompt = tok.get("prompt", 0)
        comp = tok.get("completion", 0)
        total = tok.get("total", 0)
        print(f"{uni:<15} {label:<10} {calls:>7,} {prompt:>12,} {comp:>12,} {total:>12,}")
        grand[label]["calls"]      += calls
        grand[label]["prompt"]     += prompt
        grand[label]["completion"] += comp
        grand[label]["total"]      += total
    print()

print("TOTALS ACROSS ALL UNIVERSES")
print("-" * len(thdr))
for _, _, label in systems:
    g = grand[label]
    if g["calls"] == 0:
        continue
    print(f"{'ALL':<15} {label:<10} {g['calls']:>7,} {g['prompt']:>12,} {g['completion']:>12,} {g['total']:>12,}")
PYEOF

echo ""
echo "=========================================="
echo "ADAPTIVE-ONLY RUN COMPLETE"
echo "New adaptive results: ${OUT_ROOT}/"
echo "Baselines from:       ${PREV_RUN_DIR}/"
echo "=========================================="
