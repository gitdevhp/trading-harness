#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/mda_%j.out
#SBATCH --error=log/mda_%j.out
#SBATCH --job-name=ACE_MDA
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=interactive-gpu

# Runs memory_only + intra (debater) + dual_permanent + adaptive for all 6
# universes.
#
# Output structure:
#   {OUT_ROOT}/{universe}/memory/    ← memory_only results
#   {OUT_ROOT}/{universe}/debater/   ← intra (debater-only) results
#   {OUT_ROOT}/{universe}/dual/      ← dual_permanent results
#   {OUT_ROOT}/{universe}/adaptive/  ← adaptive (AutoGovern) results + playbook
#
# Token usage is saved inside each *_results.json under "token_usage" and is
# summarised in the final cross-universe table.
#
# Resume: each step checks for existing *_results.json and skips if found.
# vLLM is only started if at least one step needs to run.

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

# ── Configurable parameters ────────────────────────────────────────────────────
INITIAL_CAPITAL=1000000
START_DATE="${START_DATE:-2024-01-01}"
END_DATE="${END_DATE:-2024-12-31}"
MAX_TOKENS="${MAX_TOKENS:-800}"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/ace_mda_${JOB_ID}}"

# Optional: path to a prior ace_ablation_* directory containing per-universe
# /ace/ subdirectories with dual_permanent results, for inclusion in plots.
ACE_RESULTS_ROOT="${ACE_RESULTS_ROOT:-}"

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
echo "ACE — memory_only + intra + adaptive"
echo "=========================================="
echo "Job:          ${JOB_ID}"
echo "Model:        ${MODEL}"
echo "Systems:      memory_only (MEMORY) | intra (DEBATER) | dual_permanent (DUAL) | adaptive (ADAPTIVE)"
echo "Universes:    ${UNIVERSE_TAGS[*]}"
echo "Period:       ${START_DATE} -> ${END_DATE}"
echo "Capital:      \$${INITIAL_CAPITAL}"
echo "Max tokens:   ${MAX_TOKENS}"
echo "Output:       ${OUT_ROOT}/"
if [[ -n "$ACE_RESULTS_ROOT" ]]; then
    echo "ACE baselines: ${ACE_RESULTS_ROOT}/"
fi
echo "=========================================="

# ── Determine if any LLM work is needed ───────────────────────────────────────
NEED_LLM=false
for TAG in "${UNIVERSE_TAGS[@]}"; do
    for SUBDIR in memory debater dual adaptive; do
        DIR="${OUT_ROOT}/${TAG}/${SUBDIR}"
        if ! compgen -G "${DIR}/*_results.json" >/dev/null 2>&1; then
            NEED_LLM=true
            break 2
        fi
    done
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
    echo "[resume] All results already exist — skipping vLLM startup."
fi

# ── Main loop: one universe at a time ─────────────────────────────────────────
for UNIVERSE_TAG in "${UNIVERSE_TAGS[@]}"; do
    TICKERS=$(get_tickers "$UNIVERSE_TAG")
    read -ra TICKER_ARRAY <<< "$TICKERS"
    UNI_DIR="${OUT_ROOT}/${UNIVERSE_TAG}"
    mkdir -p "$UNI_DIR"

    echo ""
    echo "=========================================="
    echo "Universe: ${UNIVERSE_TAG}  (${#TICKER_ARRAY[@]} stocks)"
    echo "=========================================="

    # ── MEMORY only ───────────────────────────────────────────────────────────
    MEMORY_DIR="${UNI_DIR}/memory"
    if compgen -G "${MEMORY_DIR}/*_results.json" >/dev/null 2>&1; then
        echo "  [skip] MEMORY — results already exist"
    else
        echo ""
        echo "--- Running MEMORY (memory_only) ---"
        mkdir -p "$MEMORY_DIR"
        python -m ace_harness.run_monthly \
            --tickers "${TICKER_ARRAY[@]}" \
            --start "$START_DATE" \
            --end   "$END_DATE" \
            --systems memory_only \
            --output_dir "$MEMORY_DIR" \
            --risk_harness_type conviction \
            --initial_capital "$INITIAL_CAPITAL" \
            --fallback_mode equal_weight \
            --max_tokens "$MAX_TOKENS"
        echo "Done -> ${MEMORY_DIR}/"
    fi

    # ── DEBATER only ──────────────────────────────────────────────────────────
    DEBATER_DIR="${UNI_DIR}/debater"
    if compgen -G "${DEBATER_DIR}/*_results.json" >/dev/null 2>&1; then
        echo "  [skip] DEBATER — results already exist"
    else
        echo ""
        echo "--- Running DEBATER (intra) ---"
        mkdir -p "$DEBATER_DIR"
        python -m ace_harness.run_monthly \
            --tickers "${TICKER_ARRAY[@]}" \
            --start "$START_DATE" \
            --end   "$END_DATE" \
            --systems intra \
            --output_dir "$DEBATER_DIR" \
            --risk_harness_type conviction \
            --initial_capital "$INITIAL_CAPITAL" \
            --fallback_mode equal_weight \
            --max_tokens "$MAX_TOKENS"
        echo "Done -> ${DEBATER_DIR}/"
    fi

    # ── DUAL (memory + debater) ───────────────────────────────────────────────
    DUAL_DIR="${UNI_DIR}/dual"
    if compgen -G "${DUAL_DIR}/*_results.json" >/dev/null 2>&1; then
        echo "  [skip] DUAL — results already exist"
    else
        echo ""
        echo "--- Running DUAL (dual_permanent) ---"
        mkdir -p "$DUAL_DIR"
        python -m ace_harness.run_monthly \
            --tickers "${TICKER_ARRAY[@]}" \
            --start "$START_DATE" \
            --end   "$END_DATE" \
            --systems dual_permanent \
            --output_dir "$DUAL_DIR" \
            --risk_harness_type conviction \
            --initial_capital "$INITIAL_CAPITAL" \
            --fallback_mode equal_weight \
            --max_tokens "$MAX_TOKENS"
        echo "Done -> ${DUAL_DIR}/"
    fi

    # ── ADAPTIVE ──────────────────────────────────────────────────────────────
    ADAPTIVE_DIR="${UNI_DIR}/adaptive"
    ADAPTIVE_JSON="${ADAPTIVE_DIR}/monthly_adaptive_autogover_results.json"
    if [[ -f "$ADAPTIVE_JSON" ]]; then
        echo "  [skip] ADAPTIVE — results already exist"
    else
        echo ""
        echo "--- Running ADAPTIVE ---"
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

    # ── Per-universe plot + compare ───────────────────────────────────────────
    echo ""
    echo "--- Metrics + plot: ${UNIVERSE_TAG} ---"

    # Collect all result JSONs for this universe (all 4 systems)
    ALL_RESULTS=()
    for SUBDIR in memory debater dual adaptive; do
        while IFS= read -r -d '' f; do
            ALL_RESULTS+=("$f")
        done < <(find "${UNI_DIR}/${SUBDIR}" -name "*_results.json" -print0 2>/dev/null)
    done

    if [[ ${#ALL_RESULTS[@]} -gt 0 ]]; then
        python -m ace_harness.compare_results "${ALL_RESULTS[@]}" || true
    fi

    # plot_adaptive for the adaptive run
    if [[ -f "$ADAPTIVE_JSON" ]]; then
        BASELINE_ARGS=()
        MEM_JSON=$(find "${MEMORY_DIR}" -name "*_results.json" 2>/dev/null | head -1)
        DEB_JSON=$(find "${DEBATER_DIR}" -name "*_results.json" 2>/dev/null | head -1)
        DUAL_JSON=$(find "${DUAL_DIR}" -name "*_results.json" 2>/dev/null | head -1)
        [[ -n "$MEM_JSON" ]] && BASELINE_ARGS+=("$MEM_JSON")
        [[ -n "$DEB_JSON" ]] && BASELINE_ARGS+=("$DEB_JSON")
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

# ── Cross-universe summary ─────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "CROSS-UNIVERSE METRICS SUMMARY"
echo "=========================================="

python - <<'PYEOF'
import glob, json, math, os, statistics

out_root = os.environ.get("OUT_ROOT", "")
universes = ["tech18", "mag7", "balanced15", "sp30", "diversified40", "volatile25"]
# (subdir, label)  — filename resolved by glob inside each subdir
systems = [
    ("memory",   "MEMORY"),
    ("debater",  "DEBATER"),
    ("dual",     "DUAL"),
    ("adaptive", "ADAPTIVE"),
]

def load_result(subdir, uni):
    """Return (path, data) for the first *_results.json found, or (None, None)."""
    matches = glob.glob(os.path.join(out_root, uni, subdir, "*_results.json"))
    if not matches:
        return None, None
    path = matches[0]
    with open(path) as f:
        return path, json.load(f)

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

# ── Performance table ──────────────────────────────────────────────────────────
hdr = f"{'Universe':<15} {'System':<10} {'TotalRet':>9} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7}"
print(hdr)
print("-" * len(hdr))
for uni in universes:
    for subdir, label in systems:
        _, data = load_result(subdir, uni)
        if data is None:
            continue
        m = perf_metrics(data)
        if m:
            tr, sh, dd, ca = m
            print(f"{uni:<15} {label:<10} {tr:>8.2f}% {sh:>7.2f} {dd:>7.2f}% {ca:>7.2f}")
    print()

# ── Token-usage table ──────────────────────────────────────────────────────────
print()
print("TOKEN USAGE PER SYSTEM × UNIVERSE")
thdr = f"{'Universe':<15} {'System':<10} {'Calls':>7} {'Prompt':>12} {'Completion':>12} {'Total':>12}"
print(thdr)
print("-" * len(thdr))

grand = {s: {"calls": 0, "prompt": 0, "completion": 0, "total": 0} for _, s in systems}

for uni in universes:
    for subdir, label in systems:
        _, data = load_result(subdir, uni)
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
for _, label in systems:
    g = grand[label]
    if g["calls"] == 0:
        continue
    print(f"{'ALL':<15} {label:<10} {g['calls']:>7,} {g['prompt']:>12,} {g['completion']:>12,} {g['total']:>12,}")

PYEOF

echo "=========================================="
echo "MDA RUN COMPLETE"
echo "Results: ${OUT_ROOT}/"
echo "=========================================="
