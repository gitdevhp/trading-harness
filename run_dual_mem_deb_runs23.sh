#!/usr/bin/env bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/runs23_%j.out
#SBATCH --error=log/runs23_%j.out
#SBATCH --job-name=ACE_RUNS23
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=interactive-gpu

# Runs 2 more independent replicates (run2, run3) of:
#   DUAL (fixed ReMo)  |  DEBATER (intra)  |  MEMORY (memory_only)
# across all 6 universes, with temperature=0.3 to introduce real stochastic
# variance between replicates (temperature=0.0 = greedy = identical every time).
#
# Output:
#   {OUT_ROOT}/run2/{universe}/{dual,debater,memory}/  (replicate 2)
#   {OUT_ROOT}/run3/{universe}/{dual,debater,memory}/  (replicate 3)
#
# Token usage is saved inside each *_results.json under "token_usage".
# A TSV summary (copy-pasteable into Google Sheets) is printed at the end.
#
# Resume: each step checks for existing *_results.json and skips if found.

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

# ── Configurable parameters ───────────────────────────────────────────────────
INITIAL_CAPITAL=1000000
START_DATE="${START_DATE:-2024-01-01}"
END_DATE="${END_DATE:-2024-12-31}"
MAX_TOKENS="${MAX_TOKENS:-800}"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/ace_runs23_${JOB_ID}}"

# ── Universe definitions ──────────────────────────────────────────────────────
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
echo "ACE — DUAL + DEBATER + MEMORY  (runs 2 & 3)"
echo "=========================================="
echo "Job:          ${JOB_ID}"
echo "Model:        ${MODEL}"
echo "Systems:      dual | intra (debater) | memory_only"
echo "Universes:    ${UNIVERSE_TAGS[*]}"
echo "Period:       ${START_DATE} -> ${END_DATE}"
echo "Capital:      \$${INITIAL_CAPITAL}"
echo "Max tokens:   ${MAX_TOKENS}"
echo "Output:       ${OUT_ROOT}/"
echo "=========================================="

# ── Determine if any LLM work is needed ──────────────────────────────────────
NEED_LLM=false
for RUN in run2 run3; do
    for TAG in "${UNIVERSE_TAGS[@]}"; do
        for SUBDIR in dual debater memory; do
            DIR="${OUT_ROOT}/${RUN}/${TAG}/${SUBDIR}"
            if ! compgen -G "${DIR}/*_results.json" >/dev/null 2>&1; then
                NEED_LLM=true
                break 3
            fi
        done
    done
done

# ── Start vLLM ───────────────────────────────────────────────────────────────
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
    echo "Starting vLLM server (seed=42 for this batch)..."
    vllm serve "$MODEL" \
        --host 127.0.0.1 \
        --port "$VLLM_PORT" \
        --seed 42 \
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

# ── Main loop: run2 then run3, each across all 6 universes ───────────────────
for RUN in run2 run3; do
    echo ""
    echo "######################################"
    echo "# Replicate: ${RUN}"
    echo "######################################"

    for UNIVERSE_TAG in "${UNIVERSE_TAGS[@]}"; do
        TICKERS=$(get_tickers "$UNIVERSE_TAG")
        read -ra TICKER_ARRAY <<< "$TICKERS"
        UNI_DIR="${OUT_ROOT}/${RUN}/${UNIVERSE_TAG}"
        mkdir -p "$UNI_DIR"

        echo ""
        echo "==========================================  ${RUN} / ${UNIVERSE_TAG}"

        # ── MEMORY ─────────────────────────────────────────────────────────
        MEMORY_DIR="${UNI_DIR}/memory"
        if compgen -G "${MEMORY_DIR}/*_results.json" >/dev/null 2>&1; then
            echo "  [skip] MEMORY"
        else
            echo "  Running MEMORY (memory_only)..."
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
            echo "  Done -> ${MEMORY_DIR}/"
        fi

        # ── DEBATER ────────────────────────────────────────────────────────
        DEBATER_DIR="${UNI_DIR}/debater"
        if compgen -G "${DEBATER_DIR}/*_results.json" >/dev/null 2>&1; then
            echo "  [skip] DEBATER"
        else
            echo "  Running DEBATER (intra)..."
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
            echo "  Done -> ${DEBATER_DIR}/"
        fi

        # ── DUAL ───────────────────────────────────────────────────────────
        DUAL_DIR="${UNI_DIR}/dual"
        if compgen -G "${DUAL_DIR}/*_results.json" >/dev/null 2>&1; then
            echo "  [skip] DUAL"
        else
            echo "  Running DUAL (fixed ReMo)..."
            mkdir -p "$DUAL_DIR"
            python -m ace_harness.run_monthly \
                --tickers "${TICKER_ARRAY[@]}" \
                --start "$START_DATE" \
                --end   "$END_DATE" \
                --systems dual \
                --output_dir "$DUAL_DIR" \
                --risk_harness_type conviction \
                --initial_capital "$INITIAL_CAPITAL" \
                --fallback_mode equal_weight \
                --max_tokens "$MAX_TOKENS"
            echo "  Done -> ${DUAL_DIR}/"
        fi

        echo "  ${RUN}/${UNIVERSE_TAG} complete."
    done
done

# ── TSV Summary (copy-paste into Google Sheets) ───────────────────────────────
echo ""
echo "=========================================="
echo "TSV SUMMARY  (tab-separated, paste into Sheets)"
echo "=========================================="

OUT_ROOT="$OUT_ROOT" python3 - <<'PYEOF'
import glob, json, math, os, sys
import numpy as np

TRADING_DAYS = 252
MODEL_LABEL = "qwen25"
SYSTEMS = [
    ("memory",  "HARNESS+MEMORY"),
    ("debater", "HARNESS+DEBATER"),
    ("dual",    "DUAL"),
]
UNIVERSES = ["tech18", "mag7", "balanced15", "sp30", "diversified40", "volatile25"]
RUNS = ["run2", "run3"]
OUT_ROOT = os.environ["OUT_ROOT"]

def load_json(path):
    with open(path) as f:
        return json.load(f)

def find_json(run, universe, subdir):
    pattern = os.path.join(OUT_ROOT, run, universe, subdir, "*_results.json")
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def daily_returns(values):
    return np.diff(values) / values[:-1]

def cagr(values):
    n = len(values) - 1
    if n <= 0 or values[0] <= 0:
        return float("nan")
    return float((values[-1] / values[0]) ** (TRADING_DAYS / n) - 1) * 100

def ann_vol(values):
    dr = daily_returns(values)
    return float(np.std(dr) * np.sqrt(TRADING_DAYS) * 100)

def sharpe(values):
    dr = daily_returns(values)
    std = np.std(dr)
    if std < 1e-12:
        return float("nan")
    return float(np.mean(dr) * TRADING_DAYS / (std * np.sqrt(TRADING_DAYS)))

def sortino(values):
    dr = daily_returns(values)
    neg = dr[dr < 0]
    downside = np.sqrt(np.mean(neg ** 2)) if len(neg) > 0 else 1e-9
    return float(np.mean(dr) * TRADING_DAYS / (downside * np.sqrt(TRADING_DAYS) + 1e-9))

def max_drawdown(values):
    peak = np.maximum.accumulate(values)
    return float(((values - peak) / peak).min() * 100)

def calmar(values):
    c = cagr(values)
    mdd = abs(max_drawdown(values))
    return float(c / (mdd + 1e-9)) if not math.isnan(c) else float("nan")

def fmt(v, decimals=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{round(v, decimals)}"

header = "\t".join([
    "Model", "System", "Universe",
    "TotalRet%", "CAGR%", "AnnVol%", "Sharpe", "Sortino", "Calmar", "MaxDD%",
    "LLM_Calls", "Prompt_Tokens", "Completion_Tokens", "Total_Tokens"
])

for run in RUNS:
    print(f"\n{run.upper()}")
    print(header)
    for universe in UNIVERSES:
        for subdir, label in SYSTEMS:
            path = find_json(run, universe, subdir)
            if path is None:
                print(f"\t# MISSING: {run}/{universe}/{subdir}", file=sys.stderr)
                continue
            data = load_json(path)
            records = data if isinstance(data, list) else data.get("daily_nav", data.get("results", []))
            vals = np.array([r["portfolio_value"] for r in records], dtype=float)
            total_ret = float((vals[-1] / vals[0] - 1) * 100)
            tok = data.get("token_usage", {}) if isinstance(data, dict) else {}
            calls = tok.get("calls", "")
            prompt = tok.get("prompt", "")
            compl = tok.get("completion", "")
            total = tok.get("total", "")

            row = "\t".join([
                MODEL_LABEL, label, universe,
                fmt(total_ret), fmt(cagr(vals)), fmt(ann_vol(vals)),
                fmt(sharpe(vals), 2), fmt(sortino(vals), 2), fmt(calmar(vals), 2),
                fmt(max_drawdown(vals)),
                str(calls), str(prompt), str(compl), str(total),
            ])
            print(row)

PYEOF

echo ""
echo "=========================================="
echo "RUNS 2 & 3 COMPLETE"
echo "Results: ${OUT_ROOT}/"
echo "=========================================="
