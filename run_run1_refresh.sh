#!/usr/bin/env bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/run1_refresh_%j.out
#SBATCH --error=log/run1_refresh_%j.out
#SBATCH --job-name=ACE_RUN1_REFRESH
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=interactive-gpu

# Re-runs the RUN 1 table cells that the fixes invalidated, then prints the
# table in the same 14 columns (paste into Sheets). Temperature stays 0.0.
#
# Which cells (default):
#   * sp30 + diversified40, ALL 7 systems — past 26 tickers the old asset labels
#     became ASSET_[ / ASSET_] ..., so every script (QWEN/REACT/HARNESS too)
#     parsed only 26 allocation keys.
#   * ADAP in all 6 universes — the ADAP row is now ADAP-v2 (--systems adap_v2).
#   * MEMORY, DEBATER, DUAL in tech18/mag7/balanced15/volatile25 —
#       MEMORY + DUAL: re-run without the RiskTuner (RISK_TUNER=off), so every
#         system shares the same fixed risk harness (clean ablation);
#       DEBATER: the shared Debater prompt changed on 2026-09-22 22:57 UTC
#         (b2c9a69); re-run so it is guaranteed to match the code ADAP-v2 uses.
#   Reused unchanged: QWEN / REACT / HARNESS in the 4 small universes
#   (ReAct scripts untouched since 2026-09-17 apart from the label fix, which
#   does not affect universes of <= 26 tickers).
#
# Options (environment variables):
#   RISK_TUNER=off|on   off (default): MEMORY/DUAL/ADAP all run WITHOUT the RiskTuner.
#                       on: memory systems keep the RiskTuner (then disclose it in the
#                       paper); MEMORY/DUAL in the 4 small universes are not re-run.
#   REUSE_DEBATER=1     do not re-run DEBATER in the 4 small universes — only if job
#                       1706716 ran on a checkout that already contained b2c9a69.
#   FULL=1              re-run all 42 cells from this commit (maximum consistency).
#   EXTRAS=1            also run REMO (faithful ReMo Alg. 1) and DEBATER+STOP rows.
#   RUN_LABEL=run1      output subdirectory; use run2 / run3 for replicates (FULL=1).
#   BASE_TSV=path       your current RUN 1 table (TSV, same columns): the printed
#                       table merges the new cells into it.
#   OUT_ROOT=dir        default ${ROOT_DIR}/run1_refresh_<jobid>; re-submit with the same
#                       OUT_ROOT to resume (finished cells are skipped).

set -euo pipefail

mkdir -p "$HOME/log"
module load gcc/11.3.0
module load miniforge

ROOT_DIR="/users/2/cai00317/trading-harness"
REACT_DIR="${ROOT_DIR}/ReAct"
cd "$ROOT_DIR"
source "${REACT_DIR}/venv/bin/activate"

export HF_HOME="$HOME/hf_cache"
export VLLM_USE_FLASHINFER_SAMPLER=0
unset ACE_TEMPERATURE

MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"
VLLM_PORT=8000
export REACT_MODEL="$MODEL"
export ACE_LLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export ACE_SOLVER_MODEL="$MODEL"
export ACE_DEBATER_MODEL="$MODEL"
export ACE_CONSOLIDATOR_MODEL="$MODEL"

INITIAL_CAPITAL=1000000
REBALANCE_DAYS=20
START_DATE="${START_DATE:-2024-01-01}"
END_DATE="${END_DATE:-2024-12-31}"
MAX_TOKENS=800
RISK_TUNER="${RISK_TUNER:-off}"
REUSE_DEBATER="${REUSE_DEBATER:-0}"
FULL="${FULL:-0}"
EXTRAS="${EXTRAS:-0}"
RUN_LABEL="${RUN_LABEL:-run1}"
JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/run1_refresh_${JOB_ID}}"
RUN_DIR="${OUT_ROOT}/${RUN_LABEL}"
mkdir -p "$RUN_DIR"

case "$RISK_TUNER" in
    off) RT_FLAG="--no_adaptive_risk" ;;
    on)  RT_FLAG="" ;;
    *)   echo "RISK_TUNER must be off or on" >&2; exit 1 ;;
esac

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "WARNING: working tree has uncommitted changes; manifests will record git_dirty=true"
fi
echo "commit $(git rev-parse HEAD)" | tee "${RUN_DIR}/COMMIT"

SMALL=(tech18 mag7 balanced15 volatile25)
LARGE=(sp30 diversified40)
ALL_UNIVERSES=(tech18 mag7 balanced15 sp30 diversified40 volatile25)
ALL_SYSTEMS=(qwen react harness memory debater dual adap)

# ── Cell list: "universe:system" ─────────────────────────────────────────────
CELLS=()
add() { CELLS+=("$1:$2"); }
if [[ "$FULL" == "1" ]]; then
    for u in "${ALL_UNIVERSES[@]}"; do for s in "${ALL_SYSTEMS[@]}"; do add "$u" "$s"; done; done
else
    for u in "${LARGE[@]}"; do for s in "${ALL_SYSTEMS[@]}"; do add "$u" "$s"; done; done
    for u in "${SMALL[@]}"; do
        add "$u" adap
        if [[ "$RISK_TUNER" == "off" ]]; then add "$u" memory; add "$u" dual; fi
        if [[ "$REUSE_DEBATER" != "1" ]]; then add "$u" debater; fi
    done
fi
if [[ "$EXTRAS" == "1" ]]; then
    for u in "${ALL_UNIVERSES[@]}"; do add "$u" remo; add "$u" debater_stop; done
fi

get_tickers() { python -m ace_harness.universes "$1"; }

cell_done() {  # a cell is done when its results exist (and, for ace systems, its manifest is complete)
    local dir="$1" sys="$2"
    case "$sys" in
        qwen|react|harness) compgen -G "${dir}/*.json" >/dev/null 2>&1 ;;
        *) python - "$dir" <<'PYEOF'
import glob, json, sys
from ace_harness.accounting import sidecar_paths
files = glob.glob(sys.argv[1] + "/*_results.json")
ok = len(files) == 1 and json.load(open(sidecar_paths(files[0])["manifest"])).get("status") == "complete"
sys.exit(0 if ok else 1)
PYEOF
        ;;
    esac
}

run_cell() {
    local uni="$1" sys="$2"
    local dir="${RUN_DIR}/${uni}/${sys}"
    read -ra TICKERS <<< "$(get_tickers "$uni")"
    if cell_done "$dir" "$sys" 2>/dev/null; then echo "  [skip] ${uni}/${sys}"; return; fi
    mkdir -p "$dir"
    echo "  [run]  ${uni}/${sys}  (${#TICKERS[@]} tickers)"
    local common=(--tickers "${TICKERS[@]}" --start "$START_DATE" --end "$END_DATE")
    local ace=(python -m ace_harness.run_monthly "${common[@]}" --output_dir "$dir"
               --risk_harness_type conviction --initial_capital "$INITIAL_CAPITAL"
               --fallback_mode equal_weight --max_tokens "$MAX_TOKENS")
    local react=("${common[@]}" --initial-capital "$INITIAL_CAPITAL" --rebalance-days "$REBALANCE_DAYS"
                 --warmup-days 0 --output "${dir}/${sys}_results_raw.json")
    case "$sys" in
        qwen)    (cd "$REACT_DIR" && python qwen_port.py "${react[@]}") ;;
        react)   (cd "$REACT_DIR" && python noharness.py "${react[@]}") ;;
        harness) (cd "$REACT_DIR" && python yesharnessgpt.py "${react[@]}") ;;
        memory)  "${ace[@]}" --systems memory_only $RT_FLAG ;;
        debater) "${ace[@]}" --systems intra ;;
        dual)    "${ace[@]}" --systems dual $RT_FLAG ;;
        adap)    "${ace[@]}" --systems adap_v2 $RT_FLAG ;;
        remo)    "${ace[@]}" --systems remo $RT_FLAG ;;
        debater_stop) "${ace[@]}" --systems debater_stop ;;
    esac
}

echo "=========================================="
echo "RUN 1 REFRESH  (${RUN_LABEL})"
echo "Model: ${MODEL}  temperature 0.0  |  RiskTuner: ${RISK_TUNER}  |  FULL=${FULL}  EXTRAS=${EXTRAS}"
echo "Period: ${START_DATE} -> ${END_DATE}  |  Output: ${RUN_DIR}"
echo "Cells (${#CELLS[@]}): ${CELLS[*]}"
echo "=========================================="

# ── vLLM (same server flags as the original runs) ────────────────────────────
VLLM_PID=""
cleanup() { [[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null && wait "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT
vllm serve "$MODEL" --host 127.0.0.1 --port "$VLLM_PORT" --tensor-parallel-size 1 \
    --max-model-len 32768 --gpu-memory-utilization 0.90 --enable-chunked-prefill \
    --enable-auto-tool-choice --tool-call-parser hermes &
VLLM_PID=$!
START_T=$(date +%s)
until curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vLLM died"; exit 1; }
    (( $(date +%s) - START_T >= 1800 )) && { echo "ERROR: vLLM startup timed out"; exit 1; }
    sleep 5
done
SERVED=$(curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
[[ "$SERVED" == "$MODEL" ]] || { echo "ERROR: serving ${SERVED}, expected ${MODEL}"; exit 1; }
echo "vLLM online: ${SERVED}"

for cell in "${CELLS[@]}"; do
    run_cell "${cell%%:*}" "${cell##*:}"
done

# ── Table ────────────────────────────────────────────────────────────────────
TITLE="$(echo "$RUN_LABEL" | sed -E 's/^run/RUN /')"
TABLE_ARGS=(--root "$RUN_DIR" --title "$TITLE" --out "${RUN_DIR}/table.tsv")
[[ -n "${BASE_TSV:-}" ]] && TABLE_ARGS+=(--base "$BASE_TSV")
echo ""
echo "=========================================="
echo "TABLE (tab-separated; also written to ${RUN_DIR}/table.tsv)"
echo "=========================================="
python -m ace_harness.make_table "${TABLE_ARGS[@]}"
