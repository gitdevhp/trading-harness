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
#                       Note: ADAP-v2's memory freeze also pauses its RiskTuner updates.
#                       A RUN_DIR started with one setting refuses to resume with the other.
#   REUSE_DEBATER=1     do not re-run DEBATER in the 4 small universes — only if job
#                       1706716 ran on a checkout that already contained b2c9a69.
#   FULL=1              all 42 cells, i.e. QWEN/REACT/HARNESS in every universe as well. Replicates
#                       use FULL=1, and summarize_runs needs every run to hold the same cells, so
#                       complete a FULL=0 run by re-submitting it with the same OUT_ROOT and
#                       RUN_LABEL plus FULL=1 (finished cells are skipped; only the rest run).
#   EXTRAS=1            also run the rows that tie the extension to the peer's paper:
#                       REMO (faithful ReMo Alg. 1), AUTOGOV (faithful AutoGovern Alg. 2;
#                       ADAP = AUTOGOV + outcome grounding) and DEBATER+STOP (ADAP's stops
#                       without memory).
#   RUN_LABEL=run1      output subdirectory; use run2 / run3 for replicates (FULL=1).
#   ONLY="u:s u:s"      run exactly these cells instead (e.g. ONLY="mag7:adap sp30:qwen"
#                       END_DATE=2024-03-31 for a quick GPU smoke test).
#   BASE_TSV=path       your current RUN 1 table (TSV, same columns): the printed
#                       table merges the new cells into it.
#   OUT_ROOT=dir        default ${ROOT_DIR}/run1_refresh_<jobid>; re-submit with the same
#                       OUT_ROOT to resume (finished cells are skipped). A RUN_DIR remembers its
#                       dates, model and budgets (RUN_CONFIG) and refuses to resume with others.
#
# Submit from the repo root (SLURM writes log/run1_refresh_<jobid>.out there; log/ must exist).
# Every job serves vLLM on port 8000, so chain several runs instead of running them together:
#   j=$(OUT_ROOT=$PWD/refresh_2024 RUN_LABEL=run2 FULL=1 EXTRAS=1 sbatch --parsable run_run1_refresh.sh)
#   OUT_ROOT=$PWD/refresh_2024 RUN_LABEL=run3 FULL=1 EXTRAS=1 sbatch --dependency=afterany:$j run_run1_refresh.sh
# After a run: python -m ace_harness.audit_run <RUN_DIR> checks every cell from its sidecars.

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
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"          # absolute: ReAct scripts run from REACT_DIR
RUN_DIR="${OUT_ROOT}/${RUN_LABEL}"
mkdir -p "$RUN_DIR"
if [[ -n "${BASE_TSV:-}" ]]; then
    [[ -f "$BASE_TSV" ]] || { echo "BASE_TSV not found: $BASE_TSV" >&2; exit 1; }
    BASE_TSV="$(cd "$(dirname "$BASE_TSV")" && pwd)/$(basename "$BASE_TSV")"
fi

case "$RISK_TUNER" in
    off) RT_FLAG="--no_adaptive_risk" ;;
    on)  RT_FLAG="" ;;
    *)   echo "RISK_TUNER must be off or on" >&2; exit 1 ;;
esac
if [[ -f "${RUN_DIR}/RISK_TUNER" && "$(cat "${RUN_DIR}/RISK_TUNER")" != "$RISK_TUNER" ]]; then
    echo "ERROR: ${RUN_DIR} was started with RISK_TUNER=$(cat "${RUN_DIR}/RISK_TUNER"); resume with the same" \
         "setting or use a new OUT_ROOT/RUN_LABEL" >&2
    exit 1
fi
echo "$RISK_TUNER" > "${RUN_DIR}/RISK_TUNER"
export RISK_TUNER
RUN_CONFIG="start=${START_DATE} end=${END_DATE} model=${MODEL} max_tokens=${MAX_TOKENS} capital=${INITIAL_CAPITAL} rebalance_days=${REBALANCE_DAYS}"
if [[ -f "${RUN_DIR}/RUN_CONFIG" && "$(cat "${RUN_DIR}/RUN_CONFIG")" != "$RUN_CONFIG" ]]; then
    echo "ERROR: ${RUN_DIR} was started with '$(cat "${RUN_DIR}/RUN_CONFIG")', not '${RUN_CONFIG}';" \
         "use a new OUT_ROOT/RUN_LABEL" >&2
    exit 1
fi
echo "$RUN_CONFIG" > "${RUN_DIR}/RUN_CONFIG"

DIRTY=0
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "WARNING: working tree has uncommitted changes; manifests will record git_dirty=true" \
         "(diff saved to ${RUN_DIR}/git_diff_${JOB_ID}.patch)"
    DIRTY=1
fi
COMMIT="$(git rev-parse HEAD)"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) job=${JOB_ID} commit=${COMMIT} risk_tuner=${RISK_TUNER}" \
    | tee -a "${RUN_DIR}/COMMIT"
if [[ "$DIRTY" == "1" ]]; then
    git diff HEAD > "${RUN_DIR}/git_diff_${JOB_ID}.patch"
fi
{   # software and weights this job ran with
    echo "job=${JOB_ID} host=$(hostname) commit=${COMMIT}"
    nvidia-smi -L 2>/dev/null || true
    echo "model_snapshot=$(cat "${HF_HOME}/hub/models--${MODEL//\//--}/refs/main" 2>/dev/null || echo unknown)"
    pip freeze 2>/dev/null | grep -iE '^(vllm|torch|transformers|openai|yfinance|pandas|numpy)==' || true
} > "${RUN_DIR}/ENV_${JOB_ID}.txt"

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
    for u in "${ALL_UNIVERSES[@]}"; do add "$u" remo; add "$u" autogov; add "$u" debater_stop; done
fi
if [[ -n "${ONLY:-}" ]]; then
    read -ra CELLS <<< "$ONLY"
    for c in "${CELLS[@]}"; do
        [[ "$c" =~ ^(tech18|mag7|balanced15|sp30|diversified40|volatile25):(qwen|react|harness|memory|debater|dual|adap|remo|autogov|debater_stop)$ ]] \
            || { echo "ONLY: bad cell '$c' (want universe:system)" >&2; exit 1; }
    done
fi

get_tickers() { python -m ace_harness.universes "$1"; }

cell_done() {  # 0 = done, 1 = (re)run it, 2 = inconsistent cell: abort
              # done = results exist; ace cells also need a complete manifest with the requested RiskTuner
    local dir="$1" sys="$2"
    case "$sys" in
        qwen|react|harness) [[ -f "${dir}/PROVENANCE" ]] && compgen -G "${dir}/*_results_raw.json" >/dev/null 2>&1 ;;
        *) python - "$dir" "$sys" <<'PYEOF'
import glob, json, os, sys
from ace_harness.accounting import sidecar_paths
cell, system = sys.argv[1], sys.argv[2]
files = glob.glob(cell + "/*_results.json")
if not files:
    sys.exit(1)
if len(files) > 1:
    print(f"ERROR: {cell} holds {len(files)} results files; remove the stale one", file=sys.stderr)
    sys.exit(2)
manifest = json.load(open(sidecar_paths(files[0])["manifest"]))
want_rt = os.environ["RISK_TUNER"] == "on" and system in ("memory", "dual", "adap", "remo", "autogov")
if manifest.get("status") == "complete" and manifest.get("risk_tuner") != want_rt:
    print(f"ERROR: {cell} was run with risk_tuner={manifest.get('risk_tuner')}; use a new OUT_ROOT", file=sys.stderr)
    sys.exit(2)
sys.exit(0 if manifest.get("status") == "complete" else 1)
PYEOF
        ;;
    esac
}

run_cell() {
    local uni="$1" sys="$2"
    local dir="${RUN_DIR}/${uni}/${sys}"
    read -ra TICKERS <<< "$(get_tickers "$uni")"
    local rc=0
    cell_done "$dir" "$sys" || rc=$?
    if (( rc == 0 )); then echo "  [skip] ${uni}/${sys}"; return; fi
    if (( rc == 2 )); then exit 1; fi
    mkdir -p "$dir"
    echo "  [run]  ${uni}/${sys}  (${#TICKERS[@]} tickers)"
    local common=(--tickers "${TICKERS[@]}" --start "$START_DATE" --end "$END_DATE")
    local ace=(python -m ace_harness.run_monthly "${common[@]}" --output_dir "$dir"
               --risk_harness_type conviction --initial_capital "$INITIAL_CAPITAL"
               --fallback_mode equal_weight --max_tokens "$MAX_TOKENS")
    local react=("${common[@]}" --initial-capital "$INITIAL_CAPITAL" --rebalance-days "$REBALANCE_DAYS"
                 --warmup-days 0 --output "${dir}/${sys}_results_raw.json")
    local react_script=""
    case "$sys" in
        qwen)    react_script=qwen_port.py ;;
        react)   react_script=noharness.py ;;
        harness) react_script=yesharnessgpt.py ;;
    esac
    case "$sys" in
        qwen|react|harness)
            rm -f "${dir}"/*.json
            (cd "$REACT_DIR" && python "$react_script" "${react[@]}")
            echo "commit=${COMMIT} script=${react_script} model=${MODEL} date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                > "${dir}/PROVENANCE" ;;
        memory)  "${ace[@]}" --systems memory_only $RT_FLAG ;;
        debater) "${ace[@]}" --systems intra ;;
        dual)    "${ace[@]}" --systems dual $RT_FLAG ;;
        adap)    "${ace[@]}" --systems adap_v2 $RT_FLAG ;;
        remo)    "${ace[@]}" --systems remo $RT_FLAG ;;
        autogov) "${ace[@]}" --systems autogov $RT_FLAG ;;
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
if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "ERROR: a server is already listening on port ${VLLM_PORT}; refusing to run on a server this job" \
         "does not own" >&2
    exit 1
fi
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
EXPECT="$(for c in "${CELLS[@]}"; do printf '%s ' "$c"; done)"
TABLE_ARGS=(--root "$RUN_DIR" --title "$TITLE" --out "${RUN_DIR}/table.tsv" --expect "$EXPECT")
[[ -n "${BASE_TSV:-}" ]] && TABLE_ARGS+=(--base "$BASE_TSV")
echo ""
echo "=========================================="
echo "TABLE (tab-separated; also written to ${RUN_DIR}/table.tsv)"
echo "=========================================="
python -m ace_harness.make_table "${TABLE_ARGS[@]}"
