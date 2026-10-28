#!/bin/bash
# Submit experiments as at most MAX_JOBS PBS jobs (default 4, the per-user limit).
# Experiments are spread round-robin over the jobs; each job runs its share sequentially.
#
#   bash scripts/submit_all.sh                                   # baby, scripts/experiments_baby.txt
#   bash scripts/submit_all.sh baby scripts/experiments_baby_seeds.txt
#   ONLY="vpid vpid_detach" bash scripts/submit_all.sh           # only some experiments (by name)
#   MAX_JOBS=2 bash scripts/submit_all.sh                        # fewer jobs (e.g. 2 already running)
#   HOURS_PER_EXP=3 bash scripts/submit_all.sh                   # walltime = HOURS_PER_EXP x #experiments per job
#   WALLTIME=24:00:00 bash scripts/submit_all.sh                 # or set the walltime directly
#   DRY_RUN=1 bash scripts/submit_all.sh                         # print only
#
# If data/<dataset>/cf_item_emb.npy is missing, the first job pretrains Y before its own
# experiments and the other jobs start only after it finished (they wait on a marker file).
# Re-running this script skips experiments that already have a finished log.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-baby}"
CFG_FILE="${2:-scripts/experiments_${DATASET}.txt}"
CONDA_ENV="${CONDA_ENV:-Astar}"
MAX_JOBS="${MAX_JOBS:-4}"
HOURS_PER_EXP="${HOURS_PER_EXP:-4}"
WALLTIME="${WALLTIME:-}"
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-0}"

[ -f "$CFG_FILE" ] || { echo "Experiment file not found: $CFG_FILE"; exit 1; }

# experiments to run: not commented, selected by ONLY, without a finished log
mapfile -t ALL < <(grep -v '^[[:space:]]*#' "$CFG_FILE" | tr -d '\r' | grep -v '^[[:space:]]*$')
TODO=()
for L in "${ALL[@]}"; do
    NAME=$(echo "$L" | awk '{print $1}')
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $NAME "* ]]; then continue; fi
    if grep -l 'BEST ████' log/VLIF-"${DATASET}"-"${NAME}"-*.log >/dev/null 2>&1; then
        echo "  skip $NAME (finished log exists)"; continue
    fi
    TODO+=("$L")
done
N=${#TODO[@]}
[ "$N" -gt 0 ] || { echo "Nothing to run."; exit 0; }

# warn about jobs already in the queue
if command -v qstat >/dev/null 2>&1; then
    MINE=$(qstat -u "$USER" 2>/dev/null | awk 'NR>5 && $10 ~ /^[QRHW]$/' | wc -l || true)
    if [ "${MINE:-0}" -gt 0 ]; then
        echo "Note: you already have $MINE job(s) queued/running; consider MAX_JOBS=$((MAX_JOBS - MINE))."
    fi
fi

W=$(( N < MAX_JOBS ? N : MAX_JOBS ))
STAMP=$(date '+%Y%m%d-%H%M%S')
QDIR="log/queue/${DATASET}-${STAMP}"
mkdir -p "$QDIR"
for ((w = 0; w < W; w++)); do : > "$QDIR/worker${w}.txt"; done
for i in "${!TODO[@]}"; do echo "${TODO[$i]}" >> "$QDIR/worker$((i % W)).txt"; done

NEED_Y=0
[ -f "data/${DATASET}/cf_item_emb.npy" ] || NEED_Y=1

echo "Dataset $DATASET | $N experiments from $CFG_FILE -> $W job(s) | task files in $QDIR"
FIRST_ID=""
for ((w = 0; w < W; w++)); do
    K=$(grep -c . "$QDIR/worker${w}.txt")
    if [ -n "$WALLTIME" ]; then WT="$WALLTIME"; else
        H=$(( K * HOURS_PER_EXP + NEED_Y )); WT=$(printf '%02d:00:00' "$H"); fi
    VARS="TASK_FILE=$QDIR/worker${w}.txt,DATASET=${DATASET},CONDA_ENV=${CONDA_ENV}"
    DEP=()
    if [ "$NEED_Y" = "1" ]; then
        if [ "$w" = "0" ]; then VARS="${VARS},RUN_PRETRAIN=1"
        elif [ -n "$FIRST_ID" ]; then DEP=(-W "depend=after:${FIRST_ID}"); fi
    fi
    CMD=(qsub -N "${DATASET:0:2}_w${w}_${STAMP: -4}" -l "walltime=${WT}" ${DEP[@]+"${DEP[@]}"} -v "$VARS" scripts/run_worker.pbs)
    if [ "$DRY_RUN" = "1" ]; then echo "[dry-run] ${CMD[*]}"; JID="DRYRUN.w${w}"; else JID=$("${CMD[@]}"); fi
    [ "$w" = "0" ] && FIRST_ID="$JID"
    printf '  job %d (%s, walltime %s): %s\n' "$w" "$JID" "$WT" "$(awk '{print $1}' "$QDIR/worker${w}.txt" | xargs)"
done
if [ "$NEED_Y" = "1" ]; then
    echo "Y missing: job 0 pretrains it first. Other jobs depend on job 0 having STARTED;"
    echo "they wait for data/${DATASET}/cf_item_emb.npy before running (see run_worker.pbs)."
fi
echo "Progress: tail -f <jobname>.o<id>   Results: python scripts/collect_results.py [--group_seeds]"
