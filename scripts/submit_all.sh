#!/bin/bash
# Submit one PBS job per experiment line.
#
#   bash scripts/submit_all.sh                     # all experiments for baby
#   bash scripts/submit_all.sh baby                # same
#   bash scripts/submit_all.sh sports scripts/experiments_sports.txt
#   ONLY="vpid pid_ssl" bash scripts/submit_all.sh # only some experiments (by name)
#   DRY_RUN=1 bash scripts/submit_all.sh           # print the qsub commands only
#
# If data/<dataset>/cf_item_emb.npy does not exist, a pretrain job is submitted
# first and every experiment waits for it (afterok dependency).
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-baby}"
CFG_FILE="${2:-scripts/experiments_${DATASET}.txt}"
CONDA_ENV="${CONDA_ENV:-Astar}"
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-0}"

[ -f "$CFG_FILE" ] || { echo "Experiment file not found: $CFG_FILE"; exit 1; }

run() { if [ "$DRY_RUN" = "1" ]; then echo "[dry-run] $*" >&2; echo "DRYRUN.0"; else "$@"; fi; }

DEPEND=()
Y_FILE="data/${DATASET}/cf_item_emb.npy"
if [ ! -f "$Y_FILE" ]; then
    echo "Missing $Y_FILE -> submitting pretrain job first"
    PRE_ID=$(run qsub -N "cf_${DATASET}" -v "DATASET=${DATASET},CONDA_ENV=${CONDA_ENV}" scripts/pretrain_cf.pbs)
    echo "  pretrain: $PRE_ID"
    DEPEND=(-W "depend=afterok:${PRE_ID}")
fi

mapfile -t LINES < <(grep -v '^[[:space:]]*#' "$CFG_FILE" | tr -d '\r' | grep -v '^[[:space:]]*$')
echo "Submitting from $CFG_FILE (${#LINES[@]} experiments, dataset=$DATASET)"
for i in "${!LINES[@]}"; do
    NAME=$(echo "${LINES[$i]}" | awk '{print $1}')
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $NAME "* ]]; then continue; fi
    JOBNAME=$(echo "${DATASET:0:2}_${NAME}" | tr -c 'A-Za-z0-9_.\n-' '_' | cut -c1-15)
    JID=$(run qsub -N "$JOBNAME" ${DEPEND[@]+"${DEPEND[@]}"} \
        -v "CFG_IDX=${i},CFG_FILE=${CFG_FILE},DATASET=${DATASET},CONDA_ENV=${CONDA_ENV}" \
        scripts/run_one.pbs)
    printf '  [%2d] %-16s -> %s\n' "$i" "$NAME" "$JID"
done
echo "Logs: log/VLIF-${DATASET}-<name>-<time>.log   Summary: python scripts/collect_results.py"
