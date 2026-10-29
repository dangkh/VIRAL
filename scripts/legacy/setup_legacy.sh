#!/bin/bash
# Re-run the original jepa2 code (with / without the JEPA loss) over several seeds,
# as exactly 4 PBS jobs, to check whether jepa2 vs jepa2_noloss differences are noise.
#
#   bash scripts/legacy/setup_legacy.sh            # create worktrees + submit 4 jobs
#   DRY_RUN=1 bash scripts/legacy/setup_legacy.sh  # only create worktrees, print qsub commands
#   python scripts/legacy/collect_legacy.py        # per-seed table + mean/std when done
#
# Seed 999 already exists (VIRAL_jepa2, VIRAL_jepa2_noloss) and is RE-RUN once per variant
# to measure same-seed GPU non-determinism.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd); BASE=$(dirname "$REPO")
DRY_RUN="${DRY_RUN:-0}"
CONDA_ENV="${CONDA_ENV:-Astar}"

git fetch -q origin jepa2

#        worktree name           variant  seeds
JOBS=("legacy_jepa2_a          loss     1,2"
      "legacy_jepa2_b          loss     3,4,999"
      "legacy_jepa2_noloss_a   noloss   1,2"
      "legacy_jepa2_noloss_b   noloss   3,4,999")

for J in "${JOBS[@]}"; do
    read -r NAME VARIANT SEEDS <<< "$J"
    WT="$BASE/$NAME"
    if [ ! -d "$WT" ]; then
        git worktree add -q --detach "$WT" origin/jepa2
    fi
    # seeds for quick_start (runs them one after another in the same job)
    sed -i "s/^seed:.*/seed: [${SEEDS//,/, }]/" "$WT/src/configs/overall.yaml"
    if [ "$VARIANT" = "noloss" ]; then
        sed -i 's/return loss_value + reg_loss + self.jepaLoss$/return loss_value + reg_loss + 0.0 * self.jepaLoss/' \
            "$WT/src/models/vlif.py"
    fi
    mkdir -p "$WT/data"
    ln -sfn "$REPO/data/baby" "$WT/data/baby"
    printf '%-24s %-7s %-12s | %s | %s\n' "$NAME" "$VARIANT" "$SEEDS" \
        "$(grep '^seed:' "$WT/src/configs/overall.yaml")" \
        "$(grep -o 'return loss_value.*' "$WT/src/models/vlif.py")"

    CMD=(qsub -q small_gpuq -N "${NAME#legacy_}" -v "WD=$WT,CONDA_ENV=$CONDA_ENV" scripts/legacy/run_legacy.pbs)
    if [ "$DRY_RUN" = "1" ]; then echo "  [dry-run] ${CMD[*]}"; else echo "  submitted: $("${CMD[@]}")"; fi
done
echo "When finished: python scripts/legacy/collect_legacy.py"
