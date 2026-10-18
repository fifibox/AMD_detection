#!/bin/bash
# submit_all.sh -- submit training (29-line array) and the analysis job that waits for it.
#   bash scripts/submit_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
JID=$(sbatch --parsable slurm/job.slurm)
echo "training array: $JID"
AID=$(sbatch --parsable --dependency=afterany:$JID slurm/job_analysis.slurm)
echo "analysis job:   $AID (starts after every array task has ended)"
echo "watch with:     squeue -u \$USER    |    squeue --start -j $JID"
