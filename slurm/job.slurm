#!/bin/bash
#
# Finished runs (summary.json) and finished folds are skipped, so just resubmit after a timeout.
# The logs/ folder must exist BEFORE sbatch (SLURM will not create it): mkdir -p logs
#
#SBATCH --job-name=amd_cv
#SBATCH --array=0-28%4
#SBATCH --partition=amdgpu
#SBATCH --gres=gpu:1         
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out


set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[ -f config/config.sh ] || { echo "submit from the project root (thesis_v2_hpc/), e.g. sbatch slurm/job.slurm"; exit 1; }
source config/config.sh
activate_env
cd "$SRC"

LINE=${SLURM_ARRAY_TASK_ID:-0}
echo "host $(hostname) | job ${SLURM_JOB_ID:-local} | line $LINE | $(date)"
nvidia-smi -L 2>/dev/null || rocm-smi --showproductname 2>/dev/null || true
python -c "import torch, timm, pydicom, sklearn; print('torch', torch.__version__, '| timm', timm.__version__, '| pydicom', pydicom.__version__, '| sklearn', sklearn.__version__, '| gpu', torch.cuda.get_device_name(0))"
resolve_amp

if [ "${SMOKE:-0}" = "1" ]; then
    python check_data.py --data_root "$DATA" --expected "$EXPECTED_N_IMAGES"
    python train_cv.py --data_root "$DATA" --task binary --arch retfound --strategy full \
        --checkpoint "$CKPT" --epochs 2 --warmup_epochs 1 --fold_idx 0 --amp "$AMP" \
        --num_workers "${SLURM_CPUS_PER_TASK:-8}" --output_dir "${TMPDIR:-/tmp}/_smoke_${USER:-$(id -un)}"
    echo "smoke test OK"
    exit 0
fi

python run_all.py --data_root "$DATA" --checkpoint "$CKPT" --runs "$RUNS" \
    --experiments "$EXPERIMENTS" --amp "$AMP" --num_workers "${SLURM_CPUS_PER_TASK:-8}" --lines "$LINE"
echo "done $(date)"
