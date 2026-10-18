#!/bin/bash
# slurm/job_analysis.slurm ( Run after training)
# Nothing here retrains a model; it reads the saved per-image probabilities in $RUNS.
# Submit from the project root
#
#SBATCH --job-name=amd_analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-32gb:1          # only AutoMorph M1 grading uses the GPU (~5 min)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -uo pipefail                        # no -e: one failing step should not stop the rest
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[ -f config/config.sh ] || { echo "submit from the project root (thesis_v2_hpc/), e.g. sbatch slurm/job.slurm"; exit 1; }
source config/config.sh
activate_env
cd "$SRC"
QDIR=$RESULTS/quality
mkdir -p "$QDIR"
step() { echo; echo "===== $* ====="; }

step "8. Dataset description (Section 3.2 / Table 3.3)"
python make_manifest.py --data_root "$DATA" --out "$RESULTS/manifest"

step "11. Tables (recomputed from saved probabilities)"
python aggregate.py --runs "$RUNS" --seed "$SEED"

step "11b. Paired patient-level bootstrap comparisons"
python compare.py --runs "$RUNS" --seed "$SEED" --n_boot 2000 --pairs \
    binary/resnet18_full_lr1e-3:binary/resnet18_full \
    binary/deit_tiny_patch16_224_full:binary/deit_tiny_patch16_224_full_lr1e-3 \
    binary/retfound_full_official:binary/resnet18_full_official \
    binary/retfound_full_official:binary/efficientnet_b0_full_official \
    binary/retfound_full_official:binary/deit_tiny_patch16_224_full_official \
    binary/resnet18_full_official:binary/resnet18_full_lr1e-3 \
    binary/deit_tiny_patch16_224_full_official:binary/deit_tiny_patch16_224_full \
    three_class/retfound_full_official:three_class/resnet18_full_official \
    three_class/retfound_full_official:three_class/efficientnet_b0_full_official \
    three_class/retfound_full_official:three_class/deit_tiny_patch16_224_full_official \
    binary/retfound_full_official:three_class/retfound_full_official \
    binary/retfound_full_official:four_class/retfound_full_official

step "13b. AutoMorph M1 quality grading"
if [ ! -f "$QDIR/m1_quality.csv" ] || [ ! -d "$QDIR/m0_crops" ]; then
    python quality_m1.py --data_root "$DATA" --automorph "$AUTOMORPH" \
        --out "$QDIR/m1_quality.csv" --save_m0_dir "$QDIR/m0_crops"
else
    echo "m1_quality.csv already exists - delete it to re-grade"
fi

step "13c. Quality-stratified analysis of every finished run"
python quality_analysis.py --runs "$RUNS" --quality "$QDIR/m1_quality.csv" --seed "$SEED" --n_boot 2000

step "13d. Masked manual-grading set (50 images)"
if [ ! -f "$QDIR/validation/grading_sheet.csv" ]; then
    python quality_validation.py make --quality "$QDIR/m1_quality.csv" --m0_dir "$QDIR/m0_crops" \
        --out "$QDIR/validation" --n 50
else
    echo "grading sheet exists; after filling it in, scoring:"
    python quality_validation.py score --out "$QDIR/validation" || echo "(sheet not filled in yet)"
fi

step "14. Two-stage cascade (binary -> three-class)"
if [ -f cascade.py ]; then
    python cascade.py --runs "$RUNS" --seed "$SEED" --n_boot 2000
else
    echo "cascade.py not found -- copy it from Drive (MyDrive/AMD/cascade.py) into $SRC"
fi
echo; echo "analysis finished $(date)"
