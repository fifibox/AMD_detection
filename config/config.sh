# config/config.sh -- paths and settings shared by every script (sourced, not run).
# Edit this file only; slurm/*.slurm and scripts/*.sh all read it.

BASE=/group/pmc078/fwei
CODE=$BASE/thesis_v2_hpc                                   # project root (this folder's parent)
SRC=$CODE/src                                              # all Python code
EXPERIMENTS=$CODE/config/experiments.txt
DATA=$BASE/DATASET                                         # contains 0/ 1/ 2/ 3/, SUBJ_<id>_ in filenames
CKPT=$BASE/5_fold_experiement/retfound_weights/RETFound_mae_natureCFP.pth
RUNS=$CODE/runs                                            # all training results (same layout as Drive/AMD/runs)
RESULTS=$CODE/results                                      # manifest + quality analysis outputs
AUTOMORPH=$BASE/AutoMorph                                  # cloned by setup_env.sh (compute nodes may have no internet)
VENV=$BASE/amd_venv                                        # Python environment
EXPECTED_N_IMAGES=337
SEED=43

# Mixed precision: "auto" = same rule as the Colab notebook (bf16 on A100/MI210, fp16 on V100).
# "none" = fp32 (what Kaya used before Colab).
AMP=auto

activate_env() {
    # module load ...            # add Kaya modules here if your venv needs them
    source "$VENV/bin/activate"
    export PYTHONUNBUFFERED=1
    # node-local cache of decoded DICOMs (identical pixels, just faster); one per node
    export AMD_CACHE_DIR=${TMPDIR:-/tmp}/amd_cache_${USER:-$(id -un)}
    mkdir -p "$AMD_CACHE_DIR"
}

resolve_amp() {
    if [ "$AMP" = "auto" ]; then
        AMP=$(python - <<'PY'
import torch
g = torch.cuda.get_device_name(0)
print("bf16" if torch.cuda.is_bf16_supported() and not any(x in g for x in ("T4", "V100", "P100")) else "fp16")
PY
)
    fi
    echo "AMP: $AMP"
}
