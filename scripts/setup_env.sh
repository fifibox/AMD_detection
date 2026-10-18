#!/bin/bash
# scripts/setup_env.sh -- run ONCE on the Kaya LOGIN node (needs internet):  bash scripts/setup_env.sh
# Installs packages, gets the RETFound weights and the AutoMorph M1 code/weights.
set -euo pipefail
cd "$(dirname "$0")/.."
source config/config.sh
mkdir -p logs "$RUNS" "$RESULTS"

# 1. Python environment (reuses amd_venv if it already exists)
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --upgrade pip
    # V100 (CUDA):  default PyTorch wheel
    pip install torch torchvision
    # MI210 (ROCm): use instead, e.g.
    # pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
else
    source "$VENV/bin/activate"
fi
pip install -q timm pydicom huggingface_hub scikit-learn scipy pandas matplotlib \
    efficientnet_pytorch opencv-python-headless statsmodels

# 2. RETFound weights (gated on HuggingFace: export HF_TOKEN=hf_... first if not already downloaded)
if [ ! -f "$CKPT" ]; then
    mkdir -p "$(dirname "$CKPT")"
    python - <<PY
import os, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download("YukunZhou/RETFound_mae_natureCFP", "RETFound_mae_natureCFP.pth", token=os.environ.get("HF_TOKEN"))
shutil.copy(p, "$CKPT")
PY
fi
ls -lh "$CKPT"

# 3. AutoMorph M0 + M1 (code + weights, ~600 MB)
if [ ! -d "$AUTOMORPH/M1_Retinal_Image_quality_EyePACS" ]; then
    git clone --depth 1 --filter=blob:none --sparse https://github.com/rmaphoh/AutoMorph.git "$AUTOMORPH"
    (cd "$AUTOMORPH" && git sparse-checkout set M0_Preprocess M1_Retinal_Image_quality_EyePACS)
fi
ls "$AUTOMORPH/M1_Retinal_Image_quality_EyePACS/Retinal_quality/EyePACS_quality/efficientnet"

# 4. Dataset check (CPU only, fine on the login node)
python "$SRC/check_data.py" --data_root "$DATA" --expected "$EXPECTED_N_IMAGES"
echo "setup OK"
