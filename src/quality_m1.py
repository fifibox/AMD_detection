"""quality_m1.py -- image-quality grading with AutoMorph M1 (EyeQ-trained, 8 x EfficientNet-B4 ensemble).

Faithful to the official AutoMorph code (github.com/rmaphoh/AutoMorph):
    M0  fundus_prep.process_without_gb : mask the fundus circle, crop its bounding box, pad to a square
    M1  BasicDataset_OUT.preprocess     : PIL resize to 512 x 512, standardise by mean/std of non-zero pixels
        8 EfficientNet-B4 models (seeds 28..42), softmax averaged -> P(good), P(usable), P(reject)
        AutoMorph gradability rule (merge_quality_assessment.py):
            gradable = pred == good  OR  (pred == usable AND P(reject) < 0.25)
The only change is that images are decoded with the thesis pipeline's data.load_rgb (DICOM, YBR->RGB),
and processed in memory instead of via PNG files (PNG is lossless, so the input is identical).

    python quality_m1.py --data_root /group/pmc078/fwei/DATASET --automorph /group/pmc078/fwei/AutoMorph --out quality/m1_quality.csv
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import list_dataset, load_rgb, subject_id_from_path  # noqa: E402

SEEDS = ["7_seed_28", "6_seed_30", "5_seed_32", "4_seed_34", "3_seed_36", "2_seed_38", "1_seed_40", "0_seed_42"]
CLASSES = ["good", "usable", "reject"]


def efficientnet_fl():
    """AutoMorph model.Efficientnet_fl, built without downloading ImageNet weights
    (every weight is overwritten by the M1 checkpoint)."""
    from efficientnet_pytorch import EfficientNet
    m = EfficientNet.from_name("efficientnet-b4")
    m._fc = nn.Sequential(nn.Linear(1792, 256), nn.ReLU(), nn.Dropout(p=0.5),
                          nn.Linear(256, 64), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(64, 3))
    return m


def m1_preprocess(pil_img, size=512):
    """Copy of AutoMorph M1 BasicDataset_OUT.preprocess (test mode)."""
    a = np.array(pil_img.resize((size, size))).astype(np.float64)
    a = (a - a[a > 0].mean()) / a[a > 0].std()
    return torch.from_numpy(a.transpose(2, 0, 1)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--automorph", required=True, help="clone of github.com/rmaphoh/AutoMorph")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--save_m0_dir", default=None, help="optionally save the M0 crops as PNG (for manual review)")
    ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.automorph) / "M0_Preprocess"))
    import fundus_prep as prep  # official AutoMorph M0 code

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wdir = Path(a.automorph) / "M1_Retinal_Image_quality_EyePACS/Retinal_quality/EyePACS_quality/efficientnet"
    models = []
    for s in SEEDS:
        m = efficientnet_fl()
        m.load_state_dict(torch.load(wdir / s / "best_loss_checkpoint.pth", map_location="cpu"))
        models.append(m.to(device).eval())
    print(f"[m1] loaded {len(models)} EfficientNet-B4 models on {device}")

    samples, _ = list_dataset(a.data_root, "four_class")
    if a.save_m0_dir:
        Path(a.save_m0_dir).mkdir(parents=True, exist_ok=True)

    rows, batch, meta = [], [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            p = torch.stack([torch.softmax(m(x), 1) for m in models]).cpu().numpy()  # (8, B, 3)
        mean, sd = p.mean(0), p.std(0)
        for (path, orig, radius), mu, s in zip(meta, mean, sd):
            pred = int(mu.argmax())
            rows.append({
                "path": path, "file": Path(path).name, "patient": subject_id_from_path(path),
                "orig_label": orig, "fundus_radius_px": radius,
                "p_good": mu[0], "p_usable": mu[1], "p_reject": mu[2],
                "sd_good": s[0], "sd_usable": s[1], "sd_reject": s[2],
                "m1_pred": CLASSES[pred],
                "gradable_automorph": bool(pred == 0 or (pred == 1 and mu[2] < 0.25)),
            })
        batch.clear()
        meta.clear()

    for i, (path, _, orig) in enumerate(samples):
        img = np.asarray(load_rgb(path)).copy()
        r_img, _, _, _, radii, _, _ = prep.process_without_gb(img, img.copy(), [], [], [])
        pil = Image.fromarray(r_img)
        if a.save_m0_dir:
            pil.save(Path(a.save_m0_dir) / (Path(path).stem + ".png"))
        batch.append(m1_preprocess(pil))
        meta.append((path, orig, int(radii[-1])))
        if len(batch) == a.batch_size:
            flush()
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(samples)}", flush=True)
    flush()

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    names = ["No AMD", "Early", "Intermediate", "Late"]
    print(f"\n-> {a.out}  ({len(df)} images)")
    print("\nM1 prediction by Beckman grade (rows) -- check that 'reject' is not just advanced disease:")
    ct = pd.crosstab(df.orig_label.map(dict(enumerate(names))), df.m1_pred).reindex(columns=CLASSES, fill_value=0)
    print(ct.to_string())
    print(f"\nGradable by the AutoMorph rule: {df.gradable_automorph.sum()} / {len(df)}")


if __name__ == "__main__":
    main()
