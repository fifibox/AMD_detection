"""quality_validation.py -- check AutoMorph M1 against masked manual grading on a random subset.

  make : pick N images (stratified by M1 class so all three levels are represented), copy their
         AutoMorph-M0 crops as q01.png, q02.png ... into a folder, and write
            grading_sheet.csv  (id, your_grade, notes)  <- fill in; contains NO model output
            key_DO_NOT_OPEN.csv (id -> file, M1 prediction)
  score: read the filled sheet and report agreement (linear-weighted kappa, confusion matrix,
         agreement on gradable vs ungradable).

Grade each image as good / usable / reject, using the EyeQ definitions:
  good   - no low-quality factors; all retinal pathology visible
  usable - some low-quality factors (blur, uneven illumination, artefacts) but the main structures
           and lesions are still identifiable for grading
  reject - poor quality; a reliable diagnosis cannot be made (e.g. key regions not visible)

    python quality_validation.py make  --quality quality/m1_quality.csv --m0_dir quality/m0 --out quality/validation --n 50
    python quality_validation.py score --out quality/validation
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

LEVELS = ["good", "usable", "reject"]


def make(a):
    q = pd.read_csv(a.quality)
    rng = np.random.default_rng(a.seed)
    per = {c: (q.m1_pred == c).sum() for c in LEVELS}
    take = {c: min(per[c], a.n // 3) for c in LEVELS}
    left = a.n - sum(take.values())
    for c in sorted(LEVELS, key=lambda c: -per[c]):  # top up from the largest classes
        extra = min(left, per[c] - take[c])
        take[c] += extra
        left -= extra
    pick = pd.concat([q[q.m1_pred == c].sample(take[c], random_state=int(rng.integers(1e9))) for c in LEVELS])
    pick = pick.sample(frac=1, random_state=a.seed).reset_index(drop=True)  # shuffle: no ordering cue
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    key = []
    for i, r in pick.iterrows():
        qid = f"q{i + 1:02d}"
        shutil.copy(Path(a.m0_dir) / (Path(r.file).stem + ".png"), out / "images" / f"{qid}.png")
        key.append({"id": qid, "file": r.file, "orig_label": r.orig_label, "m1_pred": r.m1_pred,
                    "gradable_automorph": r.gradable_automorph})
    pd.DataFrame({"id": [k["id"] for k in key], "your_grade": "", "notes": ""}).to_csv(out / "grading_sheet.csv", index=False)
    pd.DataFrame(key).to_csv(out / "key_DO_NOT_OPEN.csv", index=False)
    print(f"{len(key)} images -> {out/'images'}  (M1 class mix: {take})")
    print(f"Fill in {out/'grading_sheet.csv'} (your_grade = good / usable / reject), then run 'score'.")


def score(a):
    out = Path(a.out)
    s = pd.read_csv(out / "grading_sheet.csv", dtype=str).fillna("")
    k = pd.read_csv(out / "key_DO_NOT_OPEN.csv")
    d = s.merge(k, on="id")
    d["your_grade"] = d.your_grade.str.strip().str.lower()
    bad = d[~d.your_grade.isin(LEVELS)]
    if len(bad):
        print(f"[warn] {len(bad)} rows without a valid grade are skipped: {bad.id.tolist()}")
    d = d[d.your_grade.isin(LEVELS)]
    yi = d.your_grade.map(LEVELS.index)
    mi = d.m1_pred.map(LEVELS.index)
    kw = cohen_kappa_score(yi, mi, weights="linear")
    print(f"n = {len(d)} | exact agreement {np.mean(yi == mi):.1%} | linear-weighted kappa {kw:.2f}")
    print("Confusion (rows = your grade, cols = AutoMorph M1):")
    print(pd.DataFrame(confusion_matrix(yi, mi, labels=[0, 1, 2]), index=LEVELS, columns=LEVELS).to_string())
    # gradable: your 'good'/'usable' vs AutoMorph rule
    you_g = d.your_grade.isin(["good", "usable"])
    m1_g = d.gradable_automorph.astype(str).str.lower().eq("true")
    print(f"Gradable vs ungradable: agreement {np.mean(you_g == m1_g):.1%}, kappa {cohen_kappa_score(you_g, m1_g):.2f}")
    print("Note: the subset is stratified by M1 class, so these figures describe agreement across the "
          "quality range, not the prevalence of each level.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "score"])
    ap.add_argument("--quality")
    ap.add_argument("--m0_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make(a) if a.mode == "make" else score(a)
