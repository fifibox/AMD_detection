"""aggregate.py -- score finished runs and print the Chapter 4 tables.

Everything is recomputed from the per-image probabilities saved in
fold*_metrics.json (via evaluate.py), so older runs get kappa, PPV/NPV, the
referable view and patient-level bootstrap CIs without retraining.

    python aggregate.py --runs runs --seed 43
Writes, per task:
    runs/results_<task>_seed<seed>.csv      fold-level mean ± SD + pooled 95% CI
    runs/<task>/<config>/seed<seed>/eval.json   full detail for that run
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import summarise  # noqa: E402

CLASS_NAMES = {
    "binary": ["non_referable", "referable"],
    "three_class": ["non_referable", "intermediate", "late"],
    "four_class": ["no_amd", "early", "intermediate", "late"],
}
REF_COLS = ["balanced_accuracy", "sensitivity", "specificity", "kappa", "auroc"]


def fmt_ci(d, pct=True):
    if pct:
        return f"{100 * d['estimate']:.1f} [{100 * d['ci_low']:.1f}, {100 * d['ci_high']:.1f}]"
    return f"{d['estimate']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]"


def fmt_sd(d, pct=True):
    if pct:
        return f"{100 * d['mean']:.1f} ± {100 * d['sd']:.1f}"
    return f"{d['mean']:.3f} ± {d['sd']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()

    for task_dir in sorted(p for p in Path(a.runs).iterdir() if p.is_dir() and p.name in CLASS_NAMES):
        task, rows = task_dir.name, []
        for cfg in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            run = cfg / f"seed{a.seed}"
            files = sorted(run.glob("fold*_metrics.json"))
            if len(files) != 5:
                continue
            folds = [json.loads(f.read_text()) for f in files]
            s = summarise(folds, task, CLASS_NAMES[task], n_boot=a.n_boot, seed=a.seed)
            (run / "eval.json").write_text(json.dumps(s, indent=2))
            fl, fr, pb = s["fold_level"], s["fold_level_referable"], s["pooled_referable_bootstrap"]
            row = {"config": cfg.name,
                   f"{task}_bacc": fmt_sd(fl["balanced_accuracy"]),
                   f"{task}_kappa": f"{fl['kappa']['mean']:.3f} ± {fl['kappa']['sd']:.3f}"}
            if "kappa_quadratic" in fl:
                row[f"{task}_qwk"] = f"{fl['kappa_quadratic']['mean']:.3f} ± {fl['kappa_quadratic']['sd']:.3f}"
            for k in REF_COLS:
                pct = k not in ("kappa", "auroc")
                row[f"ref_{k}_folds"] = fmt_sd(fr[k], pct)
                row[f"ref_{k}_pooledCI"] = fmt_ci(pb[k], pct)
            row["ref_ppv_pooledCI"], row["ref_npv_pooledCI"] = fmt_ci(pb["ppv"]), fmt_ci(pb["npv"])
            rows.append(row)
        if not rows:
            continue
        out = Path(a.runs) / f"results_{task}_seed{a.seed}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\n=== {task} (seed {a.seed}) -> {out}")
        print("% referable, pooled with patient-bootstrap 95% CI: config & bal.acc & sens & spec & kappa & AUROC")
        for r in rows:
            cells = [r["config"].replace("_", r"\_")] + [r[f"ref_{k}_pooledCI"] for k in REF_COLS]
            print(" & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
