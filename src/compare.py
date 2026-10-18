"""compare.py -- paired, patient-level bootstrap comparison of finished runs.

Every model in a task was evaluated on the same images: each image has exactly
one out-of-fold prediction per run. Pairing the two runs' predictions image by
image, we resample PATIENTS (with replacement, the same draw for both models),
recompute each metric for both, and take the difference. Patient-level
resampling respects the correlation between a patient's eyes and repeat images.

Reported per comparison (reference minus other):
    difference (point estimate on all images), 95% percentile CI,
    two-sided bootstrap p (2 * smaller tail of the bootstrap differences around 0),
    Holm-adjusted p within each task x metric family.
A difference is "significant" when its 95% CI excludes 0.

Metrics are on the referable decision (P(referable) >= 0.5; AUROC from
P(referable)), so binary, three-class and four-class ("4 -> 2") runs can be
compared with each other on the same footing. For pairs from the same
multiclass task the task-level balanced accuracy is compared too.

    python compare.py --runs runs --seed 43
    python compare.py --runs runs --seed 43 --reference retfound_full
    python compare.py --runs runs --pairs binary/retfound_full:binary/resnet18_full_lr1e-3

Writes runs/paired_seed<seed>.csv
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import referable_view  # noqa: E402

TASKS = ("binary", "three_class", "four_class")
REF_METRICS = ("ref_balanced_accuracy", "ref_auroc", "ref_sensitivity", "ref_specificity")


# --------------------------------------------------------------------------
def load_run(runs, task, config, seed):
    """Out-of-fold predictions of one run, keyed by image path."""
    d = Path(runs) / task / config / f"seed{seed}"
    files = sorted(d.glob("fold*_metrics.json"))
    if len(files) != 5:
        return None
    preds = {}
    for f in files:
        for r in json.loads(f.read_text())["test_predictions"]:
            preds[r["path"]] = (int(r["y_true"]), np.asarray(r["prob"], float), r["patient"], int(r["orig_label"]))
    return {"task": task, "config": config, "preds": preds}


def metric_fns(task_a, task_b):
    """name -> f(run_arrays, idx) for the metrics valid for this pair."""
    def ref(run, idx, which):
        t, p = run["ref_true"][idx], run["ref_prob"][idx]
        if which == "ref_auroc":
            return roc_auc_score(t, p) if len(np.unique(t)) == 2 else np.nan
        pred = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(t, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        return {"ref_sensitivity": sens, "ref_specificity": spec,
                "ref_balanced_accuracy": np.nanmean([sens, spec])}[which]

    fns = {m: (lambda run, idx, m=m: ref(run, idx, m)) for m in REF_METRICS}
    if task_a == task_b and task_a != "binary":
        fns["task_balanced_accuracy"] = lambda run, idx: balanced_accuracy_score(
            run["y"][idx], run["prob"][idx].argmax(1))
    return fns


def arrays(run, paths):
    y = np.array([run["preds"][p][0] for p in paths])
    prob = np.stack([run["preds"][p][1] for p in paths])
    ref_true, ref_prob = referable_view(y, prob, run["task"])
    return {"y": y, "prob": prob, "ref_true": ref_true, "ref_prob": ref_prob}


def paired_bootstrap(a, b, n_boot=2000, seed=0):
    """Reference a vs other b on their common images."""
    common = sorted(set(a["preds"]) & set(b["preds"]))
    if len(common) < max(len(a["preds"]), len(b["preds"])):
        print(f"  [warn] {a['config']} vs {b['config']}: only {len(common)} common images")
    patients = np.array([a["preds"][p][2] for p in common])
    # sanity: both runs must carry the same Beckman grade for each image
    oa = np.array([a["preds"][p][3] for p in common])
    ob = np.array([b["preds"][p][3] for p in common])
    assert (oa == ob).all(), "label mismatch between runs -- were they trained on the same dataset?"

    A, B = arrays(a, common), arrays(b, common)
    fns = metric_fns(a["task"], b["task"])
    full = np.arange(len(common))
    uniq = np.unique(patients)
    idx_of = {p: np.flatnonzero(patients == p) for p in uniq}
    rng = np.random.default_rng(seed)
    draws = {m: [] for m in fns}
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
        for m, f in fns.items():
            draws[m].append(f(A, idx) - f(B, idx))
    out = []
    for m, f in fns.items():
        ea, eb = f(A, full), f(B, full)
        d = np.asarray(draws[m], float)
        d = d[~np.isnan(d)]
        lo, hi = np.quantile(d, [0.025, 0.975])
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        out.append({"metric": m, "reference": f"{a['task']}/{a['config']}", "other": f"{b['task']}/{b['config']}",
                    "ref_value": ea, "other_value": eb, "diff": ea - eb, "ci_low": lo, "ci_high": hi,
                    "p_boot": p, "n_images": len(common), "n_patients": len(uniq), "n_boot": len(d)})
    return out


def holm(rows):
    """Holm-adjust p within each (reference task, metric) family of comparisons."""
    fams = {}
    for r in rows:
        fams.setdefault((r["reference"].split("/")[0], r["family"], r["metric"]), []).append(r)
    for fam in fams.values():
        fam.sort(key=lambda r: r["p_boot"])
        m, running = len(fam), 0.0
        for i, r in enumerate(fam):
            running = max(running, min(1.0, (m - i) * r["p_boot"]))
            r["p_holm"] = running


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--reference", default="retfound_full", help="config compared against all others in its task")
    ap.add_argument("--pairs", nargs="*", default=[], help="extra 'task/config:task/config' pairs")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()

    runs = {}
    for task in TASKS:
        tdir = Path(a.runs) / task
        if not tdir.is_dir():
            continue
        for cfg in sorted(p.name for p in tdir.iterdir() if p.is_dir()):
            r = load_run(a.runs, task, cfg, a.seed)
            if r is not None:
                runs[f"{task}/{cfg}"] = r
    print(f"finished runs found: {len(runs)}")

    jobs = []  # (family, ref_key, other_key)
    for task in TASKS:  # 1. reference vs every other config in the same task
        ref = f"{task}/{a.reference}"
        if ref in runs:
            jobs += [("within_task", ref, k) for k in runs if k.startswith(task + "/") and k != ref]
    rb = f"binary/{a.reference}"  # 2. task formulations of the reference model, on the referral decision
    jobs += [("across_task", rb, f"{t}/{a.reference}") for t in ("three_class", "four_class")
             if rb in runs and f"{t}/{a.reference}" in runs]
    for p in a.pairs:  # 3. user-specified
        x, y = p.split(":")
        if x in runs and y in runs:
            jobs.append(("user", x, y))
        else:
            print(f"  [skip] {p}: run not found or unfinished")
    if not jobs:
        sys.exit(f"Nothing to compare: '{a.reference}' runs are not finished yet.")

    rows = []
    for fam, x, y in jobs:
        print(f"  {x}  vs  {y}")
        for r in paired_bootstrap(runs[x], runs[y], a.n_boot, a.seed):
            r["family"] = fam
            rows.append(r)
    holm(rows)

    out = Path(a.runs) / f"paired_seed{a.seed}.csv"
    keys = ["family", "reference", "other", "metric", "ref_value", "other_value", "diff",
            "ci_low", "ci_high", "p_boot", "p_holm", "n_images", "n_patients", "n_boot"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r[k] for k in keys} for r in rows)
    print(f"\n-> {out}")

    pct = lambda m: m != "ref_auroc"
    for m in list(REF_METRICS) + ["task_balanced_accuracy"]:
        sel = [r for r in rows if r["metric"] == m]
        if not sel:
            continue
        print(f"\n% {m}: reference - other, difference [95% CI], Holm p   (* = CI excludes 0)")
        for r in sel:
            s = 100 if pct(m) else 1
            fmt = "{:+.1f}" if pct(m) else "{:+.3f}"
            star = "*" if r["ci_low"] > 0 or r["ci_high"] < 0 else " "
            pv = f"<{1 / r['n_boot']:.4f}" if r["p_holm"] == 0 else f"={r['p_holm']:.3f}"
            print(f"{r['reference']:32s} vs {r['other']:42s} {fmt.format(s * r['diff'])} "
                  f"[{fmt.format(s * r['ci_low'])}, {fmt.format(s * r['ci_high'])}]  p{pv} {star}")


if __name__ == "__main__":
    main()
