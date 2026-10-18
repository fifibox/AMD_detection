"""quality_analysis.py -- does image quality explain model performance? (hypothesis H4)

Uses the out-of-fold test predictions already saved by every finished run (no retraining) and the
AutoMorph M1 quality grades from quality_m1.py. For each run, on the referable decision:

  1. metrics in gradable vs ungradable images (AutoMorph rule), with the difference and a
     patient-level bootstrap 95% CI (patients resampled; each group rescored on the resample);
  2. the same by M1 class (good / usable / reject), descriptive only;
  3. confounding check: is 'ungradable' associated with AMD grade? (chi-square), and a logistic
     regression  correct ~ ungradable + C(Beckman grade)  with patient-clustered standard errors,
     i.e. the effect of quality on correctness *adjusted for disease severity*.

    python quality_analysis.py --runs runs --quality quality/m1_quality.csv --seed 43
Writes  <runs>/quality_by_group_seed<seed>.csv  and prints a summary.
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import referable_view  # noqa: E402

TASKS = ("binary", "three_class", "four_class")
GRADES = ["No AMD", "Early", "Intermediate", "Late"]


def load_run(d):
    files = sorted(Path(d).glob("fold*_metrics.json"))
    if len(files) != 5:
        return None
    rows = []
    for f in files:
        rows += json.loads(f.read_text())["test_predictions"]
    return rows


def ref_metrics(t, p):
    if len(t) == 0:
        return dict(n=0, bal_acc=np.nan, sens=np.nan, spec=np.nan, auroc=np.nan)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(t, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    auc = roc_auc_score(t, p) if len(np.unique(t)) == 2 else np.nan
    return dict(n=len(t), bal_acc=np.nanmean([sens, spec]), sens=sens, spec=spec, auroc=auc)


def analyse_run(task, cfg, rows, q, n_boot, seed):
    df = pd.DataFrame(rows)
    df["file"] = df.path.map(lambda s: Path(s).name)
    df = df.merge(q[["file", "gradable_automorph", "m1_pred", "p_reject"]], on="file", how="inner")
    y = df.y_true.values
    prob = np.stack(df.prob.values)
    t, p = referable_view(y, prob, task)
    df["ref_true"], df["ref_prob"] = t, p
    df["correct"] = ((p >= 0.5).astype(int) == t).astype(int)
    g = df.gradable_automorph.values.astype(bool)

    out = []
    base = {"task": task, "config": cfg}
    for name, mask in [("all", np.ones(len(df), bool)), ("gradable", g), ("ungradable", ~g)] + \
                      [(f"m1_{c}", (df.m1_pred == c).values) for c in ("good", "usable", "reject")]:
        out.append({**base, "group": name, **ref_metrics(t[mask], p[mask])})

    # bootstrap the gradable - ungradable difference (patients resampled)
    pats = df.patient.values
    uniq = np.unique(pats)
    idx_of = {u: np.flatnonzero(pats == u) for u in uniq}
    rng = np.random.default_rng(seed)
    diffs = {k: [] for k in ("bal_acc", "sens", "spec", "auroc")}
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        a = ref_metrics(t[idx][g[idx]], p[idx][g[idx]])
        b = ref_metrics(t[idx][~g[idx]], p[idx][~g[idx]])
        for k in diffs:
            diffs[k].append(a[k] - b[k])
    full_a, full_b = out[1], out[2]
    diff = {**base, "group": "gradable_minus_ungradable", "n": f"{full_a['n']} vs {full_b['n']}"}
    for k, v in diffs.items():
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        lo, hi = np.quantile(v, [0.025, 0.975]) if len(v) else (np.nan, np.nan)
        diff[k] = full_a[k] - full_b[k]
        diff[f"{k}_lo"], diff[f"{k}_hi"] = lo, hi
        diff[f"{k}_p"] = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())) if len(v) else np.nan
    out.append(diff)

    # logistic regression, adjusted for Beckman grade, patient-clustered SE
    adj = {}
    try:
        import statsmodels.formula.api as smf
        d = df.assign(ungradable=(~g).astype(int), grade=df.orig_label.astype(str))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.logit("correct ~ ungradable + C(grade)", data=d).fit(
                disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.patient)[0]})
        b, se = fit.params["ungradable"], fit.bse["ungradable"]
        adj = {"OR_correct_if_ungradable_adj_grade": np.exp(b),
               "OR_ci": f"[{np.exp(b - 1.96 * se):.2f}, {np.exp(b + 1.96 * se):.2f}]",
               "OR_p": fit.pvalues["ungradable"]}
    except Exception as e:  # e.g. perfect separation, statsmodels missing
        adj = {"OR_correct_if_ungradable_adj_grade": np.nan, "OR_ci": f"failed: {e}", "OR_p": np.nan}
    out[-1].update(adj)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--quality", required=True)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--configs", nargs="*", default=None,
                    help="task/config to analyse (default: every finished run)")
    a = ap.parse_args()

    q = pd.read_csv(a.quality, dtype={"patient": str})
    q["gradable_automorph"] = q.gradable_automorph.astype(str).str.lower().eq("true")

    # 0. confounding: quality vs disease grade
    grade = pd.Categorical(q.orig_label.map(dict(enumerate(GRADES))), categories=GRADES, ordered=True)
    ct = pd.crosstab(grade, q.gradable_automorph.map({True: "gradable", False: "ungradable"}))
    chi2, pchi, _, _ = chi2_contingency(ct.values)
    print("Quality (AutoMorph rule) by Beckman grade:\n" + ct.to_string())
    print(f"chi-square p = {pchi:.3g}  (small p = quality is associated with disease grade -> use the adjusted OR)\n")
    print("M1 class by Beckman grade:\n" + pd.crosstab(grade, q.m1_pred).reindex(
        columns=["good", "usable", "reject"], fill_value=0).to_string() + "\n")

    rows = []
    for task in TASKS:
        tdir = Path(a.runs) / task
        if not tdir.is_dir():
            continue
        for cfg in sorted(p.name for p in tdir.iterdir() if p.is_dir()):
            key = f"{task}/{cfg}"
            if a.configs and key not in a.configs:
                continue
            r = load_run(tdir / cfg / f"seed{a.seed}")
            if r is None:
                continue
            rows += analyse_run(task, cfg, r, q, a.n_boot, a.seed)
            print(f"  analysed {key}")
    res = pd.DataFrame(rows)
    out = Path(a.runs) / f"quality_by_group_seed{a.seed}.csv"
    res.to_csv(out, index=False)
    print(f"\n-> {out}")

    d = res[res.group == "gradable_minus_ungradable"]
    print("\nGradable minus ungradable (referable decision); bootstrap 95% CI, unadjusted;"
          " OR = odds of a correct referral decision when ungradable, adjusted for Beckman grade")
    for _, r in d.iterrows():
        print(f"{r.task + '/' + r.config:48s} n {r.n:>10s} | bal.acc {100 * r.bal_acc:+5.1f} "
              f"[{100 * r.bal_acc_lo:+.1f}, {100 * r.bal_acc_hi:+.1f}] p={r.bal_acc_p:.3f} "
              f"| AUROC {r.auroc:+.3f} [{r.auroc_lo:+.3f}, {r.auroc_hi:+.3f}] "
              f"| adj. OR {r.OR_correct_if_ungradable_adj_grade:.2f} {r.OR_ci} p={r.OR_p:.3f}")


if __name__ == "__main__":
    main()
