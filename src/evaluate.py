"""evaluate.py -- all evaluation metrics, computed from saved softmax outputs.

Pure numpy/sklearn (no torch), so it is used both inside train_cv.py and by
aggregate.py to (re)score finished runs from their fold*_metrics.json files.

Referable view (thesis Section 3.7)
    Every task is also scored as the binary screening decision
    "referable = intermediate or late AMD":
        binary      P(referable)            = p[:, 1]
        three_class P(intermediate)+P(late) = p[:, 1] + p[:, 2]
        four_class  P(intermediate)+P(late) = p[:, 2] + p[:, 3]   ("4 -> 2" fusion)
    and predicted referable when that summed probability >= 0.5.

Patient-level bootstrap
    Out-of-fold predictions from the 5 test folds are pooled (each image is
    predicted exactly once, by a model that never saw its patient), then
    patients -- not images -- are resampled with replacement, so the CI
    respects the correlation between a patient's images and eyes.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, precision_recall_fscore_support, roc_auc_score)

REFERABLE_CLASSES = {"binary": [1], "three_class": [1, 2], "four_class": [2, 3]}


def _safe_div(a, b):
    return float(a) / float(b) if b else float("nan")


def referable_view(y, prob, task):
    """Return (true referable 0/1, P(referable))."""
    cls = REFERABLE_CLASSES[task]
    return np.isin(y, cls).astype(int), prob[:, cls].sum(1)


def referable_metrics(ref_true, ref_prob, threshold=0.5):
    pred = (ref_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(ref_true, pred, labels=[0, 1]).ravel()
    two = len(set(ref_true.tolist())) == 2
    return {
        "accuracy": accuracy_score(ref_true, pred),
        "balanced_accuracy": balanced_accuracy_score(ref_true, pred) if two else float("nan"),
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "ppv": _safe_div(tp, tp + fp),
        "npv": _safe_div(tn, tn + fn),
        "kappa": cohen_kappa_score(ref_true, pred) if two else float("nan"),
        "auroc": roc_auc_score(ref_true, ref_prob) if two else float("nan"),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def task_metrics(y, prob, class_names):
    """Metrics on the task's own classes (argmax prediction)."""
    k = len(class_names)
    pred = prob.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=range(k), average="macro", zero_division=0)
    pc = precision_recall_fscore_support(y, pred, labels=range(k), average=None, zero_division=0)
    m = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision_macro": p, "recall_macro": r, "f1_macro": f1,
        "kappa": cohen_kappa_score(y, pred),
        "per_class": {c: {"precision": float(a), "recall": float(b), "f1": float(d), "n": int(e)}
                      for c, a, b, d, e in zip(class_names, *pc)},
        "confusion_matrix": confusion_matrix(y, pred, labels=range(k)).tolist(),
    }
    if k > 2:
        # Classes are ordinal (severity), so also report quadratic-weighted kappa.
        m["kappa_quadratic"] = cohen_kappa_score(y, pred, weights="quadratic")
        try:
            m["auroc_macro_ovr"] = roc_auc_score(y, prob, multi_class="ovr", average="macro",
                                                 labels=list(range(k)))
        except ValueError:
            m["auroc_macro_ovr"] = float("nan")
    return m


def fold_metrics(y, prob, task, class_names):
    y, prob = np.asarray(y), np.asarray(prob)
    m = task_metrics(y, prob, class_names)
    m["referable"] = referable_metrics(*referable_view(y, prob, task))
    return _to_builtin(m)


def patient_bootstrap(y, prob, patients, task, n_boot=2000, seed=0, alpha=0.05):
    """Pooled point estimates + percentile CIs from a patient-level bootstrap."""
    y, prob, patients = np.asarray(y), np.asarray(prob), np.asarray(patients)
    ref_true, ref_prob = referable_view(y, prob, task)
    keys = ["accuracy", "balanced_accuracy", "sensitivity", "specificity",
            "ppv", "npv", "kappa", "auroc"]
    point = referable_metrics(ref_true, ref_prob)
    uniq = np.unique(patients)
    idx_of = {p: np.flatnonzero(patients == p) for p in uniq}
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
        b = referable_metrics(ref_true[idx], ref_prob[idx])
        for k in keys:
            draws[k].append(b[k])
    out = {"n_images": int(len(y)), "n_patients": int(len(uniq)), "n_boot": n_boot,
           "referable_prevalence": float(ref_true.mean())}
    for k in keys:
        v = np.asarray(draws[k], dtype=float)
        v = v[~np.isnan(v)]
        lo, hi = (np.quantile(v, [alpha / 2, 1 - alpha / 2]) if len(v) else (np.nan, np.nan))
        out[k] = {"estimate": float(point[k]), "ci_low": float(lo), "ci_high": float(hi)}
    return out


def _to_builtin(x):
    if isinstance(x, dict):
        return {k: _to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_builtin(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


TASK_KEYS = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
             "kappa", "kappa_quadratic", "auroc_macro_ovr"]
REF_KEYS = ["accuracy", "balanced_accuracy", "sensitivity", "specificity",
            "ppv", "npv", "kappa", "auroc"]


def summarise(folds, task, class_names, n_boot=2000, seed=0):
    """Summarise one 5-fold run from its fold*_metrics.json dicts.

    Metrics are recomputed from the saved per-image probabilities, so runs
    trained before evaluate.py existed are scored identically to new ones.
    Returns fold-level mean / SD (ddof=1) and the pooled patient bootstrap.
    """
    per_fold, ys, ps, pats = [], [], [], []
    for f in folds:
        tp = f["test_predictions"]
        y = np.array([r["y_true"] for r in tp])
        p = np.array([r["prob"] for r in tp], dtype=float)
        per_fold.append(fold_metrics(y, p, task, class_names))
        ys.append(y), ps.append(p), pats.extend(r["patient"] for r in tp)

    def agg(vals):
        v = np.asarray(vals, dtype=float)
        return {"mean": float(np.nanmean(v)),
                "sd": float(np.nanstd(v, ddof=1)) if np.sum(~np.isnan(v)) > 1 else 0.0,
                "per_fold": [float(x) for x in v]}

    out = {"task": task, "n_folds": len(folds), "fold_level": {}, "fold_level_referable": {}}
    for k in TASK_KEYS:
        if k in per_fold[0]:
            out["fold_level"][k] = agg([m[k] for m in per_fold])
    for k in REF_KEYS:
        out["fold_level_referable"][k] = agg([m["referable"][k] for m in per_fold])
    out["pooled_referable_bootstrap"] = patient_bootstrap(
        np.concatenate(ys), np.concatenate(ps), pats, task, n_boot=n_boot, seed=seed)
    return out
