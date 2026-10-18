"""train_cv.py -- one script for every Chapter 4 experiment.

Patient-grouped, stratified K-fold CV with an inner patient-grouped
validation split for checkpoint selection. Same splits, same augmentation,
same optimiser/schedule and same checkpoint rule for RETFound and baselines.

Examples
    # RETFound full fine-tuning, binary task, seed 42
    python train_cv.py --data_root /group/pmc078/fwei/DATASET --task binary \
        --arch retfound --strategy full --checkpoint retfound_weights/RETFound_mae_natureCFP.pth \
        --seed 42 --output_dir runs/binary/retfound_full/seed42

    # ResNet-18 baseline
    python train_cv.py --data_root ... --task binary --arch resnet18 --seed 42 \
        --output_dir runs/binary/resnet18/seed42

Outputs (per run directory)
    config.json            every argument + library versions + param counts
    folds.csv              image, patient, original label, task label, fold, role
    fold{k}_metrics.json   metrics + per-image probabilities for the test fold
    fold{k}_history.csv    per-epoch train/val curves
    summary.json           mean +/- SD (ddof=1) across folds
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find data.py/models.py from any cwd
import copy
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import sklearn
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from data import (FundusDataset, build_transforms, groups_of, inner_train_val, list_dataset,
                  outer_folds, subject_id_from_path)
from evaluate import REF_KEYS, TASK_KEYS, fold_metrics, summarise
from models import build_model, is_head_param


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--task", choices=["binary", "three_class", "four_class"], default="binary")
    p.add_argument("--arch", default="retfound", help="'retfound' or a timm model name")
    p.add_argument("--strategy", choices=["lp", "partial", "full"], default="full")
    p.add_argument("--k", type=int, default=0, help="blocks unfrozen for --strategy partial")
    p.add_argument("--checkpoint", help="RETFound .pth (required for --arch retfound)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_epochs", type=int, default=0,
                   help="never early-stop before this epoch (checkpoint selection still uses all epochs)")
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=0.0,
                   help="cosine floor for the backbone lr (RETFound code: 1e-6); other groups scale with it")
    p.add_argument("--aug", choices=["default", "retfound"], default="default",
                   help="'retfound' = official RETFound augmentation (RRC 0.08-1, RandAugment, random erasing)")
    p.add_argument("--select_metric", choices=["balanced_accuracy", "auroc"], default="balanced_accuracy",
                   help="validation metric for checkpoint selection (RETFound code: AUROC)")
    p.add_argument("--backbone_lr", type=float, default=1e-4)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--layer_decay", type=float, default=None,
                   help="RETFound only: official RETFound/MAE fine-tuning recipe -- layer-wise lr decay; "
                        "block i gets backbone_lr * decay^(25-i-1), head/fc_norm get backbone_lr "
                        "(--head_lr ignored), no weight decay on biases/norms/pos_embed/cls_token")
    p.add_argument("--drop_path_rate", type=float, default=0.2)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none",
                   help="mixed precision (bf16 on A100/MI210, fp16 on V100); 'none' = fp32 as on Kaya")
    p.add_argument("--resume", action="store_true",
                   help="skip folds whose fold{k}_metrics.json already exists (resubmitted jobs)")
    p.add_argument("--fold_idx", type=int, default=-1, help="run one fold only (debug)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.arch == "retfound" and not a.checkpoint:
        p.error("--checkpoint is required for --arch retfound")
    return a


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# --------------------------------------------------------------------------
def _vit_layer_id(name, num_layers):
    if name.startswith(("cls_token", "pos_embed", "patch_embed")):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return num_layers + 1  # fc_norm, head


def build_optimizer_lrd(model, args):
    """Layer-wise lr decay param groups, as in RETFound's util/lr_decay.py (from MAE)."""
    n = len(model.blocks)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_wd = p.ndim == 1 or name in ("pos_embed", "cls_token")
        lid = _vit_layer_id(name, n)
        key = (lid, no_wd)
        if key not in groups:
            groups[key] = {"params": [], "lr": args.backbone_lr * args.layer_decay ** (n + 1 - lid),
                           "weight_decay": 0.0 if no_wd else args.weight_decay}
        groups[key]["params"].append(p)
    g = sorted(groups.values(), key=lambda d: d["lr"])
    print(f"[optim] layer decay {args.layer_decay}: {len(g)} groups, lr {g[0]['lr']:.2e} .. {g[-1]['lr']:.2e}")
    return torch.optim.AdamW(g)


def build_optimizer(model, args):
    if args.layer_decay is not None:
        if args.arch != "retfound":
            raise ValueError("--layer_decay is only implemented for RETFound (ViT)")
        return build_optimizer_lrd(model, args)
    backbone = [p for n, p in model.named_parameters() if p.requires_grad and not is_head_param(n)]
    head = [p for n, p in model.named_parameters() if p.requires_grad and is_head_param(n)]
    groups = [{"params": head, "lr": args.head_lr}]
    if backbone:
        groups.append({"params": backbone, "lr": args.backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def val_auroc(y, prob):
    """Validation AUROC: binary on p[:,1]; multiclass macro one-vs-rest (as RETFound). NaN if undefined."""
    y = np.asarray(y)
    if prob.shape[1] == 2:
        return float(roc_auc_score(y, prob[:, 1])) if len(np.unique(y)) == 2 else float("nan")
    # macro one-vs-rest over the classes that have both positives and negatives in this
    # (small, ~25-30 image) validation split, so a missing class does not void the metric
    aucs = [roc_auc_score(y == c, prob[:, c]) for c in range(prob.shape[1]) if 0 < (y == c).sum() < len(y)]
    return float(np.mean(aucs)) if aucs else float("nan")


def build_scheduler(opt, epochs, warmup, steps_per_epoch, min_frac=0.0):
    """Per-step linear warm-up then cosine decay to min_frac * peak (as in MAE/RETFound util/lr_sched.py)."""
    total, warm = epochs * steps_per_epoch, warmup * steps_per_epoch

    def f(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        return min_frac + (1 - min_frac) * 0.5 * (1 + np.cos(np.pi * (step - warm) / max(1, total - warm)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def predict(model, loader, device, amp="none"):
    model.eval()
    probs, labels, loss_sum = [], [], 0.0
    for x, y in loader:
        with autocast(device, amp):
            logits = model(x.to(device, non_blocking=True)).float()
        loss_sum += F.cross_entropy(logits, y.to(device), reduction="sum").item()
        probs.append(F.softmax(logits.float(), 1).cpu().numpy())
        labels.append(y.numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)
    return labels, probs, loss_sum / len(labels)


AMP_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}


def autocast(device, amp):
    return torch.autocast(device_type=device.type, dtype=AMP_DTYPE.get(amp, torch.float32),
                          enabled=amp != "none" and device.type == "cuda")


def train_one_epoch(model, loader, criterion, opt, sched, device, amp="none", scaler=None):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, amp):
            loss = criterion(model(x).float(), y)
        opt.zero_grad()
        if scaler is not None:  # fp16 only
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        sched.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


# --------------------------------------------------------------------------
def run_fold(fold, train_samples, test_samples, class_names, args, out):
    device = torch.device(args.device)
    set_seed(args.seed + fold)
    tr, va = inner_train_val(train_samples, args.val_frac, seed=args.seed + fold)
    inner_train = [train_samples[i] for i in tr]
    inner_val = [train_samples[i] for i in va]

    g_tr, g_va, g_te = (set(groups_of(s)) for s in (inner_train, inner_val, test_samples))
    assert not (g_tr & g_va) and not (g_tr & g_te) and not (g_va & g_te), "patient leakage"
    print(f"\n=== fold {fold}: images train/val/test = {len(inner_train)}/{len(inner_val)}/"
          f"{len(test_samples)} | patients = {len(g_tr)}/{len(g_va)}/{len(g_te)}")

    gen = torch.Generator().manual_seed(args.seed + fold)
    mk = lambda s, train: DataLoader(
        FundusDataset(s, build_transforms(args.input_size, train, args.aug)), batch_size=args.batch_size,
        shuffle=train, drop_last=train and len(s) > args.batch_size, num_workers=args.num_workers,
        pin_memory=True, worker_init_fn=seed_worker, generator=gen if train else None)
    train_loader, val_loader, test_loader = mk(inner_train, True), mk(inner_val, False), mk(test_samples, False)

    model, n_total, n_trainable = build_model(args.arch, len(class_names), args.checkpoint,
                                              args.strategy, args.k, args.drop_path_rate)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)  # unweighted
    opt = build_optimizer(model, args)
    sched = build_scheduler(opt, args.epochs, args.warmup_epochs, len(train_loader),
                            min_frac=args.min_lr / args.backbone_lr)
    scaler = None
    if args.amp == "fp16" and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda") if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler()

    best, best_state, best_epoch, wait, history = -np.inf, None, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, criterion, opt, sched, device, args.amp, scaler)
        yv, pv, val_loss = predict(model, val_loader, device, args.amp)
        val_bacc = balanced_accuracy_score(yv, pv.argmax(1))
        val_auc = val_auroc(yv, pv)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss,
                        "val_balanced_accuracy": val_bacc, "val_auroc": val_auc})
        print(f"  ep {epoch:03d} train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} "
              f"val_bacc {val_bacc:.3f} val_auc {val_auc:.3f} | {time.time() - t0:.0f}s")
        score = val_auc if args.select_metric == "auroc" else val_bacc
        if np.isnan(score):
            score = val_bacc
        # checkpoint rule: best validation balanced accuracy (default) or AUROC (RETFound)
        if score > best + 1e-4 or best_state is None:
            best, best_state, best_epoch, wait = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= args.patience and epoch >= args.min_epochs:
                print(f"  early stop at epoch {epoch}; best epoch {best_epoch}")
                break
    model.load_state_dict(best_state)

    y, prob, _ = predict(model, test_loader, device, args.amp)
    m = fold_metrics(y, prob, args.task, class_names)
    m.update({"fold": fold, "best_epoch": best_epoch, "select_metric": args.select_metric,
              "best_val_score": float(best),
              "n_params_total": n_total, "n_params_trainable": n_trainable,
              "n_images": {"train": len(inner_train), "val": len(inner_val), "test": len(test_samples)},
              "n_patients": {"train": len(g_tr), "val": len(g_va), "test": len(g_te)},
              "test_predictions": [{"path": s[0], "patient": subject_id_from_path(s[0]),
                                    "orig_label": s[2], "y_true": int(t), "prob": pr.tolist()}
                                   for s, t, pr in zip(test_samples, y, prob)]})
    (out / f"fold{fold}_metrics.json").write_text(json.dumps(m, indent=2))
    with open(out / f"fold{fold}_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    r = m["referable"]
    print(f"  TEST fold {fold}: {args.task} bacc {m['balanced_accuracy']:.3f} kappa {m['kappa']:.3f} | "
          f"referable sens {r['sensitivity']:.3f} spec {r['specificity']:.3f} "
          f"kappa {r['kappa']:.3f} auroc {r['auroc']:.3f}")
    del model, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return m, inner_val


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples, class_names = list_dataset(args.data_root, args.task)
    folds = outer_folds(samples, args.n_splits, args.seed)

    cfg = vars(args) | {"class_names": class_names, "n_images": len(samples),
                        "n_patients": int(len(set(groups_of(samples)))),
                        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                                     "timm": timm.__version__, "sklearn": sklearn.__version__,
                                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    rows, results = [], []
    for fold, (tr_idx, te_idx) in enumerate(folds):
        if args.fold_idx >= 0 and fold != args.fold_idx:
            continue
        train_s, test_s = [samples[i] for i in tr_idx], [samples[i] for i in te_idx]
        done = out / f"fold{fold}_metrics.json"
        if args.resume and done.exists():
            # Reuse the finished fold; the inner val split is deterministic, so
            # folds.csv is rebuilt exactly as if the fold had just been trained.
            m = json.loads(done.read_text())
            _, va = inner_train_val(train_s, args.val_frac, seed=args.seed + fold)
            val_s = [train_s[i] for i in va]
            print(f"\n=== fold {fold}: already done ({done.name}), skipping")
        else:
            m, val_s = run_fold(fold, train_s, test_s, class_names, args, out)
        results.append(m)
        val_paths = {s[0] for s in val_s}
        for s in test_s:
            rows.append((s[0], subject_id_from_path(s[0]), s[2], s[1], fold, "test"))
        for s in train_s:
            rows.append((s[0], subject_id_from_path(s[0]), s[2], s[1], fold,
                         "val" if s[0] in val_paths else "train"))
    with open(out / "folds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "patient", "orig_label", "task_label", "fold", "role"])
        w.writerows(rows)

    if args.fold_idx >= 0:
        return  # single-fold debug run: no summary
    summary = summarise(results, args.task, class_names)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSUMMARY: fold-level mean ± SD (ddof=1)")
    for k, v in summary["fold_level"].items():
        print(f"  {args.task:12s} {k:20s} {v['mean']:.4f} ± {v['sd']:.4f}")
    for k, v in summary["fold_level_referable"].items():
        print(f"  {'referable':12s} {k:20s} {v['mean']:.4f} ± {v['sd']:.4f}")
    print("Pooled referable, patient-level bootstrap 95% CI:")
    for k in REF_KEYS:
        v = summary["pooled_referable_bootstrap"][k]
        print(f"  {k:20s} {v['estimate']:.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]")

if __name__ == "__main__":
    main()
