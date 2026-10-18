"""models.py -- RETFound (ViT-L/16) and ImageNet-pretrained timm baselines.

Adaptation strategies for RETFound (thesis Table 3.5):
    lp       : backbone frozen; train fc_norm + linear head only
    partial  : additionally train the last k of the 24 transformer blocks
    full     : all parameters trainable
Baselines are always fully fine-tuned.
"""

import argparse

import timm
import torch
import torch.nn as nn

from models_vit import vit_large_patch16  # unchanged copy from 5_fold_experiement

try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_

HEAD_PREFIXES = ("head", "fc_norm", "fc.", "classifier")


def build_retfound(checkpoint, num_classes, strategy="full", k=0, drop_path_rate=0.2):
    model = vit_large_patch16(num_classes=num_classes, drop_path_rate=drop_path_rate, global_pool=True)
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([argparse.Namespace])
    ckpt = torch.load(checkpoint, map_location="cpu")
    ckpt = ckpt.get("model", ckpt)
    for key in ("head.weight", "head.bias"):
        ckpt.pop(key, None)  # new task head
    msg = model.load_state_dict(ckpt, strict=False)
    print(f"[models] RETFound load: missing={msg.missing_keys} unexpected={msg.unexpected_keys}")

    # Fresh head. Tiny init protects the backbone when it is being updated;
    # a normal-magnitude init is used when only the head learns (LP).
    trunc_normal_(model.head.weight, std=0.02 if strategy == "lp" else 2e-5)
    nn.init.zeros_(model.head.bias)

    if strategy in ("lp", "partial"):
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(("head", "fc_norm"))
        if strategy == "partial":
            if not 0 < k < len(model.blocks):
                raise ValueError(f"partial needs 0 < k < {len(model.blocks)}, got {k}")
            for blk in model.blocks[-k:]:
                for p in blk.parameters():
                    p.requires_grad = True
    elif strategy != "full":
        raise ValueError(strategy)
    return model


def build_timm(name, num_classes, drop_path_rate=0.2):
    """ImageNet-1k pretrained timm model; weights must already be cached on
    Kaya (compute nodes have no internet) -- see download_timm_weights.py."""
    return timm.create_model(name, pretrained=True, num_classes=num_classes,
                             drop_path_rate=drop_path_rate)


def build_model(arch, num_classes, checkpoint=None, strategy="full", k=0, drop_path_rate=0.2):
    if arch == "retfound":
        model = build_retfound(checkpoint, num_classes, strategy, k, drop_path_rate)
    else:
        if strategy != "full":
            raise ValueError("Baselines are only run with full fine-tuning.")
        model = build_timm(arch, num_classes, drop_path_rate)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[models] {arch} strategy={strategy} k={k}: total={total:,} trainable={trainable:,}")
    return model, total, trainable


def is_head_param(name):
    return name.startswith(HEAD_PREFIXES)
