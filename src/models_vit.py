# models_vit.py
#
# ViT-Large/16 definition that matches the state_dict layout of the official
# RETFound MAE checkpoints (RETFound_mae_natureCFP, RETFound_mae_meh, etc).
# Adapted from the official RETFound repo (https://github.com/rmaphoh/RETFound),
# which itself builds on facebookresearch/mae's models_vit.py and timm's
# VisionTransformer. Only change from vanilla timm ViT: an optional
# "global_pool" mode that averages patch tokens instead of using the CLS
# token, matching how RETFound was fine-tuned in the original paper.

from functools import partial

import torch
import torch.nn as nn
import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling."""

    def __init__(self, global_pool=False, **kwargs):
        super().__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm, we use fc_norm instead

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global average pool, drop CLS token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]  # CLS token

        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head_drop(x) if hasattr(self, "head_drop") else x
        x = self.head(x)
        return x


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
