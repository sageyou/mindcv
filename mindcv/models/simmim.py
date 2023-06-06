import numpy as np
from typing import Optional
from functools import partial

import mindspore as ms
from mindspore.common.initializer import initializer, TruncatedNormal
from mindspore import nn, ops, Tensor, Parameter

from .beit import VisionTransformerEncoder, LayerNorm
from .swin_transformer import SwinTransformer
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "simmim_vit_16_224_pretrain",
]

class ViTForSimMIM(VisionTransformerEncoder):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 16,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        drop_path_rate: float = 0.1,
        init_values: Optional[float] = 0.1,
        use_abs_pos_emb: bool = False,
        use_rel_pos_bias: bool = False,
        use_shared_rel_pos_bias: bool = True,
        act_layer: nn.Cell = nn.GELU,
        norm_layer: nn.Cell = nn.LayerNorm,
        **kwargs
    ):
        super(ViTForSimMIM, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_abs_pos_emb=use_abs_pos_emb,
            use_rel_pos_bias=use_rel_pos_bias,
            use_shared_rel_pos_bias=use_shared_rel_pos_bias,
            **kwargs
        )
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_features = embed_dim
        self.hw = int(self.num_patches ** 0.5)
        self.mask_token = Parameter(initializer(TruncatedNormal(0.02), (1, 1, embed_dim)))
        self.norm = norm_layer((embed_dim, ))

        self._init_weights()
        self._fix_init_weights() 

    def forward_features(self, x, mask):
        x = self.patch_embed(x)
        bsz, seq_len, _ = x.shape

        mask_tokens = ops.broadcast_to(self.mask_token, (bsz, seq_len, -1))
        mask = ops.reshape(mask, (bsz, -1))
        w = ops.expand_dims(mask, axis=-1).astype(mask_tokens.dtype)
        x = x * (1 - w) + mask_tokens * w

        cls_tokens = ops.broadcast_to(self.cls_token, (bsz, -1, -1))
        x = ops.concat((cls_tokens, x), axis=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        if isinstance(self.rel_pos_bias, nn.CellList):
            for i, blk in enumerate(self.blocks):
                rel_pos_bias = self.rel_pos_bias[i]()
                x = blk(x, rel_pos_bias)
        else:
            rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
            for blk in self.blocks:
                x = blk(x, rel_pos_bias)

        return x

    def construct(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.norm(x)

        x = x[:, 1:]
        x = ops.transpose(x, (0, 2, 1))
        x = ops.reshape(x, (x.shape[0], x.shape[1], self.hw, self.hw))

        return x
        

class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.in_chans = self.patch_embed.in_chans
        self.patch_size = self.patch_embed.patch_size[0]
        self.mask_token = Parameter(initializer(TruncatedNormal(0.02), (1, 1, self.embed_dim)))

    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}

    def forward_features(self, x, mask):
        x = self.patch_embed(x)

        bsz, seq_len, _ = x.shape
        mask_tokens = ops.broadcast_to(self.mask_token, (bsz, seq_len, -1))
        mask = ops.reshape(mask, (bsz, -1))
        w = ops.expand_dims(mask, axis=-1).astype(mask_tokens.dtype)
        x = x * (1.0 - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        return x

    def construct(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.norm(x)

        x = ops.transpose(x, (0, 2, 1))
        bsz, chn, length = x.shape

        h = w = int(length ** 0.5)
        x = ops.reshape(x, (bsz, chn, h, w))

        return x


class PixelShuffle(nn.Cell):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.r = upscale_factor

    def construct(self, x):
        """
        x: [N, C * r ** 2, H, W]
        out: [N, C, H * r, W * r]
        """
        N, D, H, W = x.shape
        C = D // (self.r ** 2)
        assert C * self.r * self.r == D

        x = ops.reshape(x, (N, C, self.r, self.r, H, W))
        x = ops.transpose(x, (0, 1, 4, 2, 5, 3))
        x = ops.reshape(x, (N, C, H * self.r, W * self.r))
        return x


class SimMIM(nn.Cell):
    def __init__(
        self,
        encoder,
        encoder_stride
    ):
        super(SimMIM, self).__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.in_chans = encoder.in_chans
        self.patch_size = encoder.patch_size

        self.decoder = nn.Conv2d(
            in_channels=self.encoder.num_features,
            out_channels=self.encoder_stride ** 2 * 3,
            kernel_size=1, has_bias=True, pad_mode='pad'
        )

        self.pixel_shuffle = PixelShuffle(encoder_stride)

        self.l1_loss = nn.L1Loss(reduction='none')

    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

    def forward_loss(self, x, x_rec, mask):
        loss_recon = self.l1_loss(x, x_rec)

        mask = mask.astype(ms.float32)
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss

    def construct(self, x, mask):
        z = self.encoder(x, mask)
        z = self.decoder(z)
        x_rec = self.pixel_shuffle(z)

        mask = ops.expand_dims(
            ops.repeat_elements(
                ops.repeat_elements(mask, rep=self.patch_size, axis=1),
                rep=self.patch_size, axis=2
            ), axis=1
        )

        sim_loss = self.forward_loss(x, x_rec, mask)
        return sim_loss


@register_model
def simmim_vit_16_224_pretrain(pretrained=False, **kwargs):
    encoder = ViTForSimMIM(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        act_layer=partial(nn.GELU, approximate=False),
        norm_layer=partial(LayerNorm, epsilon=1e-6), **kwargs
    )
    model = SimMIM(encoder, encoder_stride=16)
    if pretrained:
        pass
    return model


@register_model
def simmim_swin_4_192_pretrain(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    kwargs.pop("mask_ratio")
    encoder = SwinTransformerForSimMIM(
        image_size=192, in_chans=in_channels, num_classes=num_classes, embed_dim=128,
        depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=6, **kwargs
    )
    model = SimMIM(encoder, encoder_stride=32)

    if pretrained:
        pass
    return model