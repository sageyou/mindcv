import numpy as np
from typing import Optional
from functools import partial

import mindspore as ms
from mindspore.common.initializer import initializer, Normal
from mindspore import nn, ops, Tensor, Parameter

from .beit import VisionTransformerEncoder
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
        use_shared_pos_bias: bool = True,
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
            norm_layer=norm_layer,
            use_abs_pos_emb=use_abs_pos_emb,
            use_rel_pos_bias=use_rel_pos_bias,
            use_shared_rel_pos_bias=use_shared_pos_bias,
            **kwargs
        )
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.hw = int(self.num_patches ** 0.5)
        self.mask_token = Parameter(initializer('truncatedNormal', (1, 1, embed_dim)))
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
            rel_pos_bias = [rpb() for rpb in self.rel_pos_bias]
        elif isinstance(self.rel_pos_bias, nn.Cell):
            rel_pos_bias = [self.rel_pos_bias() for _ in range(len(self.blocks))]
        else:
            rel_pos_bias = [None for _ in range(len(self.blocks))]

        for i, blk in enumerate(self.blocks):
            x = blk(x, rel_pos_bias[i])

        return x

    def construct(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.norm(x)

        x = x[:, 1:]
        x = ops.transpose(x, (0, 2, 1))
        x = ops.reshape(x, (x.shape[0], x.shape[1], self.hw, self.hw))

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
            in_channels=self.encoder.embed_dim,
            out_channels=self.encoder_stride ** 2 * 3,
            kernel_size=1, has_bias=True, pad_mode='pad'
        )

        self.pixel_shuffle = ops.DepthToSpace(encoder_stride)

        self.l1_loss = nn.L1Loss(reduction='none')

    def forward_loss(self, x, x_rec, mask):
        loss_recon = self.l1_loss(x, x_rec)

        mask = mask.astype(loss_recon.dtype)
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
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs
    )
    model = SimMIM(encoder, encoder_stride=16)
    if pretrained:
        pass
    return model