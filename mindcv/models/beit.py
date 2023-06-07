from collections import OrderedDict
from typing import Optional
from functools import partial

from mindspore.common.initializer import initializer, TruncatedNormal
from mindspore import nn, ops, Parameter

from .vit_encoder import LayerNorm, VisionTransformerEncoder
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "dall_e",
    "beit_b_16_224_pretrain",
    "beit_l_16_224_pretrain",
    "beit_b_16_224_finetune",
    "beit_b_16_384_finetune",
    "beit_l_16_224_finetune",
    "beit_l_16_384_finetune",
    "beit_l_16_512_finetune"
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 8192,
        "input_size": (3, 224, 224),
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }
    

default_cfgs = {
    "beit_b_16_224_finetune": _cfg(url="", use_rel_pos_bias=True),
    "beit_b_16_384_finetune": _cfg(url="", input_size=(3, 384, 384), use_rel_pos_bias=True),
    "beit_l_16_224_finetune": _cfg(url="", use_rel_pos_bias=True),
    "beit_l_16_384_finetune": _cfg(url="", input_size=(3, 384, 384), use_rel_pos_bias=True),
    "beit_l_16_512_finetune": _cfg(url="", input_size=(3, 512, 512), use_rel_pos_bias=True),
}


class DVaeEncoderBlock(nn.Cell):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_layers: int
    ):
        super(DVaeEncoderBlock, self).__init__()
        n_hid = n_out // 4
        self.post_gain = 1 / (n_layers ** 2)

        self.id_path = nn.Conv2d(
            n_in, n_out,
            kernel_size=1,
            has_bias=True
        ) if n_in != n_out else nn.Identity()

        self.res_path = nn.SequentialCell(OrderedDict([
            ('relu_1', nn.ReLU()),
            ('conv_1', nn.Conv2d(n_in, n_hid, kernel_size=3, has_bias=True)),
            ('relu_2', nn.ReLU()),
            ('conv_2', nn.Conv2d(n_hid, n_hid, kernel_size=3, has_bias=True)),
            ('relu_3', nn.ReLU()),
            ('conv_3', nn.Conv2d(n_hid, n_hid, kernel_size=3, has_bias=True)),
            ('relu_4', nn.ReLU()),
            ('conv_4', nn.Conv2d(n_hid, n_out, kernel_size=1, has_bias=True))
        ]))
    
    def construct(self, x):
        return self.id_path(x) + self.post_gain * self.res_path(x)


class DVaeEncoder(nn.Cell):
    def __init__(
        self,
        group_count: int = 4,
        n_hid: int = 256,
        n_blk_per_group: int = 2,
        input_channels: int = 3,
        vocab_size: int = 8192
    ):
        super(DVaeEncoder, self).__init__()
        self.vocab_size = vocab_size
        blk_range = range(n_blk_per_group)
        n_layers = group_count * n_blk_per_group

        self.blocks = nn.SequentialCell(OrderedDict([
			('input', nn.Conv2d(input_channels, 1 * n_hid, kernel_size=7, has_bias=True)),
			('group_1', nn.SequentialCell(OrderedDict([
				*[(f'block_{i + 1}', DVaeEncoderBlock(1 * n_hid, 
                                                  1 * n_hid,
                                                  n_layers)) for i in blk_range],
				('pool', nn.MaxPool2d(kernel_size=2, stride=2)),
			]))),
			('group_2', nn.SequentialCell(OrderedDict([
				*[(f'block_{i + 1}', DVaeEncoderBlock(1 * n_hid if i == 0 else 2 * n_hid,
                                                  2 * n_hid,
                                                  n_layers)) for i in blk_range],
				('pool', nn.MaxPool2d(kernel_size=2, stride=2)),
			]))),
			('group_3', nn.SequentialCell(OrderedDict([
				*[(f'block_{i + 1}', DVaeEncoderBlock(2 * n_hid if i == 0 else 4 * n_hid,
                                                  4 * n_hid,
                                                  n_layers)) for i in blk_range],
				('pool', nn.MaxPool2d(kernel_size=2, stride=2)),
			]))),
			('group_4', nn.SequentialCell(OrderedDict([
				*[(f'block_{i + 1}', DVaeEncoderBlock(4 * n_hid if i == 0 else 8 * n_hid,
                                                  8 * n_hid,
                                                  n_layers)) for i in blk_range],
			]))),
			('output', nn.SequentialCell(OrderedDict([
				('relu', nn.ReLU()),
				('conv', nn.Conv2d(8 * n_hid, vocab_size, kernel_size=1, has_bias=True)),
			]))),
		]))

        self.argmax = ops.Argmax(axis=1)
        self.reshape = ops.Reshape()
        self.masked_select = ops.MaskedSelect()

    def construct(self, x):
        bsz = x.shape[0]
        z_logits = self.blocks(x)
        indices = self.argmax(z_logits)
        labels = self.reshape(indices, (bsz, -1))
        return labels
        

class BEiTForPretrain(VisionTransformerEncoder):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        attn_head_dim: Optional[int] = None,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        pos_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        init_values: Optional[float] = 0.1,
        act_layer: nn.Cell = nn.GELU,
        norm_layer: nn.Cell = nn.LayerNorm,
        use_abs_pos_emb: bool = False,
        use_rel_pos_bias: bool = False,
        use_shared_rel_pos_bias: bool = True,
        vocab_size: int = 8192,
        **kwargs
    ):
        super(BEiTForPretrain, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            pos_drop_rate=pos_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_abs_pos_emb=use_abs_pos_emb,
            use_rel_pos_bias=use_rel_pos_bias,
            use_shared_rel_pos_bias=use_shared_rel_pos_bias,
            **kwargs
        )
        self.mask_token = Parameter(initializer(TruncatedNormal(0.02), (1, 1, embed_dim)))
        self.head = nn.Dense(embed_dim, vocab_size, weight_init=TruncatedNormal(0.02))
        self.norm = norm_layer((embed_dim, ))

        self._init_weights()
        self._fix_init_weights() 

    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x)
        bsz, seq_len, _ = x.shape

        mask_tokens = ops.broadcast_to(self.mask_token, (bsz, seq_len, -1))
        w = ops.expand_dims(bool_masked_pos, axis=-1).astype(mask_tokens.dtype)
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

    def construct(self, x, bool_masked_pos):
        x = self.forward_features(x, bool_masked_pos)
        x = self.norm(x)
        x = x[:, 1:]
        x = self.head(x)
        return x


class BEiTForFinetune(VisionTransformerEncoder):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        attn_head_dim: Optional[int] = None,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        pos_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        init_values: Optional[float] = 0.1,
        act_layer: nn.Cell = nn.GELU,
        norm_layer: nn.Cell = nn.LayerNorm,
        use_abs_pos_emb: bool = False,
        use_rel_pos_bias: bool = False,
        use_shared_rel_pos_bias: bool = True,
        num_classes: int = 1000,
        use_mean_pooling: bool = True,
        init_scale: float = 0.001,
        **kwargs
    ):
        super(BEiTForFinetune, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            pos_drop_rate=pos_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_abs_pos_emb=use_abs_pos_emb,
            use_rel_pos_bias=use_rel_pos_bias,
            use_shared_rel_pos_bias=use_shared_rel_pos_bias,
            **kwargs
        )
        self.use_mean_pooling = use_mean_pooling
        self.head = nn.Dense(embed_dim, num_classes, weight_init='TruncatedNormal')
        self.norm = norm_layer((embed_dim,))

        self._init_weights()
        self._fix_init_weights()
        
        self.head.weight.set_data(ops.mul(self.head.weight, init_scale))
        self.head.bias.set_data(ops.mul(self.head.bias, init_scale))

    def construct(self, x):
        x = self.forward_features(x)
        if self.use_mean_pooling:
            x = x[:, 1:].mean(axis=1)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]
        x = self.head(x)
        return x

@register_model
def dall_e(pretrained=False, freeze=True, **kwargs):
    model = DVaeEncoder(
        group_count=4, n_hid=256, n_blk_per_group=2, input_channels=3, vocab_size=8192
    )
    if pretrained:
        pass
    if freeze:
        for param in model.trainable_params():
            param.requires_grad = False
    return model


@register_model
def beit_b_16_224_pretrain(pretrained=False, **kwargs):
    model = BEiTForPretrain(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        act_layer=partial(nn.GELU, approximate=False),
        norm_layer=partial(LayerNorm, epsilon=1e-6), vocab_size=8192, **kwargs
    )
    if pretrained:
        pass
    return model


@register_model
def beit_l_16_224_pretrain(pretrained=False, **kwargs):
    model = BEiTForPretrain(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        act_layer=partial(nn.GELU, approximate=False),
        norm_layer=partial(LayerNorm, epsilon=1e-6), vocab_size=8192, **kwargs
    )
    if pretrained:
        pass
    return model
    

@register_model
def beit_b_16_224_finetune(pretrained=True, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs["beit_b_16_224_finetune"]
    model = BEiTForFinetune(
        patch_size=16, in_chans=in_chans, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        act_layer=partial(nn.GELU, approximate=False),
        qkv_bias=True, norm_layer=partial(LayerNorm, epsilon=1e-6), num_classes=num_classes, **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_chans)
    return model


@register_model
def beit_b_16_384_finetune(pretrained=True, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs["beit_b_16_384_finetune"]
    model = BEiTForFinetune(
        img_size=384, patch_size=16, in_chans=in_chans, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        act_layer=partial(nn.GELU, approximate=False),
        qkv_bias=True, norm_layer=partial(LayerNorm, epsilon=1e-6), num_classes=num_classes, **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_chans)
    return model
    

@register_model
def beit_l_16_224_finetune(pretrained=True, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs["beit_l_16_224_finetune"]
    model = BEiTForFinetune(
        patch_size=16, in_chans=in_chans, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        act_layer=partial(nn.GELU, approximate=False),
        qkv_bias=True, norm_layer=partial(LayerNorm, epsilon=1e-6), num_classes=num_classes, **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_chans)
    return model
    

@register_model
def beit_l_16_384_finetune(pretrained=True, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs["beit_l_16_384_finetune"]
    model = BEiTForFinetune(
        img_size=384, patch_size=16, in_chans=in_chans, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        act_layer=partial(nn.GELU, approximate=False),
        qkv_bias=True, norm_layer=partial(LayerNorm, epsilon=1e-6), num_classes=num_classes, **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_chans)
    return model


@register_model
def beit_l_16_512_finetune(pretrained=True, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs["beit_l_16_512_finetune"]
    model = BEiTForFinetune(
        img_size=512, patch_size=16, in_chans=in_chans, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        act_layer=partial(nn.GELU, approximate=False),
        qkv_bias=True, norm_layer=partial(LayerNorm, epsilon=1e-6), num_classes=num_classes, **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_chans)
    return model