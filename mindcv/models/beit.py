import math
import numpy as np
from collections import OrderedDict
from typing import Optional, Tuple, Union
from functools import partial

import mindspore as ms
from mindspore.common.initializer import initializer
from mindspore import nn, ops, Tensor, Parameter

from .layers.drop_path import DropPath
from .layers.mlp import Mlp
from .layers.patch_embed import PatchEmbed
from .registry import register_model

__all__ = [
    "dall_e",
    "beit_b_16_224_8k_vocab",
    "beit_l_16_224_8k_vocab"
]

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


class RelativePositionBias(nn.Cell):
    def __init__(
        self,
        window_size: Tuple[int],
        num_heads: int
    ):
        super(RelativePositionBias, self).__init__()
        self.window_size = window_size
        self.num_tokens = window_size[0] * window_size[1]

        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        # 3: cls to token, token to cls, cls to cls
        self.relative_position_bias_table = Parameter(
            Tensor(np.random.randn(num_relative_distance, num_heads), dtype=ms.float32)
        )
        coords_h = np.arange(window_size[0]).reshape(window_size[0], 1).repeat(window_size[1], 1).reshape(1, -1)
        coords_w = np.arange(window_size[1]).reshape(1, window_size[1]).repeat(window_size[0], 0).reshape(1, -1)
        coords_flatten = np.concatenate([coords_h, coords_w], axis=0) # [2, Wh * Ww]

        relative_coords = coords_flatten[:, :, np.newaxis] - coords_flatten[:, np.newaxis, :] # [2, Wh * Ww, Wh * Ww]
        relative_coords = relative_coords.transpose(1, 2, 0) # [Wh * Ww, Wh * Ww, 2]
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[0] - 1

        relative_position_index = np.zeros((self.num_tokens + 1, self.num_tokens + 1),
                                           dtype=relative_coords.dtype) # [Wh * Ww + 1, Wh * Ww + 1]
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = num_relative_distance - 3
        relative_position_index[0:, 0] = num_relative_distance - 2
        relative_position_index[0, 0] = num_relative_distance - 1
        relative_position_index = Tensor(relative_position_index.reshape(-1)) 

        self.one_hot = nn.OneHot(axis=-1, depth=num_relative_distance)
        self.index = Parameter(self.one_hot(relative_position_index), requires_grad=False)

    def construct(self):
        out = ops.matmul(self.index, self.relative_position_bias_table)
        out = ops.reshape(out, (self.num_tokens + 1, self.num_tokens + 1, -1))
        out = ops.transpose(out, (2, 0, 1))
        out = ops.expand_dims(out, 0)
        return out


class Mlp(nn.Cell):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Cell = nn.GELU,
        drop: float = 0.
    ):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=True)
        self.drop = nn.Dropout(1 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
        rel_pos_bias: Union[nn.Cell, bool] = False,
        window_size: Optional[Tuple] = None
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * num_heads

        if qk_scale:
            self.scale = Tensor(qk_scale)
        else:
            self.scale = Tensor(head_dim ** -0.5)

        if qkv_bias:
            self.qkv = nn.Dense(dim, all_head_dim * 3)
        else:
            self.qkv = nn.Dense(dim, all_head_dim * 3, has_bias=False)

        if isinstance(rel_pos_bias, RelativePositionBias):
            self.relative_position_bias = rel_pos_bias
        elif rel_pos_bias:
            self.relative_position_bias = RelativePositionBias(window_size, num_heads)
        else:
            self.relative_position_bias = None

        self.attn_drop = nn.Dropout(1 - attn_drop)
        self.proj = nn.Dense(all_head_dim, dim)
        self.proj_drop = nn.Dropout(1 - proj_drop)

        self.mul = ops.Mul()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.unstack = ops.Unstack(axis=0)
        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = self.reshape(qkv, (b, n, 3, self.num_heads, c // self.num_heads))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = self.unstack(qkv)

        attn = self.q_matmul_k(q, k)
        attn = self.mul(attn, self.scale)

        if self.relative_position_bias is not None:
            rel_pos_bias = self.relative_position_bias()
            attn  = attn + rel_pos_bias

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = self.attn_matmul_v(attn, v)
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.reshape(out, (b, n, c))
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class LayerScale(nn.Cell):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5
    ):
        super(LayerScale, self).__init__()
        self.gamma = Parameter(initializer(init_values, dim))
    
    def construct(self, x):
        return self.gamma * x
    

class Block(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        attn_head_dim: Optional[int] = None,
        mlp_ratio: float = 4.,
        drop_path: float = 0.,
        init_values: Optional[float] = None,
        act_layer: nn.Cell = nn.GELU,
        norm_layer: nn.Cell = nn.LayerNorm,
        rel_pos_bias: Union[nn.Cell, bool] = False,
        window_size: Optional[Tuple] = None
    ):
        super(Block, self).__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop, attn_head_dim=attn_head_dim,
            rel_pos_bias=rel_pos_bias, window_size=window_size
        )
        self.ls1 = LayerScale(dim=dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer((dim,))
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, drop=proj_drop
        )
        self.ls2 = LayerScale(dim=dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def construct(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformerEncoder(nn.Cell):
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
        **kwargs
    ):
        super(VisionTransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(image_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = Parameter(initializer('truncatedNormal', (1, 1, embed_dim)))
        
        self.pos_emb = Parameter(initializer('truncatedNormal', (1, self.num_patches + 1, embed_dim))) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(1 - pos_drop_rate)

        if use_shared_rel_pos_bias:
            rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patches_resolution,
                                                num_heads=num_heads)
        elif use_rel_pos_bias:
            rel_pos_bias = True
        else:
            rel_pos_bias = False

        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.CellList([
            Block(
                dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop_rate, proj_drop=proj_drop_rate, attn_head_dim=attn_head_dim,
                mlp_ratio=mlp_ratio, drop_path=dpr[i], init_values=init_values,
                act_layer=act_layer, norm_layer=norm_layer, rel_pos_bias=rel_pos_bias,
                window_size=self.patch_embed.patches_resolution 
            ) for i in range(depth)
        ])

    def get_num_layeres(self):
        return len(self.blocks)
    
    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    initializer('truncatedNormal', cell.weight.shape, cell.weight.dtype)
                )
                if cell.bias is not None:
                    cell.bias.set_data(
                        initializer('zeros', cell.bias.shape, cell.bias.dtype)
                    )
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(
                    initializer('ones', cell.gamma.shape, cell.gamma.dtype)
                )
                cell.beta.set_data(
                    initializer('zeros', cell.beta.shape, cell.beta.dtype)
                )
            elif isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    initializer('truncatedNormal', cell.weight.shape, cell.weight.dtype)
                )
                if cell.bias is not None:
                    cell.bias.set_data(
                        initializer('zeros', cell.bias.shape, cell.bias.dtype)
                    )
    
    def _fix_init_weights(self):
        for i, block in enumerate(self.blocks):
            block.attn.proj.weight.set_data(
                ops.div(block.attn.proj.weight, math.sqrt(2.0 * (i + 1)))
            )
            block.mlp.fc2.weight.set_data(
                ops.div(block.mlp.fc2.weight, math.sqrt(2.0 * (i + 1)))
            )

    def forward_features(self, x):
        x = self.patch_embed(x)
        bsz = x.shape[0]

        cls_token = ops.broadcast_to(self.cls_token, (bsz, -1, -1))
        x = ops.concat((cls_token, x), axis=1)
        
        if self.pos_emb is not None:
            x = x + self.pos_emb
        x = self.pos_drop

        for blk in self.blocks:
            x = blk(x)

        return x

    def construct(self, x):
        return self.forward_features(x)


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
        self.mask_token = Parameter(initializer('truncatedNormal', (1, 1, embed_dim)))
        self.head = nn.Dense(embed_dim, vocab_size, weight_init='truncatedNormal')
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

        if self.pos_emb is not None:
            x = x + self.pos_emb
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
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
        self.head.bias.set_data(ops.mul(self.head_bias, init_scale))

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
def beit_b_16_224_8k_vocab(pretrained=False, **kwargs):
    model = BEiTForPretrain(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), vocab_size=8192, **kwargs
    )
    if pretrained:
        pass
    return model


@register_model
def beit_l_16_224_8k_vocab(pretrained=False, **kwargs):
    model = BEiTForPretrain(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), vocab_size=8192, **kwargs
    )
    if pretrained:
        pass
    return model