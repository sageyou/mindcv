"""
Some utils while building models
"""
import collections.abc
import difflib
import logging
import os
import numpy as np
from copy import deepcopy
from itertools import repeat
from typing import Callable, Dict, List, Optional
from scipy import interpolate

import mindspore.nn as nn
from mindspore import ops, Tensor, Parameter
from mindspore import load_checkpoint, load_param_into_net

from ..utils.download import DownLoad, get_default_download_root
from .features import FeatureExtractWrapper


def get_checkpoint_download_root():
    return os.path.join(get_default_download_root(), "models")


class ConfigDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_pretrained(model, default_cfg, num_classes=1000, in_channels=3, filter_fn=None):
    """load pretrained model depending on cfgs of model"""
    if "url" not in default_cfg or not default_cfg["url"]:
        logging.warning("Pretrained model URL is invalid")
        return

    # download files
    download_path = get_checkpoint_download_root()
    os.makedirs(download_path, exist_ok=True)
    DownLoad().download_url(default_cfg["url"], path=download_path)

    param_dict = load_checkpoint(os.path.join(download_path, os.path.basename(default_cfg["url"])))

    if in_channels == 1:
        conv1_name = default_cfg["first_conv"]
        logging.info("Converting first conv (%s) from 3 to 1 channel", conv1_name)
        con1_weight = param_dict[conv1_name + ".weight"]
        con1_weight.set_data(con1_weight.sum(axis=1, keepdims=True), slice_shape=True)
    elif in_channels != 3:
        raise ValueError("Invalid in_channels for pretrained weights")

    classifier_name = default_cfg["classifier"]
    if num_classes == 1000 and default_cfg["num_classes"] == 1001:
        classifier_weight = param_dict[classifier_name + ".weight"]
        classifier_weight.set_data(classifier_weight[:1000], slice_shape=True)
        classifier_bias = param_dict[classifier_name + ".bias"]
        classifier_bias.set_data(classifier_bias[:1000], slice_shape=True)
    elif num_classes != default_cfg["num_classes"]:
        params_names = list(param_dict.keys())
        for param_name in _search_param_name(params_names, classifier_name + ".weight"):
            param_dict.pop(
                param_name,
                "Parameter {} has been deleted from ParamDict.".format(param_name),
            )
        for param_name in _search_param_name(params_names, classifier_name + ".bias"):
            param_dict.pop(
                param_name,
                "Parameter {} has been deleted from ParamDict.".format(param_name),
            )

    if filter_fn is not None:
        param_dict = filter_fn(param_dict)

    load_param_into_net(model, param_dict)


def make_divisible(
    v: float,
    divisor: int,
    min_value: Optional[int] = None,
) -> int:
    """Find the smallest integer larger than v and divisible by divisor."""
    if not min_value:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


def _search_param_name(params_names: List, param_name: str) -> list:
    search_results = []
    for pi in params_names:
        if param_name in pi:
            search_results.append(pi)
    return search_results


def auto_map(model, param_dict):
    """Raname part of the param_dict such that names from checkpoint and model are consistent"""
    updated_param_dict = deepcopy(param_dict)
    net_param = model.get_parameters()
    ckpt_param = list(updated_param_dict.keys())
    remap = {}

    for param in net_param:
        if param.name not in ckpt_param:
            print("Cannot find a param to load: ", param.name)
            poss = difflib.get_close_matches(param.name, ckpt_param, n=3, cutoff=0.6)
            if len(poss) > 0:
                print("=> Find most matched param: ", poss[0], ", loaded")
                updated_param_dict[param.name] = updated_param_dict.pop(poss[0])  # replace
                remap[param.name] = poss[0]
            else:
                raise ValueError("Cannot find any matching param from: ", ckpt_param)

    if remap != {}:
        print("WARNING: Auto mapping succeed. Please check the found mapping names to ensure correctness")
        print("\tNet Param\t<---\tCkpt Param")
        for k in remap:
            print(f"\t{k}\t<---\t{remap[k]}")

    return updated_param_dict


def load_model_checkpoint(model: nn.Cell, checkpoint_path: str = "", ema: bool = False, auto_mapping: bool = False):
    """Model loads checkpoint.

    Args:
        model (nn.Cell): The model which loads the checkpoint.
        checkpoint_path (str): The path of checkpoint files. Default: "".
        ema (bool): Whether use ema method. Default: False.
        auto_mapping (bool): Whether to automatically map the names of checkpoint weights
            to the names of model weights when there are differences in names. Default: False.
    """

    if os.path.exists(checkpoint_path):
        checkpoint_param = load_checkpoint(checkpoint_path)

        if auto_mapping:
            checkpoint_param = auto_map(model, checkpoint_param)

        ema_param_dict = dict()

        for param in checkpoint_param:
            if param.startswith("ema"):
                new_name = param.split("ema.")[1]
                ema_data = checkpoint_param[param]
                ema_data.name = new_name
                ema_param_dict[new_name] = ema_data

        if ema_param_dict and ema:
            load_param_into_net(model, ema_param_dict)
        elif bool(ema_param_dict) is False and ema:
            raise ValueError("chekpoint_param does not contain ema_parameter, please set ema is False.")
        else:
            load_param_into_net(model, checkpoint_param)


def build_model_with_cfg(
    model_cls: Callable,
    pretrained: bool,
    default_cfg: Dict,
    features_only: bool = False,
    out_indices: List[int] = [0, 1, 2, 3, 4],
    **kwargs,
):
    """Build model with specific model configurations

    Args:
        model_cls (nn.Cell): Model class
        pretrained (bool): Whether to load pretrained weights.
        default_cfg (Dict): Configuration for pretrained weights.
        features_only (bool): Output the features at different strides instead. Default: False
        out_indices (list[int]): The indicies of the output features when `features_only` is `True`.
            Default: [0, 1, 2, 3, 4]
    """
    model = model_cls(**kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, kwargs.get("num_classes", 1000), kwargs.get("in_channels", 3))

    if features_only:
        # wrap the model, output the feature pyramid instead
        try:
            model = FeatureExtractWrapper(model, out_indices=out_indices)
        except AttributeError as e:
            raise RuntimeError(f"`feature_only` is not implemented for `{model_cls.__name__}` model.") from e

    return model


def interpolate_relative_position_bias(checkpoint_params, network):
    if "rel_pos_bias.relative_position_bias_table" in checkpoint_params \
        and isinstance(network.rel_pos_bias, nn.CellList):

        num_layers = network.get_num_layers()
        rel_pos_bias = checkpoint_params["rel_pos_bias.relative_position_bias_table"]
        for i in range(num_layers):
            checkpoint_params[f"rel_pos_bias.{i}.relative_position_bias_table"] = rel_pos_bias.clone()
        checkpoint_params.pop("rel_pos_bias.relative_position_bias_table")

    elif "rel_pos_bias.0.relative_position_bias_table" in checkpoint_params \
        and not isinstance(network.rel_pos_bias, nn.CellList) \
        and isinstance(network.rel_pos_bias, nn.Cell):

        raise NotImplementedError("Converting multiple relative position bias to one is not supported.")

    all_keys = list(checkpoint_params.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_params.pop(key)

        if "relative_position_bias_table" in key:
            bias_table = checkpoint_params[key]
            src_num_pos, num_attn_heads = bias_table.shape
            dst_num_pos, _ = network.parameters_dict()[key].shape
            dst_patch_shape = network.patch_embed.patches_resolution
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError("Unsquared patch is not supported.")

            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                print("Position interpolate for %s from %dx%d to %dx%d" % \
                      (key, src_size, src_size, dst_size, dst_size))

                extra_tokens = bias_table[-num_extra_tokens:, :]
                rel_pos_bias = bias_table[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]
                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                all_rel_pos_bias = []
                for i in range(num_attn_heads):
                    z = ops.reshape(rel_pos_bias[:, i], (src_size, src_size)).asnumpy()
                    f = interpolate.interp2d(x, y, z, kind="cubic")
                    all_rel_pos_bias.append(ops.reshape(Tensor(f(dx, dy), dtype=bias_table.dtype), (-1, 1)))

                rel_pos_bias = ops.concat(all_rel_pos_bias, axis=-1)
                new_rel_pos_bias = ops.concat((rel_pos_bias, extra_tokens), axis=0)
                checkpoint_params[key] = Parameter(new_rel_pos_bias)

    return checkpoint_params


def interpolate_pos_embed(checkpoint_params, network):
    pos_embed_checkpoint = checkpoint_params["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = network.patch_embed.num_patches
    num_extra_tokens = network.pos_embed.shape[-2] - num_patches

    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = ops.reshape(pos_tokens, (-1, orig_size, orig_size, embedding_size))
        pos_tokens = ops.transpose(pos_tokens, (0, 3, 1, 2))
        pos_tokens = ops.interpolate(pos_tokens, size=(new_size, new_size), 
                                     mode='bicubic', align_corners=False) # require MindSpore 2.0
        pos_tokens = ops.transpose(pos_tokens, (0, 2, 3, 1))
        pos_tokens = ops.reshape(pos_tokens, (-1, new_size * new_size, embedding_size))
        new_pos_embed = ops.concat((extra_tokens, pos_tokens), axis=1)
        checkpoint_params['pos_embed'] = Parameter(new_pos_embed)
        
    return checkpoint_params