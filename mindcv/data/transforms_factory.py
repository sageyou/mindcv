"""
Transform operation list
"""

import math
from typing import Optional

from mindspore.dataset import vision
from mindspore.dataset.vision import Inter
from mindspore.dataset.transforms import Compose

from .auto_augment import (
    augment_and_mix_transform,
    auto_augment_transform,
    rand_augment_transform,
    trivial_augment_wide_transform,
)
from .constants import DEFAULT_CROP_PCT, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .mask_generator import BlockWiseMaskGenerator, PatchAlignedMaskGenerator

__all__ = [
    "create_transforms",
    "create_transforms_pretrain"
]


def transforms_imagenet_train(
    image_resize=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.333),
    hflip=0.5,
    vflip=0.0,
    color_jitter=None,
    auto_augment=None,
    interpolation="bilinear",
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    re_prob=0.0,
    re_scale=(0.02, 0.33),
    re_ratio=(0.3, 3.3),
    re_value=0,
    re_max_attempts=10,
    separate=False,
):
    """Transform operation list when training on ImageNet."""
    # Define map operations for training dataset
    if hasattr(Inter, interpolation.upper()):
        interpolation = getattr(Inter, interpolation.upper())
    else:
        interpolation = Inter.BILINEAR

    primary_tfl = [
        vision.RandomCropDecodeResize(
            size=image_resize,
            scale=scale,
            ratio=ratio,
            interpolation=interpolation,
        )
    ]
    if hflip > 0.0:
        primary_tfl += [vision.RandomHorizontalFlip(prob=hflip)]
    if vflip > 0.0:
        primary_tfl += [vision.RandomVerticalFlip(prob=vflip)]

    secondary_tfl = []
    if auto_augment is not None:
        assert isinstance(auto_augment, str)
        if isinstance(image_resize, (tuple, list)):
            image_resize_min = min(image_resize)
        else:
            image_resize_min = image_resize
        augement_params = dict(
            translate_const=int(image_resize_min * 0.45),
            img_mean=tuple([min(255, round(x)) for x in mean]),
        )
        augement_params["interpolation"] = interpolation
        if auto_augment.startswith("randaug"):
            secondary_tfl += [rand_augment_transform(auto_augment, augement_params)]
        elif auto_augment.startswith("autoaug") or auto_augment.startswith("3a"):
            secondary_tfl += [auto_augment_transform(auto_augment, augement_params)]
        elif auto_augment.startswith("trivialaugwide"):
            secondary_tfl += [trivial_augment_wide_transform(auto_augment, augement_params)]
        elif auto_augment.startswith("augmix"):
            augement_params["translate_pct"] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, augement_params)]
        else:
            assert False, "Unknown auto augment policy (%s)" % auto_augment
    elif color_jitter is not None:
        if isinstance(color_jitter, (list, tuple)):
            # color jitter shoulf be a 3-tuple/list for brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [vision.RandomColorAdjust(*color_jitter)]

    final_tfl = []
    final_tfl += [
        vision.Normalize(mean=mean, std=std),
        vision.HWC2CHW(),
    ]
    if re_prob > 0.0:
        final_tfl.append(
            vision.RandomErasing(
                prob=re_prob,
                scale=re_scale,
                ratio=re_ratio,
                value=re_value,
                max_attempts=re_max_attempts,
            )
        )

    if separate:
        return primary_tfl, secondary_tfl, final_tfl
    return primary_tfl + secondary_tfl + final_tfl


def transforms_imagenet_eval(
    image_resize=224,
    crop_pct=DEFAULT_CROP_PCT,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    interpolation="bilinear",
):
    """Transform operation list when evaluating on ImageNet."""
    if isinstance(image_resize, (tuple, list)):
        assert len(image_resize) == 2
        if image_resize[-1] == image_resize[-2]:
            scale_size = int(math.floor(image_resize[0] / crop_pct))
        else:
            scale_size = tuple(int(x / crop_pct) for x in image_resize)
    else:
        scale_size = int(math.floor(image_resize / crop_pct))

    # Define map operations for training dataset
    if hasattr(Inter, interpolation.upper()):
        interpolation = getattr(Inter, interpolation.upper())
    else:
        interpolation = Inter.BILINEAR
    trans_list = [
        vision.Decode(),
        vision.Resize(scale_size, interpolation=interpolation),
        vision.CenterCrop(image_resize),
        vision.Normalize(mean=mean, std=std),
        vision.HWC2CHW(),
    ]

    return trans_list


def transforms_cifar(resize=224, is_training=True):
    """Transform operation list when training or evaluating on cifar."""
    trans = []
    if is_training:
        trans += [
            vision.RandomCrop((32, 32), (4, 4, 4, 4)),
            vision.RandomHorizontalFlip(prob=0.5),
        ]

    trans += [
        vision.Resize(resize),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        vision.HWC2CHW(),
    ]

    return trans


def transforms_mnist(resize=224):
    """Transform operation list when training or evaluating on mnist."""
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    trans = [
        vision.Resize(size=resize, interpolation=Inter.LINEAR),
        vision.Rescale(rescale, shift),
        vision.Rescale(rescale_nml, shift_nml),
        vision.HWC2CHW(),
    ]
    return trans


def create_transforms(
    dataset_name="",
    image_resize=224,
    is_training=False,
    auto_augment=None,
    separate=False,
    **kwargs,
):
    r"""Creates a list of transform operation on image data.

    Args:
        dataset_name (str): if '', customized dataset. Currently, apply the same transform pipeline as ImageNet.
            if standard dataset name is given including imagenet, cifar10, mnist, preset transforms will be returned.
            Default: ''.
        image_resize (int): the image size after resize for adapting to network. Default: 224.
        is_training (bool): if True, augmentation will be applied if support. Default: False.
        auto_augment(str)：augmentation strategies, such as "augmix", "autoaug" etc.
        separate: separate the image origin and the image been transformed.
        **kwargs: additional args parsed to `transforms_imagenet_train` and `transforms_imagenet_eval`

    Returns:
        A list of transformation operations
    """

    dataset_name = dataset_name.lower()

    if dataset_name in ("imagenet", ""):
        trans_args = dict(image_resize=image_resize, **kwargs)
        if is_training:
            return transforms_imagenet_train(auto_augment=auto_augment, separate=separate, **trans_args)

        return transforms_imagenet_eval(**trans_args)
    elif dataset_name in ("cifar10", "cifar100"):
        trans_list = transforms_cifar(resize=image_resize, is_training=is_training)
        return trans_list
    elif dataset_name == "mnist":
        trans_list = transforms_mnist(resize=image_resize)
        return trans_list
    else:
        raise NotImplementedError(
            f"Only supports creating transforms for ['imagenet'] datasets, but got {dataset_name}."
        )


class RandomResizedCropWithTwoResolution:
    def __init__(self,
        first_size,
        second_size,
        first_interpolation,
        second_interpolation, # lanczos is not implemented in MindSpore
        scale,
        ratio
    ):
        self.first_transform = vision.RandomResizedCrop(first_size, scale, ratio, first_interpolation)
        self.second_transform = vision.RandomResizedCrop(second_size, scale, ratio, second_interpolation)

    def __call__(self, img):
        return self.first_transform(img), self.second_transform(img)


class TransformsForPretrain:
    def __init__(
        self,
        first_resize=224,
        second_resize: Optional[int] = None,
        tokenizer_type: str = "dall-e",
        mask_type: str = "block-wise",
        scale=(0.08, 1.0),
        ratio=(0.75, 1.333),
        hflip=0.5,
        color_jitter=None,
        first_interpolation="bicubic",
        second_interpolation="bilinear", # lanczos is not implemented is MindSpore
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        **kwargs
    ):
        if hasattr(Inter, first_interpolation.upper()):
            first_interpolation = getattr(Inter, first_interpolation.upper())
        else:
            first_interpolation = Inter.BILINEAR
            
        if second_resize is not None:
            common_transform = [
                vision.Decode()
            ]
            if color_jitter is not None:
                if isinstance(color_jitter, (list, tuple)):
                    # color jitter shoulf be a 3-tuple/list for brightness/contrast/saturation
                    # or 4 if also augmenting hue
                    assert len(color_jitter) in (3, 4)
                else:
                    color_jitter = (float(color_jitter),) * 3
                common_transform += [vision.RandomColorAdjust(*color_jitter)]

            if hflip > 0.0:
                common_transform += [vision.RandomHorizontalFlip(prob=hflip)]

            if hasattr(Inter, second_interpolation.upper()):
                second_interpolation = getattr(Inter, second_interpolation.upper())
            else:
                second_interpolation = Inter.BILINEAR
            
            common_transform += [RandomResizedCropWithTwoResolution(
                                    first_resize, second_resize,
                                    first_interpolation, second_interpolation,
                                    scale, ratio
                                )]
            self.common_transform = Compose(common_transform)

            self.patch_transform = Compose([
                vision.Normalize(mean=mean, std=std),
                vision.HWC2CHW()
            ])

            if tokenizer_type == "dall-e": # beit
                self.visual_token_transform = Compose([
                    vision.ToTensor(),
                    lambda x: (1 - 2 * 0.1) * x + 0.1
                ])
            elif tokenizer_type == "vqkd": # beit v2
                self.visual_token_transform = Compose([
                    vision.ToTensor()
                ])
            elif tokenizer_type == "clip": # eva, eva-02
                self.visual_token_transform = Compose([
                    vision.ToTensor(),
                    vision.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                        is_hwc=False
                    )
                ])

            if mask_type == "block-wise": # beit, beit v2, eva, eva-02
                self.masked_position_generator = BlockWiseMaskGenerator(
                    window_size=kwargs["window_size"],
                    num_masking_patches=kwargs["num_mask_patches"],
                    max_num_patches=kwargs["max_num_patches"],
                    min_num_patches=kwargs["min_num_patches"]
                )
            elif mask_type == "patch-aligned": # SimMIM
                self.masked_position_generator = PatchAlignedMaskGenerator(
                    input_size=first_resize,
                    mask_patch_size=kwargs["mask_patch_size"],
                    model_patch_size=kwargs["model_patch_size"],
                    mask_ratio=kwargs["mask_ratio"]
                )
            else:
                raise NotImplementedError()

            self.output_columns = ["patch", "token", "mask"]
        else:
            self.common_transform = None

            patch_transform = [
                vision.RandomCropDecodeResize(
                    size=first_resize,
                    scale=scale,
                    ratio=ratio,
                    interpolation=first_interpolation
                )
            ]

            if hflip > 0.0:
                patch_transform += [vision.RandomHorizontalFlip(hflip)]

            patch_transform += [
                vision.Normalize(mean=mean, std=std),
                vision.HWC2CHW()
            ]
            self.patch_transform = Compose(patch_transform)
            
            if mask_type == "block-wise": # beit, beit v2, eva, eva-02
                self.masked_position_generator = BlockWiseMaskGenerator(
                    window_size=kwargs["window_size"],
                    num_masking_patches=kwargs["num_mask_patches"],
                    max_num_patches=kwargs["max_num_patches"],
                    min_num_patches=kwargs["min_num_patches"]
                )
                self.output_columns = ["patch", "mask"]
            elif mask_type == "patch-aligned": # SimMIM
                self.masked_position_generator = PatchAlignedMaskGenerator(
                    input_size=first_resize,
                    mask_patch_size=kwargs["mask_patch_size"],
                    model_patch_size=kwargs["model_patch_size"],
                    mask_ratio=kwargs["mask_ratio"]
                )
                self.output_columns = ["patch", "mask"]
            elif mask_type == "none":
                self.masked_position_generator = None
                self.output_columns = ["patch"]
            else:
                raise NotImplementedError()

    def __call__(self, image):
        if self.common_transform is not None: # for beit, beit v2, eva, eva-02
            patches, visual_tokens = self.common_transform(image)
            patches = self.patch_transform(patches)
            visual_tokens = self.visual_token_transform(visual_tokens)
            masks = self.masked_position_generator()
            return patches, visual_tokens, masks
        else:
            patches = self.patch_transform(image)
            if self.masked_position_generator is not None: # for SimMIM
                masks = self.masked_position_generator()
                return patches, masks
            else: # for MAE
                return patches


def create_transforms_pretrain(
    dataset_name="",
    image_resize=224,
    **kwargs
):
    if dataset_name in ("imagenet", ""):
        trans_args = dict(first_resize=image_resize, **kwargs)
        return TransformsForPretrain(**trans_args)
    else:
        raise NotImplementedError()