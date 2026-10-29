import math
import torch
import random
import numpy as np
from typing import Union, Optional, Sequence

from mmengine.model import ImgDataPreprocessor

from mmpretrain.registry import MODELS


class MaskingGenerator:
    def __init__(
            self,
            input_size,
            num_masking_patches=None,
            min_num_patches=4,
            max_num_patches=None,
            min_aspect=0.3,
            max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


@MODELS.register_module()
class DINOV2DataPreprocessor(ImgDataPreprocessor):
    """


        collate_fn:
        >>> mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        >>> mask_probability=cfg.ibot.mask_sample_probability,
        >>> n_tokens=n_tokens,
        >>> mask_generator=mask_generator,

        >>> input_size=(img_size // patch_size, img_size // patch_size),
        >>> max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    """

    def __init__(self, mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None, pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0, to_rgb: bool = False,
                 bgr_to_rgb: bool = False, rgb_to_bgr: bool = False,
                 non_blocking: Optional[bool] = False,
                 # refer dinov2 repo. dinov2/configs/ssl_default_config.yaml
                 mask_ratio_tuple: tuple = (0.1, 0.5), mask_probability: float = 0.5,
                 n_tokens: int = (224 // 14) ** 2,
                 mask_generator: dict = dict(input_size=(224 // 14, 224 // 14), max_num_patches=0.5 * (224 // 14) ** 2)
                 # noqa
                 ):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking)
        self.n_tokens = n_tokens
        self.mask_ratio_tuple = mask_ratio_tuple
        self.mask_probability = mask_probability
        self.mask_generator = MaskingGenerator(**mask_generator)
        self._channel_conversion = to_rgb or bgr_to_rgb or rgb_to_bgr

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        data = [val for _, val in data.items()]
        global_crops, global_crops_teacher, local_crops, offsets = self.cast_data(data)

        res_dict = dict()
        global_crops = self.transform(global_crops)  # channel transform and norm
        global_crops_teacher = self.transform(global_crops_teacher)  # channel transform and norm
        local_crops = self.transform(local_crops)  # channel transform and norm

        collated_global_crops = torch.cat(global_crops, dim=0)  # b*view1 + b*view2 not b*(view1+view2)
        collated_local_crops = torch.cat(local_crops, dim=0)

        B = len(collated_global_crops)
        N = self.n_tokens
        n_samples_masked = int(B * self.mask_probability)

        probs = torch.linspace(*self.mask_ratio_tuple, n_samples_masked + 1)
        upperbound = 0
        masks_list = []

        for i in range(0, n_samples_masked):
            prob_min = probs[i]
            prob_max = probs[i + 1]
            masks_list.append(torch.BoolTensor(self.mask_generator(int(N * random.uniform(prob_min, prob_max)))))
            upperbound += int(N * prob_max)

        for i in range(n_samples_masked, B):
            masks_list.append(torch.BoolTensor(self.mask_generator(0)))

        random.shuffle(masks_list)

        collated_masks = torch.stack(masks_list).flatten(1)
        mask_indices_list = collated_masks.flatten().nonzero().flatten()

        masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[
            collated_masks]

        n_masked_patches = torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long)
        collated_masks, mask_indices_list, masks_weight, n_masked_patches = \
            self.cast_data((collated_masks, mask_indices_list, masks_weight, n_masked_patches))

        return {
            "collated_global_crops": collated_global_crops,
            "collated_local_crops": collated_local_crops,
            "collated_masks": collated_masks,
            "mask_indices_list": mask_indices_list,
            "masks_weight": masks_weight,
            "upperbound": upperbound,
            "n_masked_patches": n_masked_patches,
            "local_crop_numbers": len(local_crops)
        }

    def transform(self, batch_inputs):
        if isinstance(batch_inputs, list): # noqa
            # channel transform
            if self._channel_conversion:
                batch_inputs = [
                    _input[:, [2, 1, 0], ...] for _input in batch_inputs
                ]

            # convert to float after channel conversion to ensure efficiency
            batch_inputs = [_input.float() for _input in batch_inputs]

            # normalization.
            if self._enable_normalize:
                batch_inputs = [(_input - self.mean) / self.std
                                for _input in batch_inputs]
        else:
            # channel transform
            if self._channel_conversion:
                batch_inputs = batch_inputs[:, [2, 1, 0], ...]

            # convert to float after channel conversion to ensure efficiency
            batch_inputs = batch_inputs.float()

            # normalization.
            if self._enable_normalize:
                batch_inputs = (batch_inputs - self.mean) / self.std
        return batch_inputs
