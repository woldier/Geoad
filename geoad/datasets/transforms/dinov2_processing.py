from copy import deepcopy

import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import GaussianBlur as TVGaussianBlur
import torchvision.transforms.functional as TF

from typing import Dict, Optional, Union, Tuple, List
from PIL import Image
from mmcv.transforms.wrappers import Transform

from mmpretrain.registry import TRANSFORMS
from mmpretrain.datasets.transforms import RandomResizedCrop, RandomFlip, ColorJitter, RandomGrayscale, PackInputs, Solarize# #GaussianBlur
from mmcv.transforms import BaseTransform, Compose, RandomApply

class TVGaussianBlurImpl(TVGaussianBlur):

    def forward(self,results: Dict) -> Dict:
        img = results['img']
        pil_img = Image.fromarray(img)
        results['img'] = np.array(pil_img, dtype=img.dtype)
        return results


class GaussianBlur(RandomApply):
    # brrow from dinov2: dinov2/data/transforms.py#GaussianBlur
    def __init__(self,  prob: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # this func dont care channer index rgb or bgr, so this dont need change bgr->rgb
        super().__init__(TVGaussianBlurImpl(kernel_size=9, sigma=(radius_min, radius_max)), 1-prob)





@TRANSFORMS.register_module()
class DINOv2Transform(BaseTransform):
    def __init__(
            self,
            global_crops_scale,
            local_crops_scale,
            local_crops_number: int,
            global_crops_size=224,
            local_crops_size=96,  # for path size 14
    ):
        super().__init__()
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        self.geometric_augmentation_global = Compose([
            RandomResizedCrop(global_crops_size, crop_ratio_range=global_crops_scale),
            RandomFlip(prob=0.5, direction='horizontal'), ]
        )
        self.geometric_augmentation_local = Compose([
            RandomResizedCrop(local_crops_size, crop_ratio_range=local_crops_scale, ),
            RandomFlip(prob=0.5, direction='horizontal'),
        ])

        color_jittering = Compose([
            RandomApply([ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, backend='cv2')], prob=0.8),
            RandomGrayscale(prob=0.2, keep_channels=True),
        ])

        global_transfo1_extra = GaussianBlur(prob=1.0)
        global_transfo2_extra = Compose([
            GaussianBlur(prob=0.1), Solarize(thr=128, prob=0.2)
        ])

        local_transfo_extra = GaussianBlur(prob=0.5)

        # normalization this part move to SelfSupDataPreprocessor
        # self.normalize = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         make_normalize_transform(),
        #     ]
        # )

        self.global_transfo1 = Compose([color_jittering, global_transfo1_extra, ])  # self.normalize])
        self.global_transfo2 = Compose([color_jittering, global_transfo2_extra, ])  # self.normalize])
        self.local_transfo = Compose([color_jittering, local_transfo_extra, ])  # self.normalize])

        self.park_input = PackInputs()

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        img = results['img']

        output = {}
        im1_base = dict(img=deepcopy(img))
        im2_base = dict(img=deepcopy(img))

        im1_base = self.geometric_augmentation_global(im1_base)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(im2_base)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [self.park_input(global_crop_1)['inputs'], self.park_input(global_crop_2)['inputs']]
        #
        output["global_crops_teacher"] = [self.park_input(global_crop_1)['inputs'],
                                          self.park_input(global_crop_2)['inputs']]  # TODO 这里两个引用与global_crops是一致的，注意区分
        #
        local_crops = [
            self.park_input(self.local_transfo(self.geometric_augmentation_local(dict(img=deepcopy(img)))))['inputs'] for _ in
            range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = (1,)

        return output
