import torch

from mmengine import print_log
from torch import nn
import numpy as np
from typing import List, Optional, Union, Dict
from ..peft import FRMsVisionTransformer
from mmengine.model import BaseModel
from mmengine.logging import MessageHub

from mmpretrain.models import CosineEMA

from mmpretrain.registry import MODELS
from ...utils.typing_utils import ConfigType

try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


class CosineScheduler(nn.Module):
    def __init__(self, base_value, final_value,  # total_iters, remove total_inters for inner support
                 warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        self.base_value = base_value
        self.final_value = final_value
        # self.total_iters = total_iters
        self.register_buffer('steps', torch.tensor(0, dtype=torch.long, ))
        self.warmup_iters = warmup_iters
        self.start_warmup_value = start_warmup_value
        self.freeze_iters = freeze_iters

    def forward(self):
        message_hub = MessageHub.get_current_instance()
        max_iters = message_hub.get_info('max_iters')
        assert self.steps <= max_iters
        temp = 0.
        if self.steps < self.freeze_iters:
            pass
        elif self.freeze_iters <= self.steps < self.freeze_iters + self.warmup_iters:
            # temp = self.start_warmup_value + self.base_value * ((self.setp - self.freeze_iters) / self.warmup_iters)
            temp = self.start_warmup_value + (self.base_value - self.start_warmup_value) * \
                   ((self.steps - self.freeze_iters) / self.warmup_iters)
        elif self.freeze_iters + self.warmup_iters <= self.steps:

            iters = max_iters - self.warmup_iters - self.freeze_iters
            current = self.steps - self.warmup_iters - self.freeze_iters
            temp = self.final_value + 0.5 * (self.base_value - self.final_value) * (
                        1 + torch.cos(torch.pi * current / iters))
        self.steps += 1
        return temp


@MODELS.register_module()
class DINOv2(BaseModel):
    """

    Example:
        dino_head:
            >>> dino_head = dict(
            >>>     in_dim=embed_dim,
            >>>     out_dim=cfg.dino.head_n_prototypes,
            >>>     hidden_dim=cfg.dino.head_hidden_dim,
            >>>     bottleneck_dim=cfg.dino.head_bottleneck_dim,
            >>>     nlayers=cfg.dino.head_nlayers,
            >>> )

        ibot_head:
            >>> ibot_head = dict(
            >>>     in_dim=embed_dim,
            >>>     out_dim=cfg.ibot.head_n_prototypes,
            >>>     hidden_dim=cfg.ibot.head_hidden_dim,
            >>>     bottleneck_dim=cfg.ibot.head_bottleneck_dim,
            >>>     nlayers=cfg.ibot.head_nlayers, 
            >>> )

        teacher_temp
        >>> teacher_temp = dict(
        >>>     base_value=cfg.teacher["teacher_temp"],
        >>>     final_value=cfg.teacher["teacher_temp"],
        >>>     total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        >>>     warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        >>>     start_warmup_value=cfg.teacher["warmup_teacher_temp"], )

    """  # noqa

    def __init__(
            self,
            backbone: dict, embed_dim: Optional[int] = None,
            dino_head: Optional[dict] = None,
            ibot_head: Optional[dict] = None,
            dino_loss_weight: float = 1.0, koleo_loss_weight: float = 0.1,
            ibot_loss_weight: float = 1.0, ibot_separate_head: bool = False,
            teacher_temp: Optional[dict] = None, centering: str = 'centering',
            data_preprocessor: Optional[Union[dict, nn.Module]] = None,
            init_cfg: Optional[dict] = None):

        data_preprocessor = data_preprocessor or {}
        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'SelfSupDataPreprocessor')
            data_preprocessor = MODELS.build(data_preprocessor)
        elif not isinstance(data_preprocessor, nn.Module):
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(data_preprocessor)}')
        super(DINOv2, self).__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        print_log(f"====================== DINO INIT =========================", logger='current')

        # backbone
        if not isinstance(backbone, nn.Module):
            backbone = MODELS.build(backbone)
        self.backbone = backbone

        # check embed_dim
        embed_dim = embed_dim or getattr(self.backbone, 'embed_dims')
        assert embed_dim == getattr(self.backbone, 'embed_dims'), \
            print_log(f"embedding dimension {embed_dim} mismatch with backbone {getattr(backbone, 'embed_dim')}",
                      logger='current')
        self.embed_dim = embed_dim
        print_log(f"architecture : embed_dim: {embed_dim}", logger='current')

        self.dino_out_dim = getattr(dino_head, 'out_dim')

        self.do_dino = dino_loss_weight > 0
        self.do_koleo = koleo_loss_weight > 0
        self.do_ibot = ibot_loss_weight > 0
        self.ibot_separate_head = ibot_separate_head

        self.dino_loss_weight = dino_loss_weight
        self.koleo_loss_weight = koleo_loss_weight
        self.ibot_loss_weight = ibot_loss_weight

        if self.do_dino:
            print_log(f"DINO Head is used", logger='current')
            self.dino_loss_weight = dino_loss_weight
            if dino_head is not None and not isinstance(dino_head, nn.Module):
                dino_head.setdefault('in_dim', embed_dim)

            self.dino_loss = MODELS.build(dict(type="DINOLoss", out_dim=self.dino_out_dim))
            if self.do_koleo:
                # self.koleo_loss = KoLeoLoss()
                print_log(f"applying KOLEO regularization", logger='current')
                self.koleo_loss = MODELS.build(dict(type='KoLeoLoss'))
        else:
            print_log(f"DINO Head is not used", logger='current')

        if self.do_dino or self.do_ibot:
            dino_head.setdefault('type', 'DINOHead')
            self.dino_head = MODELS.build(dino_head)

        if self.do_ibot:
            self.ibot_loss_weight = ibot_loss_weight
            self.ibot_out_dim = getattr(ibot_head, 'out_dim')
            self.ibot_patch_loss = MODELS.build(dict(type='iBOTPatchLoss', patch_out_dim=self.ibot_out_dim))
            if self.ibot_separate_head:
                print_log(f"iBOT Head is used", logger='current')
                if ibot_head is not None and not isinstance(ibot_head, nn.Module):
                    ibot_head.setdefault('type', 'DINOHead')
                    ibot_head.setdefault('in_dim', embed_dim)
                    ibot_head = MODELS.build(ibot_head)
                self.ibot_head = ibot_head
            else:
                print_log(f"iBOT Head shared with DINO", logger='current')

        teacher_dict = dict(backbone=self.backbone, dino_head=self.dino_head)
        if self.ibot_separate_head: teacher_dict["ibot_head"] = self.ibot_head
        # TODO 如果冻结了backbone,这里需要改为需要梯度的参数, 避免不会更新的参数也进行的平滑
        # self.teacher = MomentumAnnealingEMA(
        #     nn.ModuleDict(teacher_dict), momentum=0.0002, gamma=100, interval=1, )
        self.teacher = CosineEMA(nn.ModuleDict(teacher_dict), momentum=0.008)

        self.teacher_temp = CosineScheduler(**teacher_temp)

        assert centering in ['centering', 'sinkhorn_knopp']
        self.centering = centering

    def forward(self,  # noqa
                # collated_global_crops: torch.Tensor, collated_local_crops: torch.Tensor,
                # collated_masks: torch.Tensor, mask_indices_list: torch.Tensor,
                # masks_weight: torch.Tensor, upperbound: int, n_masked_patches: torch.Tensor,
                mode: str = 'tensor', **data) -> Union[Dict[str, torch.Tensor], list]:
        # if mode == 'tensor':
        #     feats = self.extract_feat(inputs)
        #     return feats
        if mode == 'loss':
            return self.loss(data)  # noqa
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def loss(self, images):  # noqa
        # message_hub = MessageHub.get_current_instance()
        # max_iters = message_hub.get_info('max_iters')
        # max_epochs = message_hub.get_info('max_epochs')

        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = images['local_crop_numbers']

        global_crops = images["collated_global_crops"]
        local_crops = images["collated_local_crops"]

        masks = images["collated_masks"]
        mask_indices_list = images["mask_indices_list"]
        n_masked_patches_tensor = images["n_masked_patches"]
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"]

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino
        do_ibot = self.do_ibot

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        teacher_temp = self.teacher_temp()

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            # update teacher
            with torch.no_grad():
                # update momentum encoder
                teacher_dict = dict(backbone=self.backbone, dino_head=self.dino_head)
                if self.ibot_separate_head: teacher_dict["ibot_head"] = self.ibot_head

                self.teacher.update_parameters(nn.ModuleDict(teacher_dict))

            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher.module["backbone"](x)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens: n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = self.teacher.module["dino_head"](buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens: n_cls_tokens + n_masked_patches
                ]
            elif do_ibot and self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head = self.teacher.module["dino_head"](teacher_cls_tokens)
                masked_teacher_patch_tokens_after_head = self.teacher.module["ibot_head"](buffer_tensor_teacher)[
                    :n_masked_patches
                ]
            else:
                teacher_cls_tokens_after_head = self.teacher.module["dino_head"](teacher_cls_tokens)
                masked_teacher_ibot_softmaxed_centered = None

            if self.centering == "centering":
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

            elif self.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )

            else:
                raise NotImplementedError

            return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = get_teacher_output()

        loss_dict = {}

        # loss_accumulator = 0  # for backprop
        student_global_backbone_output_dict, student_local_backbone_output_dict = self.backbone(  # noqa
            [global_crops, local_crops], masks=[masks, None],  # is_training=True
        )

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]
        # 2: run
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.dino_head(cat_inputs))

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            # loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            # loss_accumulator += self.dino_loss_weight * dino_local_crops_loss
            loss_dict["dino_local_crops_loss"] = self.dino_loss_weight * dino_local_crops_loss

        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        if do_dino:
            # compute loss
            dino_global_crops_loss = (
                    self.dino_loss(
                        student_output_list=[student_global_cls_tokens_after_head],
                        teacher_out_softmaxed_centered_list=[
                            teacher_dino_softmaxed_centered_list.flatten(0, 1)
                        ],  # these were chunked and stacked in reverse so A is matched to B
                    )
                    * loss_scales
                    / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            # loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

            # accumulate loss
            # loss_accumulator += self.dino_loss_weight * dino_global_crops_loss
            loss_dict["dino_global_crops_loss"] = self.dino_loss_weight * dino_global_crops_loss
            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                # loss_accumulator += koleo_loss
                # loss_dict["koleo_loss"] = koleo_loss(
                #     koleo_loss / loss_scales
                # )  # this is to display the same losses as before but we can remove eventually
                loss_dict["koleo_loss"] = koleo_loss

        if do_ibot:
            # compute loss
            ibot_patch_loss = (
                    self.ibot_patch_loss.forward_masked(
                        student_global_masked_patch_tokens_after_head,
                        masked_teacher_ibot_softmaxed_centered,
                        student_masks_flat=masks,
                        n_masked_patches=n_masked_patches,
                        masks_weight=masks_weight,
                    )
                    * loss_scales
                    * ibot_loss_scale
            )

            # store for display
            # loss_dict["ibot_loss"] = ibot_patch_loss / 2

            # accumulate loss
            # loss_accumulator += self.ibot_loss_weight * ibot_patch_loss
            loss_dict["ibot_loss"] = self.ibot_loss_weight * ibot_patch_loss
            # self.backprop_loss(loss_accumulator)
        return loss_dict


@MODELS.register_module()
class DINOv2FRMsViT(FRMsVisionTransformer):
    def __init__(self, frms_config: ConfigType, **kwargs):
        super().__init__(frms_config, **kwargs)
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dims))
        self.mask_token.requires_grad_(False)

    def forward(self, x, masks=None):
        if isinstance(x, list):
            assert masks is None or isinstance(masks, list)
            output = []
            for _x, _mask in zip(x, masks):
                output.append(self._inner_forward(_x, _mask))
            return output

        return self._inner_forward(x, masks)

    def _inner_forward(self, x, masks):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + self.resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        x = self.pre_norm(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.frms(x, i, batch_first=True, has_cls_token=True, )  # TODO 最后一层的class token 没有 refine 过

        x_norm = self.ln1(x)

        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1: self.num_extra_tokens],
            "x_norm_patchtokens": x_norm[:, self.num_extra_tokens:],
            "x_prenorm": x,
            "masks": masks,
        }