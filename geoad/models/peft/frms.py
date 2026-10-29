import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mmengine import print_log
from torch import Tensor
from typing import Union, List, Tuple
from functools import reduce
from operator import mul
from ..utils import unfreeze_param
from mmpretrain.registry import MODELS
from mmpretrain.models import VisionTransformer

from PeftPretrain.utils.typing_utils import ConfigType

class FRMs(nn.Module):
    """

    Args:
        num_layers (int): depth of transformer.
        embed_dims (int): embedding dimension.
        patch_size (int): the patch size.
        token_length (int): feature token lenth. default: 100.
        use_softmax (bool): use softmax or not. default: True.
        link_token_to_query (bool): use feature token to generate (decoder) query or not. default: True.
        scale_init (float): feature scaling factor init value. default: 0.001.
        zero_mlp_delta_f (bool):
    """  # noqa

    def __init__(
            self,
            num_layers: int,
            embed_dims: int,
            patch_size: int,
            token_length: int = 100,
            use_softmax: bool = True,

            link_token_to_query: bool = True,
            scale_init: float = 0.001,
            zero_mlp_delta_f: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.token_length = token_length



        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        val = math.sqrt(6.0 / float(3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims))
        nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))

        if self.zero_mlp_delta_f:
            # del self.scale
            # self.scale = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)



    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.learnable_tokens
        else:
            return self.learnable_tokens[layer]

    def forward(
            self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token: # TODO 试试 不处理cls token
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
        tokens = self.get_tokens(layer)
        delta_feat = self.forward_delta_feat(
            feats,
            tokens,
            layer,
        )
        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        if self.use_softmax:
            attn = attn * (self.embed_dims ** -0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],  # TODO 改成全部token
            self.mlp_token2feat(tokens[1:, :]),
        )
        delta_f = self.mlp_delta_f(delta_f + feats)
        return delta_f

class FRMsLoRA(FRMs):

    def __init__(self, num_layers: int, embed_dims: int, patch_size: int, token_length: int = 100,
                 use_softmax: bool = True, link_token_to_query: bool = True, scale_init: float = 0.001,
                 zero_mlp_delta_f: bool = False, rank:int = 64) -> None:
        super().__init__(num_layers, embed_dims, patch_size, token_length, use_softmax, link_token_to_query, scale_init,
                         zero_mlp_delta_f)
        for p in self.parameters():
            p.requires_grad = False
        self.rank = rank

        self.learnable_tokens_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.rank])
        )
        self.learnable_tokens_b = nn.Parameter(
            torch.empty([self.num_layers, self.rank, self.embed_dims])
        )
        # val = math.sqrt(
        #     6.0
        #     / float(
        #         3 * reduce(mul, (self.patch_size, self.patch_size), 1)
        #         + (self.embed_dims * self.lora_dim) ** 0.5
        #     )
        # )
        # nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
        # nn.init.uniform_(self.learnable_tokens_b.data, -val, val)
        nn.init.kaiming_uniform_(self.learnable_tokens_a, a=math.sqrt(5))
        nn.init.zeros_(self.learnable_tokens_b)



    def get_tokens(self, layer):
        if layer == -1:
            return self.learnable_tokens + self.learnable_tokens_a @ self.learnable_tokens_b
        else:
            return self.learnable_tokens[layer] + self.learnable_tokens_a[layer] @ self.learnable_tokens_b[layer]


@MODELS.register_module()
class FRMsVisionTransformer(VisionTransformer):
    """

    Args:
        frms_config (ConfigType): FRMs config
            Example:
            >>> frms_config=dict(
            >>>     type="FRMs",
            >>>     token_length=100,
            >>>     embed_dims=1024,
            >>>     num_layers=24,
            >>>     patch_size=16,
            >>>     link_token_to_query=True,
            >>>     ),
    """ # noqa
    def __init__(self, frms_config: ConfigType, unfreeze:list=[], freeze_rein =False, **kwargs):
        super().__init__(**kwargs)
        if self.frozen_stages == -1:
            self.frozen_stages = len(self.layers)
            self._freeze_stages()
        assert len(unfreeze) ==0 or all([not ''==i for i in unfreeze]), 'unfreeze patten have empty str!'
        unfreeze_param(self.named_parameters(), unfreeze)
        assert frms_config.embed_dims == self.patch_embed.embed_dims, \
            f"{frms_config.embed_dims=} must be equal to {self.__class__.__mro__[1].__name__}.embed_dims={self.patch_embed.embed_dims}"
        assert frms_config.num_layers == len(self.layers), \
            f"{frms_config.num_layers=} must be equal to {self.__class__.__mro__[1].__name__}.layers={len(self.layers)}"
        if hasattr(frms_config, 'rank'):
            self.frms: FRMs = FRMsLoRA(**frms_config)
            print_log(f'Using FRMsLoRA', logger='current')
        else:
            self.frms: FRMs = FRMs(**frms_config)
        if freeze_rein:
            print_log(f'freeze frms', logger='current')
            for p in self.frms.parameters():
                p.requires_grad = False
        self._register_load_state_dict_pre_hook(self._prepare_patch_embed)

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

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

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.frms(x, i, batch_first=True, has_cls_token=True, )
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        # return self.frms.return_auto(tuple(outs))
        return tuple(outs)

    # def train(self, mode=True):
    #     self.frms.train(mode)

    def _prepare_patch_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'patch_embed.projection.weight'
        if name not in state_dict.keys():
            return
        ckpt_patch_embed_weight_shape = state_dict[name].shape

        if self.patch_embed.projection.weight.shape != ckpt_patch_embed_weight_shape:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(
                f'Resize the patch_embed.projection.weight shape from {ckpt_patch_embed_weight_shape} '
                f'to {self.patch_embed.projection.weight.shape}.')
            import torch.nn.functional as F
            state_dict[name] = F.interpolate(
                state_dict[name].float(),
                size=self.patch_embed.projection.weight.shape[2:],
                mode="bicubic",
                align_corners=False,
            )