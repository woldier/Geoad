import torch

from typing import Optional, Union, List
from torch import nn

from ..utils import unfreeze_param
from mmpretrain.models.necks import MAEPretrainDecoder
from mmengine.model import BaseModel

from mmpretrain.registry import MODELS

from geoad.models.peft import Reins


@MODELS.register_module()
class MAEReinsPretrainDecoder(MAEPretrainDecoder):
    def __init__(self, num_patches: int = 196, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 1024,
                 decoder_embed_dim: int = 512, decoder_depth: int = 8, decoder_num_heads: int = 16, mlp_ratio: int = 4,
                 norm_cfg: dict = dict(type='LN', eps=1e-6), predict_feature_dim: Optional[float] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None, reins_config:dict=None, unfreeze:list=[]) -> None:
        super().__init__(num_patches, patch_size, in_chans, embed_dim, decoder_embed_dim, decoder_depth,
                         decoder_num_heads, mlp_ratio, norm_cfg, predict_feature_dim, init_cfg)

        for p in self.parameters():
            p.requires_grad = False

        unfreeze_param(self.named_parameters(), unfreeze)
        self.reins = None
        if reins_config is not None:
            self.reins = Reins(**reins_config)
        # self.train()

    def init_weights(self) -> None:
        BaseModel.init_weights(self)

    def forward(self, x: torch.Tensor,
                ids_restore: torch.Tensor) -> torch.Tensor:
        """The forward function.

        The process computes the visible patches' features vectors and the mask
        tokens to output feature vectors, which will be used for
        reconstruction.

        Args:
            x (torch.Tensor): hidden features, which is of shape
                    B x (L * mask_ratio) x C.
            ids_restore (torch.Tensor): ids to restore original image.

        Returns:
            torch.Tensor: The reconstructed feature vectors, which is of
            shape B x (num_patches) x C.
        """
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x)
            if self.reins is not None:
                x = self.reins(x, i, batch_first=True, has_cls_token=True, )
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    # def train(self, mode: bool = True):
    #     self.reins.train(mode)