import math
from typing import Optional

import torch

from mmengine.model import BaseModule
from torch import Tensor


from geoad.utils.typing_utils import OptMultiConfig
from mmengine.utils import to_2tuple

class SinePositionalEncoding(BaseModule):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 num_feats: int,
                 temperature: int = 10000,
                 normalize: bool = False,
                 scale: float = 2 * math.pi,
                 eps: float = 1e-6,
                 offset: float = 0.,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                                                    'scale should be provided and in float or int type, ' \
                                                    f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: Tensor, input: Optional[Tensor] = None) -> Tensor:
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
            input (Tensor, optional): Input image/feature Tensor.
                Shape [bs, c, h, w]

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        assert not (mask is None and input is None)

        if mask is not None:
            B, H, W = mask.size()
            device = mask.device
            # For convenience of exporting to ONNX,
            # it's required to convert
            # `masks` from bool to int.
            mask = mask.to(torch.int)
            not_mask = 1 - mask  # logical_not
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
        else:
            # single image or batch image with no padding
            B, _, H, W = input.shape
            device = input.device
            x_embed = torch.arange(
                1, W + 1, dtype=torch.float32, device=device)
            x_embed = x_embed.view(1, 1, -1).repeat(B, H, 1)
            y_embed = torch.arange(
                1, H + 1, dtype=torch.float32, device=device)
            y_embed = y_embed.view(1, -1, 1).repeat(B, 1, W)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self) -> str:
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str


class RotaryEmbeddingFast(BaseModule):
    """Implements 2D rotary embedding (RoPE) for image tokens. Position
    encoding is implemented with sin and cos functions,

        .. math::
            Pos_{cos} = cos(\frac{t}{\theta^{\frac{2i}{d}}} \\
            Pos_{sin} = sin(\frac{t}{\theta^{\frac{2i}{d}}}
    Args:
        embed_dims (int): The feature dimension for each head.
        patch_resolution (int | tuple): The resolution of the
            image, in format (H, W).
        theta (float): The hyperparameter for position coding.
            Defaults to 10000.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 patch_resolution,
                 theta=10000.,
                 init_cfg=None):
        super(RotaryEmbeddingFast, self).__init__(init_cfg=init_cfg)

        self.half_dim = embed_dims // 2
        self.patch_resolution = to_2tuple(patch_resolution)
        self.theta = theta

        freqs_cos, freqs_sin = self.compute_position_embedding()
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)

    def compute_position_embedding(self):
        frequency = self.theta**(
            torch.arange(0, self.half_dim, 2).float() / self.half_dim)
        frequency = 1. / frequency

        h, w = self.patch_resolution
        th = torch.arange(h) / h * self.half_dim
        tw = torch.arange(w) / w * self.half_dim

        position_h = (th[:, None] @ frequency[None, :]).repeat(1, 2)
        position_w = (tw[:, None] @ frequency[None, :]).repeat(1, 2)

        height = position_h[:, None, :].expand(h, w, self.half_dim)
        width = position_w[None, :, :].expand(h, w, self.half_dim)
        position = torch.cat((height, width), dim=-1)

        freqs_cos = position.cos().view(-1, position.shape[-1])
        freqs_sin = position.sin().view(-1, position.shape[-1])

        return freqs_cos, freqs_sin

    def forward(self, x, patch_resolution):
        # Check whether the patch resolution is the predefined size
        patch_resolution = to_2tuple(patch_resolution)
        if patch_resolution != self.patch_resolution:
            self.patch_resolution = patch_resolution
            freqs_cos, freqs_sin = self.compute_position_embedding()
            self.register_buffer('freqs_cos', freqs_cos.to(x.device))
            self.register_buffer('freqs_sin', freqs_sin.to(x.device))

        batch, num_heads, num_patches, dim = x.shape

        inputs = x
        x = x.reshape(batch, num_heads, num_patches, -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        x = x.reshape(batch, num_heads, num_patches, dim)

        return inputs * self.freqs_cos + x * self.freqs_sin