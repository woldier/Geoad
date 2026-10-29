
import numpy as np

from mmengine.model import ModuleList
from mmpretrain.models.backbones.vision_transformer import VisionTransformer, TransformerEncoderLayer
from mmpretrain.registry import MODELS
from geoad.models.utils.attention import MultiheadAttention as QKVMultiheadAttention


class TransformerEncoderLayerWithQKV(TransformerEncoderLayer):
    """Implements one encoder layer in PEFT Transformer. support q k v"""

    def __init__(self, embed_dims, num_heads, feedforward_channels, layer_scale_init_value=0., drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., num_fcs=2, qkv_bias=True, ffn_type='origin',
                 act_cfg=dict(type='GELU'), norm_cfg=dict(type='LN'), init_cfg=None):
        super().__init__(embed_dims, num_heads, feedforward_channels, layer_scale_init_value, drop_rate, attn_drop_rate,
                         drop_path_rate, num_fcs, qkv_bias, ffn_type, act_cfg, norm_cfg, init_cfg)
        del self.attn

        # self.attn = MultiheadAttention(
        #     embed_dims=embed_dims,
        #     num_heads=num_heads,
        #     attn_drop=attn_drop_rate,
        #     proj_drop=drop_rate,
        #     dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
        #     qkv_bias=qkv_bias,
        #     layer_scale_init_value=layer_scale_init_value)
        self.attn = QKVMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias,
            layer_scale_init_value=layer_scale_init_value
        ) # support q k v split

@MODELS.register_module()
class VisionTransformerWithQKV(VisionTransformer):
    """Implements one Vision Transformer with split q k v"""

    def __init__(self, arch='base', img_size=224, patch_size=16, in_channels=3, out_indices=-1, drop_rate=0.,
                 drop_path_rate=0., qkv_bias=True, norm_cfg=dict(type='LN', eps=1e-6), final_norm=True,
                 out_type='cls_token', with_cls_token=True, frozen_stages=-1, interpolate_mode='bicubic',
                 layer_scale_init_value=0., patch_cfg=dict(), layer_cfgs=dict(), pre_norm=False, init_cfg=None):
        super().__init__(arch, img_size, patch_size, in_channels, out_indices, drop_rate, drop_path_rate, qkv_bias,
                         norm_cfg, final_norm, out_type, with_cls_token, frozen_stages, interpolate_mode,
                         layer_scale_init_value, patch_cfg, layer_cfgs, pre_norm, init_cfg)
        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                layer_scale_init_value=layer_scale_init_value,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayerWithQKV(**_layer_cfg))