import torch
import torch.nn as nn
from mmpretrain.models.utils.attention import MultiheadAttention as BaseMultiheadAttention


class MultiheadAttention(BaseMultiheadAttention):
    """
    This version split qkv to q,k,v
    """

    def __init__(self, embed_dims, num_heads, input_dims=None, attn_drop=0., proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.), qkv_bias=True, qk_scale=None, proj_bias=True,
                 v_shortcut=False, use_layer_scale=False, layer_scale_init_value=0., init_cfg=None):
        super().__init__(embed_dims, num_heads, input_dims, attn_drop, proj_drop, dropout_layer, qkv_bias, qk_scale,
                         proj_bias, v_shortcut, use_layer_scale, layer_scale_init_value, init_cfg)
        del self.qkv
        self.q = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
        self.k = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
        self.v = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
        self._register_state_dict_hooks()

    def _register_state_dict_hooks(self):
        """Register state dict hooks.

        Register state dict saving hooks to save qkv to
        the state dict. And register state dict loading hooks to handle the
        qkv to q,k,v while loading the state dict.
        """

        def _state_dict_hook(module, state_dict, prefix, local_metadata):
            qkv_list = [
                state_dict.pop(prefix + 'q.weight'),
                state_dict.pop(prefix + 'k.weight'),
                state_dict.pop(prefix + 'v.weight')]
            qkv_weight = torch.cat(qkv_list, dim=0)

            state_dict[prefix + "qkv.weight"] = qkv_weight
            if any(['q.bias' in k for k in state_dict.keys()]):
                qkv_bias_list = [
                    state_dict.pop(prefix + 'q.bias'),
                    state_dict.pop(prefix + 'k.bias'),
                    state_dict.pop(prefix + 'v.bias')]
                qkv_bias = torch.cat(qkv_bias_list, dim=0)
                state_dict[prefix + "qkv.bias"] = qkv_bias

        self._register_state_dict_hook(_state_dict_hook)

        def _load_state_dict_pre_hook(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs):
            if not any([k for k in state_dict.keys() if 'qkv' in k]):
                return
            if prefix + "qkv.weight" in state_dict:
                qkv_weight = state_dict.pop(prefix+"qkv.weight")  # pop this key,
                dim = qkv_weight.shape[0] // 3
                state_dict[prefix+'q.weight'] = qkv_weight[:dim]
                state_dict[prefix+'k.weight'] = qkv_weight[dim:2 * dim]
                state_dict[prefix+'v.weight'] = qkv_weight[2 * dim:]
            if prefix+"qkv.bias" in state_dict:
                qkv_bias = state_dict.pop(prefix+"qkv.bias")
                dim = qkv_bias.shape[0] // 3

                state_dict[prefix+'q.bias'] = qkv_bias[:dim]
                state_dict[prefix+'k.bias'] = qkv_bias[dim:2 * dim]
                state_dict[prefix+'v.bias'] = qkv_bias[2 * dim:]

        self._register_load_state_dict_pre_hook(_load_state_dict_pre_hook)

        # def _load_state_dict_post_hook(module, incompatible_keys):
        #     """Handle the incompatible keys while loading the state dict."""
        #     missing_keys = incompatible_keys.missing_keys.copy()
        #     for key in missing_keys:
        #         if any([k in key for k in ['q','k', 'v']]):
        #             incompatible_keys.missing_keys.remove(key)
        #
        #     # unexpected_keys = incompatible_keys.unexpected_keys.copy()
        #     # for key in unexpected_keys:
        #     #     pass
        # self.register_load_state_dict_post_hook(_load_state_dict_post_hook)

    def forward(self, x):
        B, N, _ = x.shape

        # =========origin===============
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dims).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]
        # =========origin===============

        q, k, v = self.q(x).unsqueeze(2), self.k(x).unsqueeze(2), self.v(x).unsqueeze(2)
        qkv = torch.cat((q, k, v), dim=2).reshape(B, N, 3, self.num_heads, self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_drop = self.attn_drop if self.training else 0.
        x = self.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
        x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


def test():
    q_w, q_b = torch.randn(512 * 3, 512), torch.randn(512 * 3)
    p_w, p_b = torch.randn(512, 512), torch.randn(512)
    model = MultiheadAttention(512, 8)
    state_dict = {
        "qkv.weight": q_w,
        "qkv.bias": q_b,
        'proj.weight': p_w,
        'proj.bias': p_b
    }
    model.load_state_dict(state_dict)

    print(model.state_dict().keys())

    assert torch.allclose(model.state_dict().pop('proj.weight'), p_w)
    assert torch.allclose(model.state_dict().pop('proj.bias'), p_b)
    assert torch.allclose(model.state_dict().pop('qkv.weight'), q_w)
    assert torch.allclose(model.state_dict().pop('qkv.bias'), q_b)

    base_model = BaseMultiheadAttention(512, 8)
    base_model.load_state_dict(state_dict)
    input = torch.randn(2, 8, 512)
    assert torch.allclose(model(input.detach()), base_model(input.detach()))
