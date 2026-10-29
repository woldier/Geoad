_base_ = [
    './frms_mae_vit-base-p16_8xb512-amp-coslr-300e_million-aid.py',
]
train_dataloader = dict(
    batch_size=128,
)

model = dict(
    backbone=dict(
        arch='l',
        frms_config=dict(
            token_length=100, embed_dims=1024,
            num_layers=24, patch_size=16,
        ),
        init_cfg=dict(checkpoint='pretrained/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k_20220825-cc7e98c9.pth', )
    ),
    neck=dict(
        embed_dim=1024,
        init_cfg=dict(checkpoint='pretrained/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k_20220825-cc7e98c9.pth',)
    ),
)