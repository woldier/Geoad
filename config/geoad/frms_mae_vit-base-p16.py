_base_ = [
    '../_base_/models/mae_vit-base-p16.py',

]
model = dict(
    type='MAE',
    backbone=dict(
        type='MAEFRMsViT', arch='b', patch_size=16, mask_ratio=0.75,
        frms_config=dict(
            token_length=100, embed_dims=768,
            num_layers=12, patch_size=16,
        ),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth',
            prefix='backbone.')
    ),
    neck=dict(
        type='MAEFRMsPretrainDecoder',
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        frms_config=dict(
            token_length=100, embed_dims=512,
            num_layers=8, patch_size=16,
        ),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth',
            prefix='neck.')
    ),
    head=dict(
        type='MAEPretrainHead',
        norm_pix=True,
        patch_size=16,
        loss=dict(type='PixelReconstructionLoss', criterion='L2')),
    )