_base_ = [
    './adamw_100e.py',
    './resisc45_bs128_224.py'
]
work_dir = "work_dirs/finetune/resics45/{{fileBasenameNoExtension}}"
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='FRMsVisionTransformer',
        frms_config=dict(
            token_length=100, embed_dims=1024,
            num_layers=24, patch_size=16, # rank=64
        ),
        arch='large', img_size=224, patch_size=16,
        frozen_stages=23, drop_path_rate=0.0,

        layer_scale_init_value=0.1, # for dinov2
        out_type='avg_featmap',
        final_norm=True,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.',checkpoint=''), # TODO replace checkpoint path
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes={{_base_.data_preprocessor.num_classes}},
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))

optim_wrapper = dict(
    constructor='PEFTOptimWrapperConstructor',
    paramwise_cfg=dict(
        _delete_=True,
        # layer_decay_rate=0.65, # for LearningRateDecayOptimWrapperConstructor use
        custom_keys={
            '.ln': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            "learnable_tokens": dict(decay_mult=0.),
            "frms.scale": dict(decay_mult=0.),
        }),
)

