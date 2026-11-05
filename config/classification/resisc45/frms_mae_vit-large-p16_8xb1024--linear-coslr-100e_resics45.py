_base_ = [
    './adamw_100e.py',
    '../../_base_/datasets/resisc45_bs256_224.py',
]
work_dir = "work_dirs/finetune/resics45/{{fileBasenameNoExtension}}"

train_dataloader = dict(batch_size=1024)
val_dataloader = dict(batch_size=1024)
test_dataloader = dict(batch_size=1024)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='FRMsVisionTransformer',
        frms_config=dict(
            token_length=100, embed_dims=1024,
            num_layers=24, patch_size=16, # rank=64
        ), freeze_frms =True,
        arch='large', img_size=224, patch_size=16,
        frozen_stages=24, drop_path_rate=0.0,

        layer_scale_init_value=0.0, # for mae
        out_type='cls_token',
        final_norm=True,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.',checkpoint=''), # TODO replace checkpoint path
    ),
    neck=dict(type='ClsBatchNormNeck', input_features=1024),
    head=dict(
        type='VisionTransformerClsHead',
        num_classes={{_base_.data_preprocessor.num_classes}},
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss'),
        init_cfg=[dict(type='TruncNormal', layer='Linear', std=0.01)]
    )
)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper', loss_scale='dynamic', dtype='bfloat16',
    optimizer=dict(type='LARS', lr=0.1 * 2048 / 256, weight_decay=0.0, momentum=0.9),
    constructor='PEFTOptimWrapperConstructor',
)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=90,
        by_epoch=True,
        begin=10,
        end=100,
        eta_min=0.0,
        convert_to_iter_based=True)
]

auto_scale_lr = dict(base_batch_size=2048)