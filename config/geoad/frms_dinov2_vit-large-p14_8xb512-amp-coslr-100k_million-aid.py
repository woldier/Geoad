_base_ = [
    './frms_dinov2_vit-large-p14.py',
    '../_base_/datasets/million-aid_bs256_dinov2_224.py',
    '../_base_/default_runtime.py',
]

train_dataloader = dict(
    batch_size=32,
)
cfg = dict(
    # model_wrapper_cfg=dict(type='MMFullyShardedDataParallel', cpu_offload=True)
    # strategy = strategy
)
model = dict(
    # backbone=dict(unfreeze=['.*ln.*', 'pre_norm'], frozen_stages=23)
    backbone=dict(unfreeze=[], frozen_stages=24),
    teacher_temp = dict(
        warmup_iters=30 * 1000,
    ),
)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        # lr=1.5e-4 * 4096 / 256,
        lr=1e-4 * 4096 / 1024,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),  # TODO DINOv2里面使用了这个?
            # 'cls_token': dict(decay_mult=0.),  # TODO 源码中最后一层的学习率不一样
            "learnable_tokens": dict(decay_mult=0.),
            "frms.scale": dict(decay_mult=0.),

        }),
    constructor = 'PEFTOptimWrapperConstructor'
)

# learning rate scheduler
param_scheduler = [
    # dict(
    #     type='LinearLR',
    #     start_factor=0.0001,
    #     by_epoch=True,
    #     begin=0,
    #     end=40,
    #     convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=100 * 1000,
        by_epoch=False,
        begin=0,
        end=100 * 1000,
        # convert_to_iter_based=True
    )
]

# runtime settings
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
train_cfg = dict(by_epoch=False, max_iters=100 * 1000)
default_hooks = dict(
    logger=dict(type='LoggerHook', log_metric_by_epoch=False),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000, max_keep_ckpts=3))
log_processor = dict(by_epoch=False)
randomness = dict(seed=0, diff_rank_seed=True)

# auto resume
# resume = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=4096)
