_base_ = [
    '../../_base_/default_runtime.py'
]

optim_wrapper = dict(
    type='AmpOptimWrapper', loss_scale='dynamic',
    # https://massedcompute.com/faq-answers/?question=What%20are%20the%20supported%20NVIDIA%20GPUs%20for%20BF16
    dtype='bfloat16',  # A100 support bf16. V100 not support, in this case pls use `float16`
    optimizer=dict(
        type='AdamW', lr=2e-3 * 1024 / 256, weight_decay=0.05, betas=(0.9, 0.999)),
    constructor='LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        layer_decay_rate=0.65, # for LearningRateDecayOptimWrapperConstructor use
        custom_keys={ ## custom_keys for ViTs
            '.ln': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
        }),
)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=100 - 5 ,
        by_epoch=True,
        begin=5,
        end=100,
        eta_min=0,
        convert_to_iter_based=True)
]
# runtime settings
# save checkpoint per 50 epochs.
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=50, max_keep_ckpts=3))

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
# use auto_scale_lr = dict(base_batch_size=1024,enable=True)
# or `tools/train.py config --auto-scale-lr` to enable it.
auto_scale_lr = dict(base_batch_size=1024)

randomness = dict(seed=0, diff_rank_seed=True)

