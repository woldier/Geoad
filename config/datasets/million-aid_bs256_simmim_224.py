# dataset settings
dataset_type = 'CustomDataset'
# 可用环境变量替换配置， 例如为了加快读取将数据集全部放入内存中
# 此时只需要给出 DATASET=/dev/shm/million-aid/
data_root = '{{$DATASET:data/million-aid/}}'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, crop_ratio_range=(0.67, 1.0)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='SimMIMMaskGenerator',
        input_size=224,
        mask_patch_size=16,
        model_patch_size=16,
        mask_ratio=0.6),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        with_label=False,
        pipeline=train_pipeline))
