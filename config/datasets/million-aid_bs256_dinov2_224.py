# dataset settings
dataset_type = 'CustomDataset'
# 可用环境变量替换配置， 例如为了加快读取将数据集全部放入内存中
# 此时只需要给出 DATASET=/dev/shm/million-aid/
data_root = '{{$DATASET:data/million-aid/}}'
data_preprocessor = dict(
    type='DINOV2DataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DINOv2Transform', global_crops_scale=[0.32,1.0], local_crops_scale=[0.05,0.32], local_crops_number=8,
        global_crops_size=224, local_crops_size=96, # local_crops_size=96 is for path_size 14
         )
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        with_label=False,
        pipeline=train_pipeline))
