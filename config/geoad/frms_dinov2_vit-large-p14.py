model = dict(
    type='DINOv2',
    backbone=dict(
        type='DINOv2FRMsViT', arch='l', patch_size=14, layer_scale_init_value=1e-5,
        frms_config=dict(
            token_length=100, embed_dims=1024,
            num_layers=24, patch_size=14,
        ),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/vit-large-p14_dinov2-pre_3rdparty_20230426-f3302d9e.pth',
            prefix='backbone.')
    ),
    dino_head = dict(
        in_dim = 1024,
        out_dim = 131072, # cfg.dino.head_n_prototypes,
        hidden_dim = 2048,  # cfg.dino.head_hidden_dim,
        bottleneck_dim =  384, # cfg.dino.head_bottleneck_dim,
        nlayers = 3, #  cfg.dino.head_nlayers,
        ),
    ibot_head = dict(
        in_dim = 1024,  #embed_dim,
        out_dim = 131072, #cfg.ibot.head_n_prototypes,
        hidden_dim = 2048, #cfg.ibot.head_hidden_dim,
        bottleneck_dim = 256, # cfg.ibot.head_bottleneck_dim,
        nlayers = 3, # cfg.ibot.head_nlayers,
        ),
    dino_loss_weight=1.0, koleo_loss_weight=0.1,
    ibot_loss_weight=1.0, ibot_separate_head= True,
    teacher_temp = dict(
        base_value=0.07     ,# cfg.teacher["teacher_temp"],
        final_value=0.07    ,# cfg.teacher["teacher_temp"],
        # inner support total_iters=        ,# cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=30     ,# cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0.04,# cfg.teacher["warmup_teacher_temp"],
    ),
    centering = 'sinkhorn_knopp'
)