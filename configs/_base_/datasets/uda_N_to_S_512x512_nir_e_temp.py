# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
dataset_type = 'Flair_Dataset'
data_root = '/data/datasets/FLAIR2/UDA/target/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
source_train_pipeline = [
    dict(type='Resize', img_scale=(1024, 512), temporal=True, img_scale_t=(80,40)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='AddNIR_E'),
    dict(type='Normalize_Values',temporal = True, nir_e = True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'sp_image', 'dates']),
]
target_train_pipeline = [
    dict(type='Resize', img_scale=(1024, 512), temporal=True, img_scale_t=(80,40)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='AddNIR_E'),
    dict(type='Normalize_Values',temporal = True, nir_e = True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg','sp_image', 'dates']),
]
test_pipeline = [
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, temporal=True, img_scale_t=(80,40)),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='AddNIR_E'),
            dict(type='Normalize_Values',temporal = True, nir_e = True),
            dict(type='ImageToTensor', keys=['img','sp_image','dates']),
            dict(type='Collect', keys=['img', 'sp_image','dates']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='Flair_Dataset',
            data_root='/data/datasets/FLAIR2/',
            img_dir='flair_aerial_train',
            img_sp_dir='flair_sen_train',
            ann_dir='flair_labels_train',
            list_sp_coords='flair-2_centroids_sp_to_patch.json',
            dataset_domains=["D081", "D032", "D031", "D046", "D034"],
            dataset_name="South",
            pipeline=source_train_pipeline),
        target=dict(
            type='Flair_Dataset',
            data_root='/data/datasets/FLAIR2/',
            img_dir='flair_aerial_train',
            img_sp_dir='flair_sen_train',
            ann_dir='flair_labels_train',
            list_sp_coords='flair-2_centroids_sp_to_patch.json',
            dataset_domains=["D080_","D060_", "D008_", "D051", "D077"],
            dataset_name="North", 
            pipeline=target_train_pipeline)),
    val=dict(
        type='Flair_Dataset',
        data_root='/data/datasets/FLAIR2/',
        img_dir='flair_aerial_train',
        img_sp_dir='flair_sen_train',
        ann_dir='flair_labels_train',
        list_sp_coords='flair-2_centroids_sp_to_patch.json',
        dataset_domains=["D080_","D060_", "D008_", "D051", "D077"],
        dataset_name="North",
        pipeline=test_pipeline),
    test=dict(
        type='Flair_Dataset',
        data_root='/data/datasets/FLAIR2/',
        img_dir='flair_aerial_train',
        img_sp_dir='flair_sen_train',
        ann_dir='flair_labels_train',
        list_sp_coords='flair-2_centroids_sp_to_patch.json',
        dataset_domains=["D080_","D060_", "D008_", "D051", "D077"],
        dataset_name="North",
        pipeline=test_pipeline))
