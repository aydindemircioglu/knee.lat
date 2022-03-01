_base_ = './cascade_rcnn_r50_fpn_1x_coco.py'

# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=15)

model = dict(
    type='CascadeRCNN',
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'))
    # roi_head=dict(
    #     bbox_head=dict(num_classes=1),
    #     mask_head=dict(num_classes=1)))

dataset_type = 'COCODataset'
classes = ('ASM',)

data = dict(
    samples_per_gpu = 4,
    workers_per_gpu = 4,
    train=dict(
        img_prefix='.',
        classes=classes,
        ann_file='/data/uke/data/knee.lat/asm_detect/train.json'),
    val=dict(
        img_prefix='../export/train/',
        classes=classes,
        ann_file='/data/uke/data/knee.lat/asm_detect/train.json'),
    test=dict(
        img_prefix='export/train/',
        classes=classes,
        ann_file='/data/uke/data/knee.lat/asm_detect/train.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
#load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
