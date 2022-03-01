python3 ./train.py configs/cascade_rcnn_x101_64x4d_fpn_1x_coco.py --cfg-options data.samples_per_gpu=2 \
    --gpu-ids 0 --seed 42 --deterministic --work-dir /data/uke/data/knee.lat/asm_detect/checkpoints/cascade
