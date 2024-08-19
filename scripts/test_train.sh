python -m torch.distributed.launch --nproc_per_node=4 test_train.py \
    --batch_size 16 \
    --num_workers 48 \
    --data_path /data/gsx1/datasets/SOD/DUTS \
    --sam_ckpt ckpts/sam_vit_b_01ec64.pth \
    --resume ckpts/pretrained/mdsam_512.pth \
    --img_size 512





