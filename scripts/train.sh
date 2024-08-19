CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --master-port=8989 train.py \
    --batch_size 16 \
    --num_workers 48 \
    --lr_rate 0.0005 \
    --data_path /www/gsx/SOD/DUTS \
    --sam_ckpt ckpts/sam_vit_b_01ec64.pth \
    --img_size 384





