#!/usr/bin/env sh

PARTITION=$1    # input your partition on lustre
WORK_PATH=$(dirname $0)

# make sure that the image list length could be divided by
# the number of threads, e.g., 1000 % 4 = 0
srun -p $PARTITION -n4 --gres=gpu:4 --ntasks-per-node=4 \
python -u -W ignore main.py \
--dist \
--port 12346 \
--seed 0 \
--exp_path $WORK_PATH \
--root_dir /path_to_your_ImageNet/val \
--list_file scripts/imagenet_val_1k.txt \
--dgp_mode inpainting \
--update_G \
--update_embed \
--ftr_num 8 8 8 8 8 \
--ft_num 6 6 6 6 6 \
--lr_ratio 1 1 1 1 1 \
--w_D_loss 1 1 1 0.1 0.1 \
--w_nll 0.02 \
--w_mse 10 10 10 100 100 \
--select_num 500 \
--sample_std 0.3 \
--iterations 200 200 200 200 200 \
--G_lrs 2e-4 2e-4 1e-4 1e-4 1e-5 \
--z_lrs 1e-3 1e-3 1e-4 1e-4 1e-5 \
--use_in True True True True True \
--resolution 128 \
--weights_root pretrained \
--load_weights 128 \
--G_ch 96 --D_ch 96 \
--G_shared \
--hier --dim_z 120 --shared_dim 128 \
--skip_init --use_ema \
2>&1 | tee $WORK_PATH/log.txt
