#!/usr/bin/env sh

WORK_PATH=$(dirname $0)

python -u -W ignore main.py \
--seed 0 \
--exp_path $WORK_PATH \
--root_dir /path_to_your_ImageNet/val \
--list_file scripts/imagenet_val_1k.txt \
--dgp_mode SR \
--update_G \
--ftr_num 8 8 8 8 8 \
--ft_num 2 3 4 5 6 \
--lr_ratio 1 1 1 1 1 \
--w_D_loss 1 1 1 1 1 \
--w_nll 0.02 \
--w_mse 1 1 1 1 1 \
--select_num 500 \
--sample_std 0.3 \
--iterations 200 200 200 200 200 \
--G_lrs 5e-5 5e-5 2e-5 1e-5 1e-5 \
--z_lrs 2e-3 1e-3 2e-5 1e-5 1e-5 \
--use_in False False False False False \
--resolution 128 \
--weights_root pretrained \
--load_weights 128 \
--G_ch 96 --D_ch 96 \
--G_shared \
--hier --dim_z 120 --shared_dim 128 \
--skip_init --use_ema \
2>&1 | tee $WORK_PATH/log.txt
