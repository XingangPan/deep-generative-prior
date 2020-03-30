#!/usr/bin/env sh

WORK_PATH=$(dirname $0)
IMAGE_PATH=${1:-data/ILSVRC2012_val_00008229.JPEG}
CLASS=${2:-174}

python -u -W ignore example.py \
--exp_path $WORK_PATH \
--image_path $IMAGE_PATH \
--class $CLASS \
--seed 4 \
--dgp_mode category_transfer \
--update_G \
--ftr_num 8 8 8 \
--ft_num 7 7 7 \
--lr_ratio 1 1 1 \
--w_D_loss 1 1 1 \
--w_nll 0.2 \
--w_mse 0 0 0 \
--select_num 500 \
--sample_std 0.5 \
--iterations 125 125 100 \
--G_lrs 2e-7 2e-5 2e-6 \
--z_lrs 1e-1 1e-2 2e-4 \
--use_in False False False \
--resolution 256 \
--weights_root pretrained \
--load_weights 256 \
--G_ch 96 --D_ch 96 \
--G_shared \
--hier --dim_z 120 --shared_dim 128 \
--skip_init --use_ema
