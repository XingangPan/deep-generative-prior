#!/usr/bin/env sh

WORK_PATH=$(dirname $0)
IMAGE_PATH=${1:-data/ILSVRC2012_val_00042095.JPEG}
CLASS=${2:-260}
#IMAGE_PATH=data/ILSVRC2012_val_00000525.JPEG
#CLASS=863

python -u -W ignore example.py \
--exp_path $WORK_PATH \
--image_path $IMAGE_PATH \
--class $CLASS \
--seed 0 \
--dgp_mode SR \
--update_G \
--ftr_num 8 8 8 8 8 \
--ft_num 2 3 4 5 7 \
--lr_ratio 1.0 1.0 1.0 1.0 1.0 \
--w_D_loss 1 1 1 1 1 \
--w_nll 0.02 \
--w_mse 1 1 1 1 1 \
--select_num 500 \
--sample_std 0.3 \
--iterations 200 200 200 200 200 \
--G_lrs 5e-5 5e-5 2e-5 1e-5 1e-5 \
--z_lrs 2e-3 1e-3 2e-5 1e-5 1e-5 \
--use_in False False False False False \
--resolution 256 \
--weights_root pretrained \
--load_weights 256 \
--G_ch 96 --D_ch 96 \
--G_shared \
--hier --dim_z 120 --shared_dim 128 \
--skip_init --use_ema
