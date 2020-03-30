#!/usr/bin/env sh

WORK_PATH=$(dirname $0)
IMAGE_PATH=${1:-data/ILSVRC2012_val_00003004.JPEG}
CLASS=${2:-693} # set CLASS=-1 if you don't know the class
#IMAGE_PATH=data/ILSVRC2012_val_00004291.JPEG
#CLASS=442

python -u -W ignore example.py \
--exp_path $WORK_PATH \
--image_path $IMAGE_PATH \
--class $CLASS \
--seed 0 \
--dgp_mode colorization \
--update_G \
--ftr_num 7 7 7 7 7 \
--ft_num 2 3 4 5 6 \
--lr_ratio 0.7 0.7 0.8 0.9 1.0 \
--w_D_loss 1 1 1 1 1 \
--w_nll 0.02 \
--w_mse 0 0 0 0 0 \
--select_num 500 \
--sample_std 0.5 \
--iterations 200 200 300 400 300 \
--G_lrs 5e-5 5e-5 5e-5 5e-5 2e-5 \
--z_lrs 2e-3 1e-3 5e-4 5e-5 2e-5 \
--use_in False False False False False \
--resolution 256 \
--weights_root pretrained \
--load_weights 256 \
--G_ch 96 --D_ch 96 \
--G_shared \
--hier --dim_z 120 --shared_dim 128 \
--skip_init --use_ema
