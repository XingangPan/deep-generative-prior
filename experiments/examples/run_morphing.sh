#!/usr/bin/env sh

WORK_PATH=$(dirname $0)
IMAGE_PATH=${1:-data/ILSVRC2012_val_00001172.JPEG}
CLASS=${2:-269}
IMAGE_PATH2=${3:-data/ILSVRC2012_val_00020814.JPEG}
CLASS2=${4:-185}
#IMAGE_PATH=data/ILSVRC2012_val_00044065.JPEG
#CLASS=425
#IMAGE_PATH2=data/ILSVRC2012_val_00044640.JPEG
#CLASS2=425

python -u -W ignore example.py \
--exp_path $WORK_PATH \
--seed 0 \
--image_path $IMAGE_PATH \
--class $CLASS \
--image_path2 $IMAGE_PATH2 \
--class2 $CLASS2 \
--dgp_mode morphing \
--update_G \
--update_embed \
--ftr_num 8 8 8 \
--ft_num 8 8 8 \
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
--load_weights ch64_256 \
--G_ch 64 --D_ch 64 \
--G_shared \
--hier --dim_z 120 --shared_dim 128 \
--skip_init --use_ema
