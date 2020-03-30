#!/usr/bin/env sh

PARTITION=$1    # input your partition on lustre
WORK_PATH=$(dirname $0)

# make sure that the image list length could be divided by
# the number of threads, e.g., 1000 % 4 = 0
srun -p $PARTITION -n4 --gres=gpu:4 --ntasks-per-node=4 \
python -u -W ignore main.py \
--dist \
--port 12345 \
--seed 0 \
--exp_path $WORK_PATH \
--root_dir /path_to_your_ImageNet/val \
--list_file scripts/imagenet_val_1k.txt \
--dgp_mode colorization \
--update_G \
--ftr_num 7 7 7 7 7 \
--ft_num 2 3 4 5 6 \
--lr_ratio 0.7 0.7 0.8 0.9 1 \
--w_D_loss 1 1 1 1 1 \
--w_nll 0.02 \
--w_mse 0 0 0 0 0 \
--select_num 500 \
--sample_std 0.5 \
--iterations 200 200 300 400 300 \
--G_lrs 5e-5 5e-5 5e-5 5e-5 2e-5 \
--z_lrs 2e-3 1e-3 5e-4 5e-5 2e-5 \
--use_in False False False False False \
--resolution 128 \
--weights_root pretrained \
--load_weights 128 \
--G_ch 96 --D_ch 96 \
--G_shared \
--hier --dim_z 120 --shared_dim 128 \
--skip_init --use_ema \
2>&1 | tee $WORK_PATH/log.txt
