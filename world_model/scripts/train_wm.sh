#!/bin/bash

manual=$1
hidden_size=256
num_layers=4
num_heads=4

dataset=custom_dataset/wm_data_mixed_100k_train.pickle

use_wandb=0

seed=9434

version=wm
exp_name=${version}_${manual}_seed_${seed}


if [ -d "experiments/${exp_name}" ]; then
    rm -rf experiments/${exp_name}
fi

python train_wm.py \
        --version ${version} \
        --exp_name ${exp_name} \
        --manual ${manual} \
        --use_wandb ${use_wandb} \
        --hidden_size ${hidden_size} \
        --encoder_layers ${num_layers} \
        --decoder_layers ${num_layers} \
        --encoder_num_heads ${num_heads} \
        --decoder_num_heads ${num_heads} \
        --dataset_path ${dataset} \
        --seed ${seed}

exit 0

