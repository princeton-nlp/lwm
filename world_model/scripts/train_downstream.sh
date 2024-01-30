#!/bin/bash

use_wandb=0

task=$1
manual=$2
split=$3
game=$4
oracle_ckpt=$5

max_batches=2000
seed=9434
oracle_weights_name=emma_conv_9434 # change to your oracle weights path!

version=downstream_${task}
exp_name=${version}_${manual}_${split}_game_${game}


if [ -d "experiments/${exp_name}" ]; then
    rm -rf experiments/${exp_name}
fi

if [ ${oracle_ckpt} = "half" ] then
    oracle_weights_ckpt=policy_2000
else
    oracle_weights_ckpt=dev_ne_nr_or_nm_best_avg_return
fi

python -u train_downstream.py \
        --version ${version} \
        --exp ${exp_name} \
        --wm_weights_path experiments/wm_${manual}_seed_${seed}/dev_ne_nr_or_nm_best_total_loss.ckpt \
        --manual ${manual} \
        --emma_policy.base_arch conv \
        --downstream.fix_split ${split} \
        --downstream.fix_game ${game} \
        --log_every 100 \
        --downstream.task ${task} \
        --batch_size 10 \
        --max_batches ${max_batches} \
        --downstream.oracle_weights_path experiments/${oracle_weights_name}/${oracle_weights_ckpt}.ckpt \
        --save_every 100 \
        --use_wandb ${use_wandb}

exit 0

