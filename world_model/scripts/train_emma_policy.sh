#!/bin/bash

use_wandb=0

seed=8291
exp_name=emma_policy_${seed}


if [ -d "experiments/${exp_name}" ]; then
    rm -rf experiments/${exp_name}
fi

python train_emma.py \
        --exp ${exp_name} \
        --batch_size 10 \
        --log_every 500 \
        --save_every 500 \
        --use_wandb ${use_wandb} \
        --seed ${seed} \
        --emma_policy.base_arch conv

exit 0

