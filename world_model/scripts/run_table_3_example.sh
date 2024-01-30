#!/bin/bash

# Example of launching slurm / cluster jobs for all downstream tasks (table 3). 

oracle_weights_name=emma_conv_9434

for task in "imitation" "filtered_bc"; do
    for ckpt_name in "half" "dev_ne_nr_or_nm_best_avg_return"; do
        if [ ${ckpt_name} = "half" ] && [ ${task} = "imitation" ]; then
            continue
        fi
        for split in "easy" "medium" "hard"; do 
            for manual in "none" "oracle" "emma" "direct" "fixed_standard_standardv2"; do
                for game in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29; do
                    # (RUN JOB, e.g. srun for slurm) train_downstream.slurm ${task} ${manual} ${split} ${game} ${ckpt_name}
                done
            done
        done
    done
done