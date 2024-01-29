#!/bin/bash

manual=$1
seed=9434
exp_name=wm_${manual}_seed_${seed}
model_path=experiments/${exp_name}/dev_ne_nr_or_nm_best_total_loss.ckpt

python evaluate_wm_imagine.py --exp imagine_eval_${manual} --manual ${direct} --wm_weights_path ${model_path} --eval_mode 1
