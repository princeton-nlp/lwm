#!/bin/bash

split=dev_ne_sr_and_sm
manual=$1
exp=wm_${manual}_seed_9434

python play_wm.py --env.name transformer --exp_name fun --manual ${manual} --eval_mode 1 --wm_weights_path experiments/${exp}/${split}_best_total_loss.ckpt
