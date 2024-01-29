#!/bin/bash

data_file=custom_dataset/wm_data_mixed_100k_train.pickle
data_size=100000

if [ -f ${data_file} ]; then
  echo "Data file already exists!"
  exit 0
fi

python generate_wm_data.py --data_gen.behavior_policy mixed --data_gen.save_path ${data_file} --data_gen.num_train ${data_size}
