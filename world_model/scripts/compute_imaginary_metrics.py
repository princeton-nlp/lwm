""" This file assumes you have moved all json files of all models to a directory"""

import json
import os
from collections import defaultdict
import numpy as np

# TODO: replace this with folder that contains the json result files
current_directory = os.getcwd()

model_names = ['none','standardv2', 'direct', 'emma','oracle']

splits = ['test_ne_sr_and_sm',  'test_se_nr_or_nm', 'test_ne_nr_or_nm']

all_items = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# Loop through each file in the directory
for filename in os.listdir(current_directory):
    if filename.endswith(".json"):
        # TODO: change this to correctly math the file name with the model
        for n in model_names:
            if n + "_emma" in filename:
                break
        print(n)
        with open(filename) as f:
            data = json.load(f)
            for split in splits:
                dist_list = [abs(x[-1] - x[-2]) for x in data[split]['dist']]
                reward_list = data[split]['reward']
                done_list = data[split]['done']
                all_items['dist'][split][n].append(np.average(dist_list))
                all_items['reward'][split][n].append(np.average(reward_list))
                all_items['done'][split][n].append(np.average(done_list))


for m in all_items:
    for s in all_items[m]:
        for n in model_names:
            v = all_items[m][s][n]
            print(m, s, n, len(v), np.average(v), sep="\t")
        print()
    print()
