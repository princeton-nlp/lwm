import sys

sys.path.append("..")
import random
import json
from collections import defaultdict
from pprint import pprint

import torch
import numpy as np


from messenger.envs import make_env
from chatgpt_groundings.utils import load_gpt_groundings
import flags
import train_wm as utils

args = flags.make()
world_model = utils.make_model(args)

gpt_groundings = load_gpt_groundings(args)

env = make_env(
    args, world_model=world_model, max_episode_steps=32, gpt_groundings=gpt_groundings
)

combs = defaultdict(lambda: defaultdict(int))

dataset = utils.make_dataset(args)

with open("custom_dataset/data_splits_final_with_messenger_names.json") as f:
    splits = json.load(f)

split = "dev_ne_nr_or_nm"

# TODO: change this ID to visualize a different game
game_id = 2

game = splits[split][game_id]

# what roles and movements have this entity taken on during training?
movements = defaultdict(lambda: defaultdict(int))
roles = defaultdict(lambda: defaultdict(int))
combs = defaultdict(lambda: defaultdict(int))
for g in splits["train"]:
    for e in g:
        movements[e[0]][e[1]] += 1
        roles[e[0]][e[2]] += 1
        combs[e[0]][(e[1], e[2])] += 1

obs, info = env.reset(split=split, entities=game)

env.render()

while True:
    action = int(input("Please input next action (0: up, 1: down, 2: left, 3: right, 4: stay)\n"))
    obs, reward, done, _ = env.step(action)

    print()
    print("Role during training time:")
    for e in game:
        print(e[0], movements[e[0]], roles[e[0]], combs[e[0]])
    print()
    env.render()

    if done:
        break
