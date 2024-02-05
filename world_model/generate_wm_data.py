"""
Collect rollouts of custom games of Stage 2 using a random policy and store into a pickle with the following format:
{
    'texts': {
        entity: {
            movement: {
                role: {
                    split_name: ["a", "b", "c", ...],
                },
            },
        },
    },
    'keys': {
        'entities': list(texts.keys()),
        'movements': list(list(texts.values())[0].keys()),
        'roles': list(list(list(texts.values())[0].values())[0].keys()),
    },
    'rollouts': {
        split_name: {
            'manual_idxs': [
                [2, 3, 2],
            ],
            'ground_truth_idxs': [
                [(2, 1, 1), (11, 1, 2), (10, 2, 0)],
            ],
            'grid_sequences': [
                [grid0, grid1, grid2, ...],
            ],
            'action_sequences': [
                [0, action0-1, action1-2, ...],
            ],
            'reward_sequences': [
                [0, reward1, reward2, ...],
            ],
            'done_sequences': [
                [False, False, False, ...],
            ],
        },
    }
}
"""
import os
import sys

sys.path.append("..")

from typing import Dict, List, Tuple
from collections import namedtuple
import argparse
import math
import random
import pickle
import time
import json

import numpy as np
from tqdm import tqdm
import gym
import torch

from messenger.envs import make_env
from downstream import Oracle
from train_wm import make_model
import flags


RolloutResult = namedtuple(
    "RolloutResult",
    ["grid_seq", "act_seq", "reward_seq", "done_seq", "manual_idx", "ground_truth_idx"],
)


class DataGenerator:
    """Class for generating custom datasets.
    Current two uses cases are:
        (1) Generating a dataset of rollouts for world model training
        (2) Generating a dataset of rollouts for imitation learning
    """

    def __init__(self, args: argparse.Namespace):
        self.save_path = args.data_gen.save_path
        if not self.save_path.endswith(".pickle"):
            self.save_path = self.save_path + ".pickle"

        self.behavior_policy = args.data_gen.behavior_policy
        self.num_train = args.data_gen.num_train
        self.num_eval = args.data_gen.num_eval

        behavior_policy_weights_path = args.data_gen.behavior_policy_weights_path
        wm_weights_path = args.wm_weights_path

        # load data
        with open(args.splits_path) as f:
            self.splits = json.load(f)

        with open(args.texts_path) as f:
            self.texts = json.load(f)

        self.policy = Oracle(args)

        # setup environment
        world_model = make_model(args) if wm_weights_path is not None else None
        self.env = make_env(args, world_model=world_model)

        self.random = random.Random(args.seed + 2340)

    def generate_data(self):
        self.keys = {
            "entities": list(self.texts.keys()),  # ["robot", ..., "sword"]
            "movements": list(
                list(self.texts.values())[0].keys()
            ),  # ["chasing", ..., "fleeing"]
            "roles": list(
                list(list(self.texts.values())[0].values())[0].keys()
            ),  # ["enemy", ..., "goal"]
        }

        dataset = {
            "texts": self.texts,
            "keys": self.keys,
            "rollouts": {},
        }

        for split, games in self.splits.items():
            idxs = []
            manual_idxs = []
            ground_truth_idxs = []
            grid_sequences = []
            action_sequences = []
            reward_sequences = []
            done_sequences = []

            if split == "train":
                num_repeats = math.ceil(self.num_train / len(games))
            elif self.behavior_policy == "mixed":
                num_repeats = len(self.policy.INTENTIONS)
            elif isinstance(self.policy, Oracle):
                assert self.behavior_policy in self.policy.INTENTIONS
                num_repeats = 1
            else:
                num_repeats = math.ceil(self.num_eval / len(games))

            print(
                f"Starting to generate rollouts for split: {split}, trajectories per game {num_repeats}"
            )

            returns = []
            for i in tqdm(range(len(games))):
                for n in range(num_repeats):
                    rollout_result = self.rollout(split, games[i], n)
                    returns.append(sum(rollout_result.reward_seq))

                    manual_idxs.append(rollout_result.manual_idx)
                    ground_truth_idxs.append(rollout_result.ground_truth_idx)
                    grid_sequences.append(rollout_result.grid_seq)
                    action_sequences.append(rollout_result.act_seq)
                    reward_sequences.append(rollout_result.reward_seq)
                    done_sequences.append(rollout_result.done_seq)

            print(
                f"Average return for split {split}: {np.mean(returns):.2f} +/- {np.std(returns)/np.sqrt(num_repeats):.2f}"
            )

            dataset["rollouts"][split] = {
                "manual_idxs": manual_idxs,
                "ground_truth_idxs": ground_truth_idxs,
                "grid_sequences": grid_sequences,
                "action_sequences": action_sequences,
                "reward_sequences": reward_sequences,
                "done_sequences": done_sequences,
            }

        with open(self.save_path, "wb") as f:
            pickle.dump(dataset, f)
            print(f"Successfully saved dataset at {self.save_path}!")

    def rollout(self, split, game, n):
        # reset environment
        env = self.env
        obs, info = env.reset(split=split, entities=game)

        raw_manual = info["raw_manual"]
        true_parsed_manual = info["true_parsed_manual"]

        # these are the indices of the texts in the big texts dictionary
        manual_idx = self._find_manual_idx(raw_manual, true_parsed_manual, split)

        # these are the indices of the parsed manuals in the keys dictionary
        ground_truth_idx = self._find_ground_truth_idx(true_parsed_manual)

        grid_seq, act_seq, reward_seq, done_seq = [obs], [0], [0], [False]

        # choose an intention for the episode
        if self.behavior_policy == "mixed":
            if "train" in split:
                episode_intention = self.random.choice(self.policy.INTENTIONS)
            else:
                episode_intention = self.policy.INTENTIONS[n]
        elif self.behavior_policy == "emma":
            episode_intention = None
        else:
            episode_intention = self.behavior_policy
            assert episode_intention in self.policy.INTENTIONS

        done = False
        while not done:
            action = self.policy.act(obs, true_parsed_manual, episode_intention)
            obs, reward, done, _ = env.step(action)

            # append to sequences
            grid_seq.append(obs)
            act_seq.append(action)
            reward_seq.append(reward)
            done_seq.append(done)

        return RolloutResult(
            grid_seq, act_seq, reward_seq, done_seq, manual_idx, ground_truth_idx
        )

    def _find_manual_idx(
        self, manual: List[str], ground_truth: List[Tuple[str]], split: str
    ) -> List[int]:
        """Find the index of the manual descriptors in the texts dict.

        Args:
            manual (List[str]): list of manual descriptors
            ground_truth (List[Tuple[str]]): list of parsed descriptors
            split (str): dataset split

        Returns:
            List[int]: list of indices of the manual descriptors
        """
        manual_idx = []
        for idx in range(len(manual)):
            gt_entity = ground_truth[idx][0]
            gt_movement = ground_truth[idx][1]
            gt_role = ground_truth[idx][2]

            descriptors = self.texts[gt_entity][gt_movement][gt_role][split]
            desc_idx = descriptors.index(manual[idx])
            manual_idx.append(desc_idx)

        return manual_idx

    def _find_ground_truth_idx(
        self, ground_truth: List[Tuple[str]]
    ) -> List[Tuple[int]]:
        """Find the index of the parsed manuals in the keys dict.

        Args:
            ground_truth (List[Tuple[str]]): list of parsed descriptors

        Returns:
            List[Tuple[int]]: list of indices tuples of the parsed manuals
        """ """"""
        entities = self.keys["entities"]
        movements = self.keys["movements"]
        roles = self.keys["roles"]

        ground_truth_idx = []
        for idx in range(len(ground_truth)):
            gt_entity = ground_truth[idx][0]
            gt_movement = ground_truth[idx][1]
            gt_role = ground_truth[idx][2]

            entity_idx = entities.index(gt_entity)
            movement_idx = movements.index(gt_movement)
            role_idx = roles.index(gt_role)

            ground_truth_idx.append((entity_idx, movement_idx, role_idx))

        return ground_truth_idx


if __name__ == "__main__":
    args = flags.make()

    assert args.data_gen.save_path, "Specify a path for saving the data"

    data_generator = DataGenerator(args)
    data_generator.generate_data()
