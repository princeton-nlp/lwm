"""
Implements wrappers on top of the basic messenger environments
"""
import random

import numpy as np
import gym

from messenger.envs.base import MessengerEnv
from messenger.envs.stage_one import StageOne
from messenger.envs.stage_two import StageTwo
from messenger.envs.stage_three import StageThree


class TwoEnvWrapper(MessengerEnv):
    """
    Switches between two Messenger environments
    """

    def __init__(
        self, stage: int, split_1: str, split_2: str, prob_env_1=0.5, **kwargs
    ):
        super().__init__()
        if stage == 1:
            self.env_1 = StageOne(split=split_1, **kwargs)
            self.env_2 = StageOne(split=split_2, **kwargs)
        elif stage == 2:
            self.env_1 = StageTwo(split=split_1, **kwargs)
            self.env_2 = StageTwo(split=split_2, **kwargs)
        elif stage == 3:
            self.env_1 = StageThree(split=split_1, **kwargs)
            self.env_2 = StageThree(split=split_2, **kwargs)

        self.prob_env_1 = prob_env_1
        self.cur_env = None

    def reset(self, **kwargs):
        if random.random() < self.prob_env_1:
            self.cur_env = self.env_1
        else:
            self.cur_env = self.env_2
        return self.cur_env.reset(**kwargs)

    def step(self, action):
        return self.cur_env.step(action)


class PermuteChannelsWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self, **kwargs):
        raw_obs, info = super().reset(**kwargs)

        """
        raw_manual = info["raw_manual"]
        true_parsed_manual = info["true_parsed_manual"]
        """

        self.ordered_channel_ids = np.random.permutation(3)
        """
        info = {}
        info["raw_manual"] = [raw_manual[j] for j in self.ordered_channel_ids]
        info["true_parsed_manual"] = [
            true_parsed_manual[j] for j in self.ordered_channel_ids
        ]
        """
        raw_obs["entities"] = raw_obs["entities"][..., self.ordered_channel_ids]

        return raw_obs, info

    def step(self, action):
        raw_obs, reward, done, info = super().step(action)
        raw_obs["entities"] = raw_obs["entities"][..., self.ordered_channel_ids]
        return raw_obs, reward, done, info


class GridObsWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def _wrap_obs(self, obs):
        return np.concatenate((obs["entities"], obs["avatar"]), axis=-1)

    def reset(self, **kwargs):
        raw_obs, info = super().reset(**kwargs)
        info["raw_obs"] = raw_obs
        obs = self._wrap_obs(raw_obs)
        return obs, info

    def step(self, action):
        raw_obs, reward, done, info = super().step(action)
        info["raw_obs"] = raw_obs
        obs = self._wrap_obs(raw_obs)
        return obs, reward, done, info
