from typing import List, Tuple
from pathlib import Path
import random

from vgdl.interfaces.gym import VGDLEnv
from torch.distributions import Categorical
import torch
import numpy as np

from messenger.envs.stage_two_custom import StageTwoCustom
from world_model.chatgpt_groundings.utils import parse_manuals
from world_model.transformer_world_model.utils import (
    process_parsed_manual_for_encoding,
    ENTITY_IDS,
)


class StageTwoTransformer(StageTwoCustom):
    """Stage Two Messenger environment which uses a world model
    to generate observations and rewards, instead of using the
    actual Messenger simulator.
    """

    def __init__(
        self,
        world_model,
        gpt_groundings=None,
        shuffle_obs: bool = False,
        fix_order: bool = False,
    ):
        super().__init__(shuffle_obs=shuffle_obs, fix_order=fix_order)

        self.world_model = world_model
        self.world_model.eval()

        assert self.world_model.manual_type in [
            "none",
            "emma",
            "standard",
            "standardv2",
            "direct",
            "oracle",
        ]

        self.gpt_groundings = gpt_groundings

    def _reset_world_model(self, action, obs, reward, done):
        past_keys_values = self.world_model.reset(action.shape[0])
        tokens = torch.cat(
            (action.unsqueeze(-1), obs, reward.unsqueeze(-1), done.unsqueeze(-1)),
            dim=-1,
        )
        self.world_model.decode(
            tokens, self.encoded_manual, past_keys_values=past_keys_values
        )
        return past_keys_values

    def reset(
        self,
        split: str,
        entities: List[List[str]],
        freeze_manual: bool = False,
        **kwargs
    ) -> Tuple:
        # call super reset
        obs, info = super().reset(split, entities, freeze_manual, **kwargs)

        self.raw_manual = info["raw_manual"]
        self.true_parsed_manual = info["true_parsed_manual"]
        if self.gpt_groundings:
            info["parsed_manual"] = parse_manuals(
                [self.raw_manual], self.gpt_groundings
            )[0]

        manual = self.world_model.get_manuals(info)
        if self.world_model.manual_type == "direct":
            manual = [manual[0]], [manual[1]]
        else:
            manual = [manual]

        merged_obs = np.concatenate((obs["entities"], obs["avatar"]), axis=-1)
        device = self.world_model.device
        merged_obs = torch.from_numpy(merged_obs).unsqueeze(0).to(device).long()

        # encode manual
        self.encoded_manual = self.world_model.encode(manual, merged_obs)

        self.cur_action = torch.tensor([0]).to(device)
        self.cur_obs = self.world_model.tokenize_observations(merged_obs)
        self.cur_reward = torch.tensor([0]).to(device)
        self.cur_done = torch.tensor([0]).to(device)

        # reset and decode first state
        self.past_keys_values = self._reset_world_model(
            self.cur_action, self.cur_obs, self.cur_reward, self.cur_done
        )

        return obs, info

    @torch.no_grad()
    def step(self, action: int) -> Tuple:
        """Step in the environment.

        Args:
            action (int): agent action

        Returns:
            Tuple: observation, reward, done, info
        """ """"""

        max_tokens = self.world_model.decoder.config.max_tokens
        tokens_per_block = self.world_model.decoder.config.tokens_per_block
        tokens_per_observation = self.world_model.decoder.tokens_per_observation

        if self.past_keys_values.size + tokens_per_block > max_tokens:
            self.past_keys_values = self._reset_world_model(
                self.cur_action, self.cur_obs, self.cur_reward, self.cur_done
            )

        self.cur_action = torch.tensor([action], device=self.world_model.device)

        observation_tokens = []
        token = self.cur_action.unsqueeze(-1)
        for i in range(tokens_per_block):
            output = self.world_model.decode(
                token,
                self.encoded_manual,
                past_keys_values=self.past_keys_values,
            )

            if i < tokens_per_observation:
                token = Categorical(logits=output.logits_observations).sample()
                observation_tokens.append(token)
            elif i < tokens_per_observation + 1:
                token = Categorical(logits=output.logits_rewards).sample()
                reward = token.squeeze(-1)
            elif i < tokens_per_observation + 2:
                token = Categorical(logits=output.logits_dones).sample()
                done = token.squeeze(-1)

        self.cur_obs = torch.cat(observation_tokens, dim=1)
        grid = (
            self.world_model.detokenize_observations(self.cur_obs)
            .squeeze(0)
            .cpu()
            .numpy()
        )
        obs = {"entities": grid[..., :-1], "avatar": grid[..., -1:]}

        self.cur_reward = reward
        reward = self.world_model.label_to_reward(reward).item()

        self.cur_done = done
        done = bool(done.item())

        return obs, reward, done, {}

    def render(self, mode):
        grid = self.world_model.detokenize_observations(self.cur_obs).cpu().numpy()
        print(ENTITY_IDS)
        print(self.true_parsed_manual)
        print(grid.sum(-1))
        print("Reward:", self.cur_reward.item())
        print("Done: ", self.cur_done.item())
