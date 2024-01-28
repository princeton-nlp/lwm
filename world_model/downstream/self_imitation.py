from sortedcontainers import SortedList
import random
import numpy as np


import torch
import torch.nn.functional as F

from .rollout import RolloutGenerator, Memory
from .oracle import TrainOracleWithMemory
from .conv_emma import ConvEMMA
from .filtered_bc import FilteredBCTrainer
from transformer_world_model.utils import ENTITY_IDS


class SelfImitationTrainer(FilteredBCTrainer):
    def train(self, policy, env, optimizer, dataset=None, game=None, split=None):
        policy.train()

        self.iter += 1

        assert dataset is None or game is None

        if dataset is not None:
            split = "train"
            batch = dataset["train"].sample_batch(self.batch_size, 1)
            games = batch["true_parsed_manual"]

        if game is not None:
            games = [game] * self.batch_size

        # generate trajectories
        memory = self.rollout_generator.generate(policy, env, split, games, 1)

        # update buffer
        self.update_buffer(memory)

        # sample batch
        best_memory = Memory()
        best_ret = self.buffer[0][0]
        for ret, mem in self.buffer[: self.batch_size]:
            if abs(ret - best_ret) < 1e-6:
                best_memory.states.extend(mem.states)
                best_memory.actions.extend(mem.actions)
                best_memory.texts.extend(mem.texts)

        actions = torch.tensor(best_memory.actions).to(self.device)
        states = torch.stack(best_memory.states).detach()
        texts = torch.stack(best_memory.texts).detach()
        logprobs, _, _ = policy.evaluate(states, actions, texts)

        loss = -logprobs.mean()

        policy.train()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}
