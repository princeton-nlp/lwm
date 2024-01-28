from sortedcontainers import SortedList
import random
import numpy as np


import torch
import torch.nn.functional as F

from .rollout import RolloutGenerator, Memory
from .oracle import TrainOracleWithMemory
from .conv_emma import ConvEMMA
from transformer_world_model.utils import ENTITY_IDS


class FilteredBCTrainer:
    def __init__(self, args):
        self.rollout_generator = RolloutGenerator(args)

        if args.downstream.oracle_weights_path:
            self.oracle = ConvEMMA().to(args.device)
            self.oracle.load_state_dict(torch.load(args.downstream.oracle_weights_path))
            print(f"Loaded EMMA oracle from {args.downstream.oracle_weights_path}")
        else:
            self.oracle = TrainOracleWithMemory(args)

        self.batch_size = args.batch_size
        self.device = args.device
        self.policy_base_arch = args.emma_policy.base_arch

        self.buffer = []
        self.max_buffer_size = 1000
        self.iter = 0

        self.random = random.Random(args.seed + 9832)

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
        memory = self.rollout_generator.generate(self.oracle, env, split, games, 1)

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

    def update_buffer(self, memory):
        # add to buffer
        item = Memory()
        ret = 0
        for i in range(len(memory.states)):
            item.states.append(memory.states[i])
            item.actions.append(memory.actions[i])
            item.texts.append(memory.texts[i])
            ret += memory.rewards[i]
            if memory.is_terminals[i]:
                self.buffer.append((ret, item))
                item = Memory()
                ret = 0

        # shuffle buffer
        self.random.shuffle(self.buffer)

        # fine best return
        best_ret = max([x[0] for x in self.buffer])

        # pick top return trajectories
        new_buffer = []
        for ret, item in self.buffer:
            if abs(ret - best_ret) < 1e-6:
                new_buffer.append((ret, item))
                if len(new_buffer) >= self.max_buffer_size:
                    break

        self.buffer = new_buffer
