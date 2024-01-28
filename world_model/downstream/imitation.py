import random
import numpy as np

import torch
import torch.nn.functional as F

from .rollout import RolloutGenerator
from .oracle import TrainOracleWithMemory
from transformer_world_model.utils import ENTITY_IDS


class ImitationTrainer:
    def __init__(self, args):
        self.rollout_generator = RolloutGenerator(args)
        self.oracle = TrainOracleWithMemory(args)
        self.batch_size = args.batch_size
        self.device = args.device
        self.policy_base_arch = args.emma_policy.base_arch

    def train(self, policy, env, optimizer, dataset=None, game=None, split=None):
        policy.train()

        assert dataset is None or game is None

        if dataset is not None:
            split = "train"
            batch = dataset["train"].sample_batch(self.batch_size, 1)
            games = batch["true_parsed_manual"]

        if game is not None:
            games = [game] * self.batch_size

        # generate trajectories
        memory = self.rollout_generator.generate(policy, env, split, games, 1)

        # query oracle for reference actions
        ref_actions = []
        manual_iter = iter(memory.manuals)
        for i, obs in enumerate(memory.observations):
            if i == 0 or memory.is_terminals[i - 1]:
                manual = next(manual_iter)
            ref_actions.append(self.oracle.act(obs, manual, "go_to_goal"))

        if self.policy_base_arch == "transformer":
            states, actions, ref_actions, texts, masks = self._make_sequences(
                policy, memory, ref_actions
            )
            logprobs, _, _ = policy.evaluate(states, actions, ref_actions, texts, masks)
        else:
            ref_actions = torch.tensor(ref_actions).to(self.device)
            states = torch.stack(memory.states).detach()
            texts = torch.stack(memory.texts).detach()
            logprobs, _, _ = policy.evaluate(states, ref_actions, texts)

        loss = -logprobs.mean()

        policy.train()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}

    def _make_sequences(self, policy, memory, ref_actions):
        # break into sequences
        seqs = {"obs": [], "action": [], "ref_action": [], "text": [], "mask": []}
        for i in range(len(memory.is_terminals)):
            if i == 0 or memory.is_terminals[i - 1]:
                for k in seqs:
                    if k == "text":
                        seqs[k].append(memory.texts[i])
                    else:
                        seqs[k].append([])
            seqs["obs"][-1].append(memory.observations[i])
            seqs["action"][-1].append(memory.actions[i])
            seqs["ref_action"][-1].append(ref_actions[i])
            seqs["mask"][-1].append(1)

        for i in range(len(seqs["action"])):
            seqs["action"][i] = [policy.STAY_ACTION] + seqs["action"][i][:-1]

        # sample a segment in each episode
        for i in range(len(seqs["obs"])):
            start = random.randint(0, len(seqs["obs"][i]) - 1)
            end = start + policy.decoder.config.max_blocks
            for k in seqs:
                if k != "text":
                    seqs[k][i] = seqs[k][i][start:end]

        # find max len
        max_len = max([len(seq) for seq in seqs["obs"]])
        # pad
        for i, (sq, aq, raq, mq) in enumerate(
            zip(seqs["obs"], seqs["action"], seqs["ref_action"], seqs["mask"])
        ):
            while len(sq) < max_len:
                sq.append(np.zeros_like(sq[-1]))
                aq.append(policy.STAY_ACTION)
                raq.append(policy.STAY_ACTION)
                mq.append(0)

        assert len(seqs["obs"]) == len(seqs["ref_action"])

        states = torch.tensor(np.array(seqs["obs"])).to(self.device).long()
        actions = torch.tensor(seqs["action"]).to(self.device).long()
        ref_actions = torch.tensor(seqs["ref_action"]).to(self.device).long()
        texts = torch.stack(seqs["text"]).detach()
        masks = torch.tensor(seqs["mask"]).to(self.device).bool()

        """
        print(states.shape)
        id = 0
        for i in range(states.shape[1]):
            print(ENTITY_IDS)
            print(memory.manuals[0])
            print("action", actions[id, i])
            print(states[id, i].sum(-1))
            print("ref action", ref_actions[id, i])
            input()
        """

        return states, actions, ref_actions, texts, masks
