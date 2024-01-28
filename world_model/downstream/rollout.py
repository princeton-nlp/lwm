import numpy as np
import torch

from messenger.models.utils import Encoder
from transformers import AutoModel, AutoTokenizer

from .oracle import TrainOracleWithMemory
from transformer_world_model.utils import ENTITY_IDS


class RolloutGenerator:
    def __init__(self, args):
        bert_model = AutoModel.from_pretrained("bert-base-uncased")
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_encoder = Encoder(
            model=bert_model,
            tokenizer=bert_tokenizer,
            device=args.device,
            max_length=36,
        )
        self.buffer = ObservationBuffer(
            device=args.device, buffer_size=args.emma_policy.hist_len
        )
        self.memory = Memory()

    @torch.no_grad()
    def generate(self, policy, env, split, games, num_episodes):
        self.memory.clear_memory()

        for game in games:
            for _ in range(num_episodes):
                # reset policy
                policy.reset(1)
                # reset environment
                obs, info = env.reset(split=split, entities=game)
                # reset observation buffer
                self.buffer.reset(obs)

                true_parsed_manual = info["true_parsed_manual"]
                self.memory.manuals.append(true_parsed_manual)
                encoded_manual, _ = self.text_encoder.encode(info["raw_manual"])

                done = False
                # episode loop
                while not done:
                    self.memory.observations.append(obs)
                    if isinstance(policy, TrainOracleWithMemory):
                        action = policy.act(
                            obs,
                            true_parsed_manual,
                            self.buffer.get_obs(),
                            encoded_manual,
                            self.memory,
                        )
                    else:
                        action = policy.act(
                            self.buffer.get_obs(), encoded_manual, self.memory
                        )

                    obs, reward, done, info = env.step(action)

                    self.buffer.update(obs)
                    self.memory.rewards.append(reward)
                    self.memory.is_terminals.append(done)

        return self.memory


class Memory:
    """Class to store information used by the PPO class"""

    def __init__(self):
        self.manuals = []
        self.observations = []
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.texts = []

    def clear_memory(self):
        self.manuals = []
        self.observations = []
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.texts = []


class ObservationBuffer:
    """
    Maintains a buffer of observations along the 0-dim. Observations
    are currently expected to be a dict of np arrays. Currently keeps
    observations in a list and then stacks them via torch.stack().
    TODO: pre-allocate memory for faster calls to get_obs().

    Parameters:
    buffer_size
        How many previous observations to track in the buffer
    device
        The device on which buffers are loaded into
    """

    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.buffer = None
        self.device = device

    def _np_to_tensor(self, obs):
        return torch.from_numpy(obs).long().to(self.device)

    def reset(self, obs):
        # initialize / reset the buffer with the observation
        self.buffer = [obs for _ in range(self.buffer_size)]

    def update(self, obs):
        # update the buffer by appending newest observation
        assert self.buffer, "Please initialize buffer first with reset()"
        del self.buffer[0]  # delete the oldest entry
        self.buffer.append(obs)  # append the newest observation

    def get_obs(self):
        if isinstance(self.buffer[0], dict):
            # get a stack of all observations currently in the buffer
            stacked_obs = {}
            for key in self.buffer[0].keys():
                stacked_obs[key] = torch.stack(
                    [self._np_to_tensor(obs[key]) for obs in self.buffer]
                )
        else:
            assert isinstance(self.buffer[0], np.ndarray)
            stacked_obs = torch.stack([self._np_to_tensor(obs) for obs in self.buffer])

        return stacked_obs
