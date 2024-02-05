""" adapted from https://github.com/eloialonso/iris """

from typing import Any, Optional, Tuple, List
from collections import OrderedDict
from copy import deepcopy as dc
import random
import sys

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import Batch
from .kv_caching import KeysValues
from .slicer import Embedder, Head
from .transformer import Transformer, TransformerConfig, ExternalAttention
from .utils import ENTITY_IDS, process_parsed_manual_for_encoding
from .encoder import *
from .decoder import *


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_dones: torch.FloatTensor


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


class WorldModel(nn.Module):
    def __init__(
        self,
        args,
        encoder_config: TransformerConfig,
        decoder_config: TransformerConfig,
    ):
        super().__init__()
        self.device = encoder_config.device
        self.random = random.Random(args.seed + 968)
        self.manual_type = args.manual
        self.shuffle_manual = args.shuffle_manual

        if self.manual_type == "none":
            self.encoder = RawTextEncoder(encoder_config)
            self.decoder = EmmaDecoder(decoder_config)

        elif self.manual_type == "standard":
            self.encoder = StandardEncoder(encoder_config)
            self.decoder = StandardDecoder(decoder_config)

        elif self.manual_type == "standardv2":
            self.encoder = StandardEncoderV2(encoder_config)
            self.decoder = StandardDecoder(decoder_config)

        elif self.manual_type == "emma":
            self.encoder = RawTextEncoder(encoder_config)
            self.decoder = EmmaDecoder(decoder_config)

        elif self.manual_type == "direct":
            self.encoder = RawTextEncoder(encoder_config)
            self.decoder = TrueGroundDecoder(decoder_config)

        elif self.manual_type == "oracle":
            self.encoder = ParsedFeatureEncoder(encoder_config)
            self.decoder = TrueGroundDecoder(decoder_config)

        else:
            print("WorldModelError: Manual type not supported!")
            sys.exit(1)

        if hasattr(self.encoder, "embedder"):
            self.encoder.embedder.embedding_tables[
                "manual"
            ] = self.decoder.embedder.embedding_tables["observation"]

        # initialize parameters
        if args.special_init:
            self.apply(init_weights)

    def __repr__(self) -> str:
        return f"World Model:\n  Encoder: {self.encoder}\n  Decoder: {self.decoder}"

    def reset(self, n: int) -> KeysValues:
        return self.decoder.reset(n)

    def _reorder_raw_manuals(self, manuals: Tuple, observations: torch.Tensor) -> List:
        raw_manuals, parsed_manuals = manuals
        new_raw_manuals = []
        for raw_manual, parsed_manual, observation in zip(
            raw_manuals, parsed_manuals, observations
        ):
            ids = observation.flatten(0, 1).max(0)[0][:-1].tolist()
            new_raw_manual = []
            for id in ids:
                pos = None
                for i, e in enumerate(parsed_manual):
                    e_id = ENTITY_IDS[e[0]]
                    if e_id == id:
                        pos = i
                        break
                if pos is not None:
                    new_raw_manual.append(raw_manual[pos])
                else:
                    new_raw_manual.append(self.random.choice(raw_manual))
            assert len(new_raw_manual) == 3
            new_raw_manuals.append(new_raw_manual)
        return new_raw_manuals

    def _reorder_parsed_manual(self, manual: List, observation: torch.Tensor) -> List:
        ids = observation.flatten(0, 1).max(0)[0][:-1].tolist()
        new_manual = []
        for id in ids:
            found_e = None
            for e in manual:
                e_id = ENTITY_IDS[e[0]]
                if e_id == id:
                    found_e = e
                    break
            if found_e is not None:
                new_manual.append(found_e)
            else:
                new_manual.append(self.random.choice(manual))
        assert len(new_manual) == 3
        return new_manual

    def _process_parsed_manuals(
        self, manuals: List, observations: torch.Tensor
    ) -> torch.Tensor:
        out = []
        for m, o in zip(manuals, observations):
            # IMPORTANT: reorder entities in manual to match with observation channels
            m = self._reorder_parsed_manual(m, o)
            m = process_parsed_manual_for_encoding(m)
            x = []
            for i, e in enumerate(m):
                e = self.encoder.tokenizer["manual"].words_to_ids(e)
                x.append(e)
            out.append(x)

        out = torch.tensor(out).to(
            self.encoder.device
        )

        return out

    def get_manuals(self, batch: Batch) -> List:
        if self.manual_type == "none":
            return batch["raw_manual"]
        elif self.manual_type == "standard" or self.manual_type == "standardv2":
            return batch["raw_manual"]
        elif self.manual_type == "emma":
            return batch["raw_manual"]
        elif self.manual_type == "direct":
            return batch["raw_manual"], batch["parsed_manual"]
        elif self.manual_type == "oracle":
            return batch["true_parsed_manual"]

    def _shuffle_manuals(self, manuals):
        manuals = dc(manuals)
        for m in manuals:
            self.random.shuffle(m)
        return manuals

    def encode(self, manuals: List, observations: torch.Tensor) -> torch.Tensor:
        if self.shuffle_manual:
            if self.manual_type == "direct":
                manuals = self._shuffle_manuals(manuals[0]), self._shuffle_manuals(
                    manuals[1]
                )
            else:
                manuals = self._shuffle_manuals(manuals)

        if self.manual_type == "none":
            encoded_manuals = torch.zeros_like(self.encoder(manuals))

        elif self.manual_type == "standard" or self.manual_type == "standardv2":
            encoded_manuals = self.encoder(manuals)

        elif self.manual_type == "emma":
            encoded_manuals = self.encoder(manuals)

        elif self.manual_type == "direct":
            encoded_manuals = self.encoder(
                self._reorder_raw_manuals(manuals, observations)
            )

        elif self.manual_type == "oracle":
            encoded_manuals = self.encoder(
                self._process_parsed_manuals(manuals, observations)
            )

        return encoded_manuals

    def decode(self, *args, **kwargs) -> WorldModelOutput:
        return self.decoder(*args, **kwargs)

    def compute_loss(self, batch: Batch, is_eval: bool = False):
        # encode manuals
        encoded_manuals = self.encode(
            self.get_manuals(batch), batch["observations"][:, 0]
        )

        # decode trajectories
        act_tokens = rearrange(batch["actions"], "b l -> b l 1")
        obs_tokens = self.tokenize_observations(batch["observations"])
        reward_tokens = rearrange(
            self.reward_to_label(batch["rewards"]), "b l -> b l 1"
        )
        done_tokens = rearrange(batch["dones"], "b l -> b l 1")
        tokens = rearrange(
            torch.cat(
                (act_tokens, obs_tokens, reward_tokens, done_tokens),
                dim=2,
            ),
            "b l k -> b (l k)",
        )
        outputs = self.decode(tokens, encoded_manuals)

        # compute labels
        labels = {}
        (
            labels["observation"],
            labels["reward"],
            labels["done"],
        ) = self._compute_labels(
            obs_tokens, reward_tokens, done_tokens, batch["mask_padding"]
        )

        # omit first time step in logits
        logits = {}
        logits["observation"] = outputs.logits_observations[
            :, self.decoder.tokens_per_observation :
        ]
        logits["reward"] = outputs.logits_rewards[:, 1:]
        logits["done"] = outputs.logits_dones[:, 1:]

        # compute loss
        loss = {}
        for k in logits:
            loss[k] = F.cross_entropy(logits[k].flatten(0, 1), labels[k].flatten(0, 1))

        return (
            LossWithIntermediateLosses(
                loss_obs=loss["observation"],
                loss_reward=loss["reward"],
                loss_done=loss["done"],
            ),
            logits,
            labels,
        )

    def tokenize_observations(self, observations: torch.tensor) -> torch.tensor:
        return self.decoder.tokenizer["observation"].encode(observations)

    def detokenize_observations(self, tokens: torch.tensor) -> torch.tensor:
        return self.decoder.tokenizer["observation"].decode(tokens)

    def reward_to_label(self, reward: torch.Tensor) -> torch.Tensor:
        return ((reward + 1) * 2).long()

    def label_to_reward(self, label: torch.Tensor) -> torch.Tensor:
        return (label.float() / 2) - 1

    def _compute_labels(
        self,
        obs_tokens: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        mask_padding: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(dones.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = rearrange(
            obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100)[
                :, 1:
            ],
            "b l k -> b (l k)",
        )
        labels_rewards = rewards.squeeze(-1).masked_fill(mask_fill, -100).long()[:, 1:]
        labels_dones = dones.squeeze(-1).masked_fill(mask_fill, -100)[:, 1:]
        return labels_observations, labels_rewards, labels_dones
