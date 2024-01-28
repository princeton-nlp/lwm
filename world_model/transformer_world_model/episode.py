from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class EpisodeMetrics:
    episode_length: int
    episode_return: float


@dataclass
class Episode:
    observations: torch.ByteTensor
    actions: torch.LongTensor
    rewards: torch.FloatTensor
    dones: torch.LongTensor
    mask_padding: torch.BoolTensor
    raw_manual: List
    parsed_manual: List
    true_parsed_manual: List

    def __post_init__(self):
        assert (
            len(self.observations)
            == len(self.actions)
            == len(self.rewards)
            == len(self.dones)
            == len(self.mask_padding)
        )
        if self.dones.sum() > 0:
            idx_done = torch.argmax(self.dones) + 1
            self.observations = self.observations[:idx_done]
            self.actions = self.actions[:idx_done]
            self.rewards = self.rewards[:idx_done]
            self.dones = self.dones[:idx_done]
            self.mask_padding = self.mask_padding[:idx_done]

    def __len__(self) -> int:
        return self.observations.size(0)

    def merge(self, other: Episode) -> Episode:
        return Episode(
            torch.cat((self.observations, other.observations), dim=0),
            torch.cat((self.actions, other.actions), dim=0),
            torch.cat((self.rewards, other.rewards), dim=0),
            torch.cat((self.dones, other.dones), dim=0),
            torch.cat((self.mask_padding, other.mask_padding), dim=0),
            self.raw_manual,
            self.parsed_manual,
            self.true_parsed_manual,
        )

    def segment(self, start: int, stop: int, should_pad: bool = False) -> Episode:
        assert start < len(self) and stop > 0 and start < stop
        padding_length_right = max(0, stop - len(self))
        padding_length_left = max(0, -start)
        assert padding_length_right == padding_length_left == 0 or should_pad

        def pad(x):
            pad_right = (
                torch.nn.functional.pad(
                    x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]
                )
                if padding_length_right > 0
                else x
            )
            return (
                torch.nn.functional.pad(
                    pad_right,
                    [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0],
                )
                if padding_length_left > 0
                else pad_right
            )

        start = max(0, start)
        stop = min(len(self), stop)
        segment = Episode(
            self.observations[start:stop],
            self.actions[start:stop],
            self.rewards[start:stop],
            self.dones[start:stop],
            self.mask_padding[start:stop],
            self.raw_manual,
            self.parsed_manual,
            self.true_parsed_manual,
        )

        segment.observations = pad(segment.observations)
        segment.actions = pad(segment.actions)
        segment.rewards = pad(segment.rewards)
        segment.dones = pad(segment.dones)
        segment.mask_padding = torch.cat(
            (
                torch.zeros(padding_length_left, dtype=torch.bool),
                segment.mask_padding,
                torch.zeros(padding_length_right, dtype=torch.bool),
            ),
            dim=0,
        )

        return segment

    def compute_metrics(self) -> EpisodeMetrics:
        return EpisodeMetrics(len(self), self.rewards.sum())

    def save(self, path: Path) -> None:
        torch.save(self.__dict__, path)
