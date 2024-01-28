from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import itertools
import math
import random
import psutil

import numpy as np
import torch

from .utils import process_parsed_manual_for_encoding
from .episode import Episode

Batch = Dict[str, torch.Tensor]


def get_raw_manual(dataset, split, i):
    texts = dataset["texts"]
    entities = dataset["keys"]["entities"]
    movements = dataset["keys"]["movements"]
    roles = dataset["keys"]["roles"]
    manual_idx = dataset["rollouts"][split]["manual_idxs"][i]
    ground_truth_idx = dataset["rollouts"][split]["ground_truth_idxs"][i]
    return [
        texts[entities[ground_truth_idx[j][0]]][movements[ground_truth_idx[j][1]]][
            roles[ground_truth_idx[j][2]]
        ][split][manual_idx[j]]
        for j in range(len(manual_idx))
    ]


def get_true_parsed_manual(dataset, split, i):
    entities = dataset["keys"]["entities"]
    movements = dataset["keys"]["movements"]
    roles = dataset["keys"]["roles"]
    ground_truth_idx = dataset["rollouts"][split]["ground_truth_idxs"][i]
    return [
        [
            entities[ground_truth_idx[j][0]],
            movements[ground_truth_idx[j][1]],
            roles[ground_truth_idx[j][2]],
        ]
        for j in range(len(ground_truth_idx))
    ]


class EpisodesDataset:
    def __init__(
        self,
        dataset: Dict,
        split: str,
        gpt_groundings: Dict,
        seed: int,
    ) -> None:
        self.max_num_episodes = int(1e9)
        self.num_seen_episodes = 0
        self.episodes = deque()
        self.episode_id_to_queue_idx = dict()
        self.newly_modified_episodes, self.newly_deleted_episodes = set(), set()
        self.random = random.Random(seed + sum([ord(x) for x in split]))
        self.split = split
        self._load_data(dataset, split, gpt_groundings)

    def _load_data(self, dataset: Dict, split: str, gpt_groundings: Dict):
        self.raw_dataset = dataset
        rollouts = dataset["rollouts"][split]
        gpt_is_correct = []
        max_len = []
        for i in tqdm(range(len(rollouts["grid_sequences"]))):
            raw_manual = get_raw_manual(dataset, split, i)
            parsed_manual = [gpt_groundings[e] for e in raw_manual]
            true_parsed_manual = get_true_parsed_manual(dataset, split, i)

            observations = torch.tensor(np.array(rollouts["grid_sequences"][i])).long()
            actions = torch.tensor(rollouts["action_sequences"][i]).long()
            rewards = torch.tensor(rollouts["reward_sequences"][i]).float()
            dones = torch.tensor(rollouts["done_sequences"][i]).long()
            mask_padding = torch.ones(observations.shape[0]).bool()

            assert (
                observations.shape[0]
                == actions.shape[0]
                == rewards.shape[0]
                == dones.shape[0]
                == mask_padding.shape[0]
            )

            gpt_is_correct.append(
                sorted([x[0] for x in parsed_manual])
                == sorted([x[0] for x in true_parsed_manual])
            )

            episode = Episode(
                observations,
                actions,
                rewards,
                dones,
                mask_padding,
                raw_manual,
                parsed_manual,
                true_parsed_manual,
            )
            self.add_episode(episode)
            max_len.append(len(episode))
        print("max len", max(max_len))
        print(f"Split {split} GPT grounding accuracy: {np.average(gpt_is_correct):.2f}")

    def __len__(self) -> int:
        return len(self.episodes)

    def clear(self) -> None:
        self.episodes = deque()
        self.episode_id_to_queue_idx = dict()

    def add_episode(self, episode: Episode) -> int:
        if (
            self.max_num_episodes is not None
            and len(self.episodes) == self.max_num_episodes
        ):
            self._popleft()
        episode_id = self._append_new_episode(episode)
        return episode_id

    def get_episode(self, episode_id: int) -> Episode:
        assert episode_id in self.episode_id_to_queue_idx
        queue_idx = self.episode_id_to_queue_idx[episode_id]
        return self.episodes[queue_idx]

    def update_episode(self, episode_id: int, new_episode: Episode) -> None:
        assert episode_id in self.episode_id_to_queue_idx
        queue_idx = self.episode_id_to_queue_idx[episode_id]
        merged_episode = self.episodes[queue_idx].merge(new_episode)
        self.episodes[queue_idx] = merged_episode
        self.newly_modified_episodes.add(episode_id)

    def _popleft(self) -> Episode:
        id_to_delete = [k for k, v in self.episode_id_to_queue_idx.items() if v == 0]
        assert len(id_to_delete) == 1
        self.newly_deleted_episodes.add(id_to_delete[0])
        self.episode_id_to_queue_idx = {
            k: v - 1 for k, v in self.episode_id_to_queue_idx.items() if v > 0
        }
        return self.episodes.popleft()

    def _append_new_episode(self, episode):
        episode_id = self.num_seen_episodes
        self.episode_id_to_queue_idx[episode_id] = len(self.episodes)
        self.episodes.append(episode)
        self.num_seen_episodes += 1
        self.newly_modified_episodes.add(episode_id)
        return episode_id

    def iterate_batches(self, batch_size: int, sequence_length: int):
        for i in range(0, len(self.episodes), batch_size):
            episodes = list(itertools.islice(self.episodes, i, i + batch_size))
            assert episodes
            segments = []
            for ep in episodes:
                j = 0
                while j + 1 < len(ep):
                    segments.append(ep.segment(j, j + sequence_length, should_pad=True))
                    j = j + sequence_length - 1
            assert segments
            yield self._collate_episodes_segments(segments)

    def sample_batch(
        self,
        batch_num_samples: int,
        sequence_length: int,
        weights: Optional[Tuple[float]] = None,
        sample_from_start: bool = True,
    ) -> Batch:
        return self._collate_episodes_segments(
            self._sample_episodes_segments(
                batch_num_samples, sequence_length, weights, sample_from_start
            )
        )

    def _sample_episodes_segments(
        self,
        batch_num_samples: int,
        sequence_length: int,
        weights: Optional[Tuple[float]],
        sample_from_start: bool,
    ) -> List[Episode]:
        num_episodes = len(self.episodes)
        num_weights = len(weights) if weights is not None else 0

        if num_weights < num_episodes:
            weights = [1] * num_episodes
        else:
            assert all([0 <= x <= 1 for x in weights]) and sum(weights) == 1
            sizes = [
                num_episodes // num_weights
                + (num_episodes % num_weights) * (i == num_weights - 1)
                for i in range(num_weights)
            ]
            weights = [w / s for (w, s) in zip(weights, sizes) for _ in range(s)]

        sampled_episodes = self.random.choices(
            self.episodes, k=batch_num_samples, weights=weights
        )

        sampled_episodes_segments = []
        for sampled_episode in sampled_episodes:
            if sample_from_start:
                start = self.random.randint(0, len(sampled_episode) - 1)
                stop = start + sequence_length
            else:
                stop = self.random.randint(1, len(sampled_episode))
                start = stop - sequence_length
            sampled_episodes_segments.append(
                sampled_episode.segment(start, stop, should_pad=True)
            )
            assert len(sampled_episodes_segments[-1]) == sequence_length
        return sampled_episodes_segments

    def _collate_episodes_segments(self, episodes_segments: List[Episode]) -> Batch:
        episodes_segments = [e_s.__dict__ for e_s in episodes_segments]
        batch = {}
        for k in episodes_segments[0]:
            batch[k] = [e_s[k] for e_s in episodes_segments]
            if isinstance(batch[k][0], torch.Tensor):
                batch[k] = torch.stack(batch[k])
        return batch

    def traverse(self, batch_num_samples: int, chunk_size: int):
        for episode in self.episodes:
            chunks = [
                episode.segment(
                    start=i * chunk_size, stop=(i + 1) * chunk_size, should_pad=True
                )
                for i in range(math.ceil(len(episode) / chunk_size))
            ]
            batches = [
                chunks[i * batch_num_samples : (i + 1) * batch_num_samples]
                for i in range(math.ceil(len(chunks) / batch_num_samples))
            ]
            for b in batches:
                yield self._collate_episodes_segments(b)

    def update_disk_checkpoint(self, directory: Path) -> None:
        assert directory.is_dir()
        for episode_id in self.newly_modified_episodes:
            episode = self.get_episode(episode_id)
            episode.save(directory / f"{episode_id}.pt")
        for episode_id in self.newly_deleted_episodes:
            (directory / f"{episode_id}.pt").unlink()
        self.newly_modified_episodes, self.newly_deleted_episodes = set(), set()

    def load_disk_checkpoint(self, directory: Path) -> None:
        assert directory.is_dir() and len(self.episodes) == 0
        episode_ids = sorted([int(p.stem) for p in directory.iterdir()])
        self.num_seen_episodes = episode_ids[-1] + 1
        for episode_id in episode_ids:
            episode = Episode(**torch.load(directory / f"{episode_id}.pt"))
            self.episode_id_to_queue_idx[episode_id] = len(self.episodes)
            self.episodes.append(episode)
