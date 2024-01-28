import math
from typing import List, Dict

import torch
import torch.nn as nn


class Slicer(nn.Module):
    def __init__(
        self, max_blocks: int, block_mask: torch.Tensor, offset_steps: int = 0
    ) -> None:
        super().__init__()
        self.offset_steps = offset_steps
        self.block_size = block_mask.size(0)
        self.num_kept_tokens = block_mask.sum().long().item()
        kept_indices = torch.where(block_mask)[0].repeat(max_blocks)
        offsets = torch.arange(max_blocks).repeat_interleave(self.num_kept_tokens)
        self.register_buffer("indices", kept_indices + block_mask.size(0) * offsets)

    def compute_slice(self, num_steps: int, prev_steps: int = 0) -> torch.Tensor:
        prev_steps -= self.offset_steps
        total_steps = num_steps + prev_steps
        num_blocks = math.ceil(total_steps / self.block_size)
        indices = self.indices[: num_blocks * self.num_kept_tokens]
        return (
            indices[torch.logical_and(prev_steps <= indices, indices < total_steps)]
            - prev_steps
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Head(Slicer):
    def __init__(
        self,
        max_blocks: int,
        block_mask: torch.Tensor,
        head_module: nn.Module,
        offset_steps: int = 0,
    ) -> None:
        super().__init__(max_blocks, block_mask, offset_steps)
        assert isinstance(head_module, nn.Module)
        self.head_module = head_module

    def forward(self, x: torch.Tensor, num_steps: int, prev_steps: int) -> torch.Tensor:
        x_sliced = x[:, self.compute_slice(num_steps, prev_steps)]  # x is (B, T, E)
        return self.head_module(x_sliced)


class Embedder(nn.Module):
    def __init__(
        self,
        max_blocks: int,
        block_masks: Dict[str, torch.Tensor],
        embedding_tables: nn.ModuleDict,
        offset_steps: int = 0,
    ) -> None:
        super().__init__()
        assert len(block_masks) == len(embedding_tables)
        assert (
            sum(block_masks.values()) == 1
        ).all()  # block mask are a partition of a block
        self.embedding_dim = list(embedding_tables.values())[0].embedding_dim
        assert all(
            [e.embedding_dim == self.embedding_dim for e in embedding_tables.values()]
        )
        self.embedding_tables = embedding_tables
        self.slicers = {
            name: Slicer(max_blocks, block_mask, offset_steps)
            for name, block_mask in block_masks.items()
        }

    def forward(
        self, tokens: torch.Tensor, num_steps: int, prev_steps: int
    ) -> torch.Tensor:
        assert tokens.ndim == 2  # x is (B, T)
        output = torch.zeros(*tokens.size(), self.embedding_dim, device=tokens.device)
        for name in self.embedding_tables:
            s = self.slicers[name].compute_slice(num_steps, prev_steps)
            output[:, s] = self.embedding_tables[name](tokens[:, s])
        return output
