"""
Credits to https://github.com/karpathy/minGPT and https://github.com/eloialonso/iris
"""

from dataclasses import dataclass
import math
from typing import Optional, Dict, Callable, Union, Tuple

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F

from .kv_caching import KeysValues, KVCache


@dataclass
class TransformerConfig:
    tokens_per_block: int
    max_blocks: int
    # attention: str

    num_layers: int
    num_heads: int
    embed_dim: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    tokenizers: Dict[str, Callable]
    has_external_memory: bool
    device: torch.device

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        device = (
            self.ln_f.weight.device
        )  # Assumption that all submodules are on the same device
        return KeysValues(
            n,
            self.config.num_heads,
            max_tokens,
            self.config.embed_dim,
            self.config.num_layers,
            device,
        )

    def forward(
        self,
        sequences: torch.Tensor,
        external_keys_values: Optional[torch.Tensor] = None,
        past_keys_values: Optional[KeysValues] = None,
        self_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)
        x = self.drop(sequences)
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                external_keys_values=external_keys_values,
                past_keys_values=None
                if past_keys_values is None
                else past_keys_values[i],
                self_attention_mask=self_attention_mask,
            )

        x = self.ln_f(x)
        return x


class Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln_self_attn = nn.LayerNorm(config.embed_dim)
        self.ln_ff = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.resid_pdrop),
        )
        self.has_external_memory = config.has_external_memory
        if self.has_external_memory:
            self.ln_external_attn = nn.LayerNorm(config.embed_dim)
            self.external_attn = ExternalAttention(config)

    def forward(
        self,
        x: torch.Tensor,
        external_keys_values: Optional[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        past_keys_values: Optional[KeysValues] = None,
        self_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # self attention
        x_attn = self.attn(
            self.ln_self_attn(x),
            kv_cache=past_keys_values,
            attn_mask=self_attention_mask,
        )
        x = x + x_attn
        # external attention
        if external_keys_values is not None:
            if isinstance(external_keys_values, tuple):
                external_keys, external_values = external_keys_values
            else:
                assert isinstance(external_keys_values, torch.Tensor)
                external_keys = external_values = external_keys_values
            x_external_attn = self.external_attn(
                self.ln_external_attn(x), external_keys, external_values
            )
            x = x + x_external_attn
        # feed-forward
        x = x + self.mlp(self.ln_ff(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        # assert config.attention in ("causal", "block_causal")
        self.num_heads = config.num_heads
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        """
        causal_mask = torch.tril(torch.ones(config.max_tokens, config.max_tokens))
        block_causal_mask = torch.max(
            causal_mask,
            torch.block_diag(
                *[
                    torch.ones(config.tokens_per_block, config.tokens_per_block)
                    for _ in range(config.max_blocks)
                ]
            ),
        )
        self.register_buffer(
            "mask", causal_mask if config.attention == "causal" else block_causal_mask
        )
        """

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()
        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
            assert nh == self.num_heads and b == B and c * nh == C
        else:
            L = 0

        q = (
            self.query(x)
            .view(B, T, self.num_heads, C // self.num_heads)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        k = (
            self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x)
            .view(B, T, self.num_heads, C // self.num_heads)
            .transpose(1, 2)
        )  # (B, nh, T, hs)

        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, float("-inf"))
            # att = att.masked_fill(self.mask[L : L + T, : L + T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = rearrange(y, "b h t e -> b t (h e)")

        y = self.resid_drop(self.proj(y))

        return y


class ExternalAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        self.num_heads = config.num_heads
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(
        self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor
    ) -> torch.Tensor:
        B_q, T_q, C_q = x_q.size()
        q = (
            self.query(x_q)
            .view(B_q, T_q, self.num_heads, C_q // self.num_heads)
            .transpose(1, 2)
        )  # (B, nh, T, hs)

        B_k, T_k, C_k = x_k.size()
        k = (
            self.key(x_k)
            .view(B_k, T_k, self.num_heads, C_k // self.num_heads)
            .transpose(1, 2)
        )  # (B, nh, T, hs)

        B_v, T_v, C_v = x_v.size()
        v = (
            self.value(x_v)
            .view(B_v, T_v, self.num_heads, C_v // self.num_heads)
            .transpose(1, 2)
        )  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        # for i in range(att.size(1)):
        #    print(att[0, i, 0].tolist())
        # print()
        att = self.attn_drop(att)

        y = att @ v
        y = rearrange(y, "b h t e -> b t (h e)")

        y = self.resid_drop(self.proj(y))

        return y


"""
class KeyValueGenerator(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.key_attn = self.RepeatAttention(config)
        self.value_attn = self.RepeatAttention(config)

    def forward(
        self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        keys = self.key_attn(x_q, x_k, x_v)
        values = self.value_attn(x_q, x_k, x_v)
        return keys, values


class RepeatAttention(nn.Module):

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        self.num_heads = config.num_heads
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

    def _compute(
        self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, name: str
    ) -> torch.Tensor:
        B_q, T_q, C_q = x_q.size()
        q = (
            self.query(x_q)
            .view(B_q, T_q, self.num_heads, C_q // self.num_heads)
            .transpose(1, 2)
        )  # (B, nh, T, hs)

        B_k, T_k, C_k = x_k.size()
        k = (
            self.key(x_k)
            .view(B_k, T_k, self.num_heads, C_k // self.num_heads)
            .transpose(1, 2)
        )  # (B, nh, T, hs)

        B_v, N_v, T_v, C_v = x_v.size()
        v = (
            self.value(x_v)
            .view(B_v, N_v, T_v, self.num_heads, C_v // self.num_heads)
            .transpose(2, 3)
        )  # (B, N, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)  # (B, nh, T_q, T_v)

        att = att.unsqueeze(1).expand(-1, N_v, -1, -1, -1)  # (B, N, nh, T_q, T_v)

        y = att @ v  # (B, N, nh, T_q, hs)
        y = rearrange(y, "b n h t e -> b n t (h e)")

        y = self.resid_drop(self.proj(y))

        return y
"""
