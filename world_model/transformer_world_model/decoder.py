from dataclasses import dataclass
from typing import Any, Optional, Tuple, List
from copy import deepcopy as dc
from einops import rearrange

import torch
import torch.nn as nn

from .transformer import Transformer, TransformerConfig, ExternalAttention
from .kv_caching import KeysValues
from .slicer import Embedder, Head


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_dones: torch.FloatTensor


class BaseDecoder(nn.Module):
    # num tokens per entity
    tokens_per_entity = 3
    # number of observation tokens
    tokens_per_observation = 12

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config
        self.device = config.device
        self.tokenizer = {name: func() for name, func in config.tokenizers.items()}

        self._make_embedder()
        self._make_body()
        self._make_head()
        self._make_attention_masks()

    def _make_body(self):
        pass

    def _make_embedder(self):
        config = self.config

        n_o = self.tokens_per_observation
        n_r = 1

        token_types = ["action", "observation", "reward", "done"]
        vocab_size = {
            "action": 5,
            "observation": self.tokenizer["observation"].vocab_size,
            "reward": 10,
            "done": 2,
        }

        tokens_pattern = {}
        for t in token_types:
            tokens_pattern[t] = torch.zeros(config.tokens_per_block)
        # first token is action
        tokens_pattern["action"][0] = 1
        # observation tokens
        tokens_pattern["observation"][1 : (n_o + 1)] = 1
        # reward token
        tokens_pattern["reward"][(n_o + 1) : (n_o + n_r + 1)] = 1
        # done token
        tokens_pattern["done"][(n_o + n_r + 1) : (n_o + n_r + 2)] = 1

        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks={t: tokens_pattern[t] for t in token_types},
            embedding_tables=nn.ModuleDict(
                {t: nn.Embedding(vocab_size[t], config.embed_dim) for t in token_types}
            ),
        )
        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

    def _make_head(self):
        config = self.config

        n_o = self.tokens_per_observation
        n_r = 1

        head_token_types = ["observation", "reward", "done"]
        vocab_size = {
            "observation": self.tokenizer["observation"].vocab_size,
            "reward": 10,
            "done": 2,
        }

        head_tokens_pattern = {}
        for t in head_token_types:
            head_tokens_pattern[t] = torch.zeros(config.tokens_per_block)
        head_tokens_pattern["observation"][:n_o] = 1
        head_tokens_pattern["reward"][n_o : (n_o + n_r)] = 1
        head_tokens_pattern["done"][(n_o + n_r) : (n_o + n_r + 1)] = 1

        self.head = nn.ModuleDict(
            {
                t: Head(
                    max_blocks=config.max_blocks,
                    block_mask=head_tokens_pattern[t],
                    head_module=nn.Sequential(
                        nn.Linear(config.embed_dim, config.embed_dim),
                        nn.ReLU(),
                        nn.Linear(config.embed_dim, vocab_size[t]),
                    ),
                )
                for t in head_token_types
            }
        )

    def _make_attention_masks(self):
        history_masks = torch.tril(
            torch.ones(self.config.max_tokens, self.config.max_tokens)
        )
        self.register_buffer("history_masks", history_masks)

    def reset(self, n: int) -> KeysValues:
        return self.decoder.generate_empty_keys_values(n, self.config.max_tokens)


class StandardDecoder(BaseDecoder):
    """Standard Transformer decoder with interleaved multi-headed self- and cros- attentions"""

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

    def __repr__(self) -> str:
        return "standard_decoder"

    def _make_body(self):
        self.decoder = Transformer(self.config)

    def forward(
        self,
        tokens: torch.Tensor,
        encoded_manuals: torch.Tensor,
        past_keys_values: Optional[KeysValues] = None,
    ) -> WorldModelOutput:
        num_steps = tokens.size(1)  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        embeds = self.embedder(tokens, num_steps, prev_steps) + self.pos_emb(
            prev_steps + torch.arange(num_steps, device=tokens.device)
        )

        history_attn_mask = self.history_masks[
            prev_steps : (prev_steps + num_steps), : (prev_steps + num_steps)
        ]

        x = self.decoder(
            embeds,
            external_keys_values=encoded_manuals,
            past_keys_values=past_keys_values,
            self_attention_mask=history_attn_mask,
        )

        outputs = {
            f"logits_{t}s": h(x, num_steps=num_steps, prev_steps=prev_steps)
            for t, h in self.head.items()
        }

        return WorldModelOutput(output_sequence=x, **outputs)


class TrueGroundDecoder(StandardDecoder):
    """Transformer Decoder that grounds each description to correct entity token in observation"""

    def __repr__(self) -> str:
        return "true_ground_decoder"

    def _make_body(self):
        config = self.config
        self.manual_key_attn = nn.Sequential(
            nn.Linear(config.embed_dim, 1), nn.Softmax(dim=2)
        )
        self.decoder = Transformer(config)

    def forward(
        self,
        tokens: torch.Tensor,
        encoded_manuals: torch.Tensor,
        past_keys_values: Optional[KeysValues] = None,
    ) -> WorldModelOutput:
        num_steps = tokens.size(1)  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        manual_keys = (self.manual_key_attn(encoded_manuals) * encoded_manuals).sum(
            dim=-2
        )

        attr_embeds = []
        for i in range(num_steps):
            k = (i + prev_steps) % self.config.tokens_per_block
            if k in [1, 4, 7]:
                attr_embeds.append(manual_keys[:, k // 3])
            else:
                attr_embeds.append(torch.zeros_like(manual_keys[:, 0]))
        attr_embeds = torch.stack(attr_embeds, dim=1)

        embeds = self.embedder(tokens, num_steps, prev_steps) + self.pos_emb(
            prev_steps + torch.arange(num_steps, device=tokens.device)
        )
        embeds = embeds + attr_embeds

        history_attn_mask = self.history_masks[
            prev_steps : (prev_steps + num_steps), : (prev_steps + num_steps)
        ]

        x = self.decoder(
            embeds,
            past_keys_values=past_keys_values,
            self_attention_mask=history_attn_mask,
        )

        outputs = {
            f"logits_{t}s": h(x, num_steps=num_steps, prev_steps=prev_steps)
            for t, h in self.head.items()
        }

        return WorldModelOutput(output_sequence=x, **outputs)


class EmmaDecoder(BaseDecoder):
    """Transformer decoder with EMMA-style attention"""

    def __repr__(self) -> str:
        return "emma_decoder"

    def _make_body(self):
        config = self.config
        self.manual_key_attn = nn.Sequential(
            nn.Linear(config.embed_dim, 1), nn.Softmax(dim=2)
        )
        self.manual_value_attn = nn.Sequential(
            nn.Linear(config.embed_dim, 1), nn.Softmax(dim=2)
        )
        manual_attn_config = dc(config)
        manual_attn_config.num_heads = 1
        self.manual_attn = ExternalAttention(manual_attn_config)

        self.decoder = Transformer(config)

    def forward(
        self,
        tokens: torch.Tensor,
        encoded_manuals: torch.Tensor,
        past_keys_values: Optional[KeysValues] = None,
    ) -> WorldModelOutput:
        num_steps = tokens.size(1)  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        manual_keys = (self.manual_key_attn(encoded_manuals) * encoded_manuals).sum(
            dim=-2
        )
        manual_values = (self.manual_value_attn(encoded_manuals) * encoded_manuals).sum(
            dim=-2
        )

        embeds = self.embedder(tokens, num_steps, prev_steps)

        manual_queries = []
        for i in range(num_steps):
            k = (i + prev_steps) % self.config.tokens_per_block
            if k in [1, 4, 7]:
                manual_queries.append(embeds[:, i])

        if manual_queries:
            manual_queries = torch.stack(manual_queries, dim=1)
            attn_values = self.manual_attn(manual_queries, manual_keys, manual_values)

        j = 0
        for i in range(num_steps):
            k = (i + prev_steps) % self.config.tokens_per_block
            if k in [1, 4, 7]:
                embeds[:, i] = embeds[:, i] + attn_values[:, j]
                j += 1

        embeds = embeds + self.pos_emb(
            prev_steps + torch.arange(num_steps, device=tokens.device)
        )

        history_attn_mask = self.history_masks[
            prev_steps : (prev_steps + num_steps), : (prev_steps + num_steps)
        ]

        x = self.decoder(
            embeds,
            past_keys_values=past_keys_values,
            self_attention_mask=history_attn_mask,
        )

        outputs = {
            f"logits_{t}s": h(x, num_steps=num_steps, prev_steps=prev_steps)
            for t, h in self.head.items()
        }

        return WorldModelOutput(output_sequence=x, **outputs)
