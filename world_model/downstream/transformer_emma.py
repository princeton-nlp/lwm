""" EMMA policy with Transformer base architecture"""

from typing import Any, Optional, Tuple, List
from copy import deepcopy as dc
from einops import rearrange

from torch.distributions import Categorical
import torch
import torch.nn as nn

from transformer_world_model.tokenizer import ObservationTokenizer
from transformer_world_model.transformer import (
    TransformerConfig,
    Transformer,
    ExternalAttention,
)
from transformer_world_model.slicer import Embedder, Head
from transformer_world_model.kv_caching import KeysValues
from transformer_world_model.world_model import init_weights


class TransformerEMMA(nn.Module):
    STAY_ACTION = 4

    def __init__(self, args):
        super().__init__()

        encoder_config = TransformerConfig(
            tokens_per_block=args.encoder_tokens_per_block,
            max_blocks=args.encoder_max_blocks,
            num_layers=args.encoder_layers,
            num_heads=args.encoder_num_heads,
            embed_dim=args.hidden_size,
            embed_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            tokenizers={},
            has_external_memory=False,
            device=args.device,
        )

        decoder_config = TransformerConfig(
            tokens_per_block=args.decoder_tokens_per_block,
            max_blocks=args.decoder_max_blocks,
            num_layers=args.decoder_layers,
            num_heads=args.decoder_num_heads,
            embed_dim=args.hidden_size,
            embed_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            tokenizers={"observation": ObservationTokenizer},
            has_external_memory=False,
            device=args.device,
        )

        self.encoder = RawTextPolicyEncoder(encoder_config)
        self.decoder = EmmaPolicyDecoder(decoder_config)
        self.device = args.device

        self.apply(init_weights)

    def reset(self, n):
        self.encoded_manuals = None
        self.last_action = self.STAY_ACTION
        self.past_keys_values = self.decoder.reset(n)

    def tokenize_observations(self, observations: torch.tensor) -> torch.tensor:
        return self.decoder.tokenizer["observation"].encode(observations)

    def act(self, obs, temb, memory):
        if self.encoded_manuals is None:
            self.encoded_manuals = self.encoder(temb.unsqueeze(0))

        if (
            self.past_keys_values.size + self.decoder.tokens_per_block
            > self.decoder.config.max_tokens
        ):
            self.past_keys_values = self.decoder.reset(1)
            self.decoder(self.last_tokens, self.encoded_manuals, self.past_keys_values)

        act_tokens = (
            torch.tensor([self.last_action]).to(self.device).long().reshape(-1, 1, 1)
        )
        obs_tokens = self.tokenize_observations(obs.unsqueeze(1))
        tokens = rearrange(
            torch.cat((act_tokens, obs_tokens), dim=2), "b l k -> b (l k)"
        )
        self.last_tokens = tokens

        action_dists, _ = self.decoder(
            tokens, self.encoded_manuals, self.past_keys_values
        )
        action_dists = action_dists.squeeze()

        if self.training:
            dist = Categorical(action_dists)
            action = dist.sample()

            # store everything in memory
            memory.states.append(obs)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
            memory.texts.append(temb)
            action = action.item()
        else:
            action = torch.argmax(action_dists).item()
            """
            if random.random() < 0.05:  # random action with 0.05 prob
                action = random.randrange(0, self.action_dim)
            """

        self.last_action = action
        return action

    def evaluate(self, obs, action, ref_action, temb, mask):
        encoded_manuals = self.encoder(temb)

        obs_tokens = self.tokenize_observations(obs)
        act_tokens = rearrange(action, "b l -> b l 1")
        tokens = rearrange(
            torch.cat((act_tokens, obs_tokens), dim=2), "b l k -> b (l k)"
        )

        action_dists, value = self.decoder(tokens, encoded_manuals)
        """
        print(obs[0, 0].sum(-1))
        print(action_dists[0, 0])
        for i in range(action_dists.shape[1]):
            print("--->", action_dists[0, i].log()[ref_action[0, i]], mask[0, i])
        print(ref_action[0, 0])
        """

        action_dists = action_dists.reshape(-1, action_dists.shape[-1])
        value = value.flatten()

        dist = Categorical(action_dists)
        action_logprobs = dist.log_prob(ref_action.flatten())
        dist_entropy = dist.entropy()

        mask = mask.flatten()
        action_logprobs = torch.masked_select(action_logprobs, mask)
        value = torch.masked_select(value.squeeze(), mask)
        dist_entropy = torch.masked_select(dist_entropy, mask)

        return action_logprobs, value, dist_entropy


class RawTextPolicyEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.transformer = Transformer(config)
        self.linear_proj = nn.Linear(768, config.embed_dim)

    def __repr__(self) -> str:
        return "raw_text_policy_encoder"

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        embeds = self.linear_proj(embeds)
        shape = embeds.shape
        embeds = rearrange(embeds, "b n l e -> (b n) l e")
        x = self.transformer(embeds)
        x = x.reshape(*shape[:2], *x.shape[1:])

        return x


class EmmaPolicyDecoder(nn.Module):
    """Transformer decoder with EMMA-style attention"""

    # tokens per entity
    tokens_per_entity = 3
    # tokens per observations
    tokens_per_observation = 12
    # tokens per block
    tokens_per_block = 13

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config
        self.device = config.device
        self.tokenizer = {name: func() for name, func in config.tokenizers.items()}

        self._make_embedder()
        self._make_body()
        self._make_head()
        self._make_attention_masks()

    def _make_embedder(self):
        config = self.config

        token_types = ["action", "observation"]
        vocab_size = {
            "action": 5,
            "observation": self.tokenizer["observation"].vocab_size,
        }

        tokens_pattern = {}
        for t in token_types:
            tokens_pattern[t] = torch.zeros(self.tokens_per_block)
        # first token is action
        tokens_pattern["action"][0] = 1
        # observation tokens
        tokens_pattern["observation"][1:] = 1

        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks={t: tokens_pattern[t] for t in token_types},
            embedding_tables=nn.ModuleDict(
                {t: nn.Embedding(vocab_size[t], config.embed_dim) for t in token_types}
            ),
        )
        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)
        self.pos_emb_2 = nn.Embedding(config.max_tokens, config.embed_dim)

    def _make_head(self):
        config = self.config

        n_o = self.tokens_per_observation

        head_token_types = ["action", "value"]
        head_tokens_pattern = {}
        for t in head_token_types:
            head_tokens_pattern[t] = torch.zeros(self.tokens_per_block)
        head_tokens_pattern["action"][-1] = 1
        head_tokens_pattern["value"][-1] = 1

        n_hidden_layers = 1
        hidden_layers = (
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.LeakyReLU(),
        ) * n_hidden_layers

        layer = {}
        layer["action"] = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.LeakyReLU(),
            *hidden_layers,
            nn.Linear(config.embed_dim, 5),
            nn.Softmax(dim=-1)
        )

        # critic
        layer["value"] = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.LeakyReLU(),
            *hidden_layers,
            nn.Linear(config.embed_dim, 1)
        )

        self.action_layer = layer["action"]
        self.value_layer = layer["value"]

        self.head = nn.ModuleDict(
            {
                t: Head(
                    max_blocks=config.max_blocks,
                    block_mask=head_tokens_pattern[t],
                    head_module=layer[t],
                )
                for t in head_token_types
            }
        )

    def __repr__(self) -> str:
        return "emma_policy_decoder"

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

    def _make_attention_masks(self):
        history_masks = torch.tril(
            torch.ones(self.config.max_tokens, self.config.max_tokens)
        )
        self.register_buffer("history_masks", history_masks)

    def reset(self, n: int) -> KeysValues:
        return self.decoder.generate_empty_keys_values(n, self.config.max_tokens)

    def forward(
        self,
        tokens: torch.Tensor,
        encoded_manuals: torch.Tensor,
        past_keys_values: Optional[KeysValues] = None,
    ) -> Tuple:
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
            k = (i + prev_steps) % self.tokens_per_block
            if k in [1, 4, 7]:
                manual_queries.append(embeds[:, i])

        if manual_queries:
            manual_queries = torch.stack(manual_queries, dim=1)
            attn_values = self.manual_attn(manual_queries, manual_keys, manual_values)

        j = 0
        for i in range(num_steps):
            k = (i + prev_steps) % self.tokens_per_block
            if k in [1, 4, 7]:
                embeds[:, i] = embeds[:, i] + attn_values[:, j]
                j += 1

        embeds = embeds + self.pos_emb(
            prev_steps + torch.arange(num_steps, device=tokens.device)
        )

        """
        embeds = embeds + self.pos_emb(torch.arange(num_steps, device=tokens.device))

        embeds = embeds + self.pos_emb_2(
            torch.tensor(prev_steps // self.tokens_per_block).to(self.device)
        )
        """

        history_attn_mask = self.history_masks[
            prev_steps : (prev_steps + num_steps), : (prev_steps + num_steps)
        ]

        x = self.decoder(
            embeds,
            past_keys_values=past_keys_values,
            self_attention_mask=history_attn_mask,
        )

        outputs = {
            t: h(x, num_steps=num_steps, prev_steps=prev_steps)
            for t, h in self.head.items()
        }

        return outputs["action"], outputs["value"]

        """
        x = embeds.mean(1)
        output_action = self.action_layer(x)
        output_value = self.value_layer(x)

        return output_action, output_value
        """
