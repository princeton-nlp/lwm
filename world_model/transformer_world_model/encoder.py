from einops import rearrange

import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer
from messenger.models.utils import BatchedEncoder

from .transformer import Transformer, TransformerConfig
from .slicer import Embedder


class BaseEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.tokenizer = {name: func() for name, func in config.tokenizers.items()}
        self.transformer = Transformer(config)


class StandardEncoder(BaseEncoder):
    """Transformer encoder for raw manuals
    Use BERT to embed the descriptions
    All descriptions are concatenated into a single sequence
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        bert_model = AutoModel.from_pretrained("bert-base-uncased")
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_encoder = BatchedEncoder(
            model=bert_model,
            tokenizer=bert_tokenizer,
            device=config.device,
            max_length=36,
        )
        self.linear_proj = nn.Linear(768, config.embed_dim)
        self.pos_emb = nn.Embedding(40 * 3, config.embed_dim)

    def __repr__(self) -> str:
        return "standard_encoder"

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embeds, _ = self.bert_encoder.encode(tokens)
        embeds = self.linear_proj(embeds)

        shape = embeds.shape
        embeds = rearrange(embeds, "b n l e -> b (n l) e")
        num_steps = embeds.size(1)
        embeds = embeds + self.pos_emb(torch.arange(num_steps, device=embeds.device))

        x = self.transformer(embeds)

        return x


class StandardEncoderV2(StandardEncoder):
    """Similar to StandardEncoder but run each description through encoder separately"""

    def __repr__(self) -> str:
        return "standard_encoder_v2"

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embeds, _ = self.bert_encoder.encode(tokens)
        embeds = self.linear_proj(embeds)

        shape = embeds.shape
        embeds = rearrange(embeds, "b n l e -> (b n) l e")

        num_steps = embeds.size(1)
        embeds = embeds + self.pos_emb(torch.arange(num_steps, device=embeds.device))

        x = self.transformer(embeds)
        x = x.reshape(*shape[:2], *x.shape[1:])
        x = rearrange(x, "b n l e -> b (n l) e")

        return x


class ParsedFeatureEncoder(BaseEncoder):
    """Transformer encoder for descriptions that have been parsed into features by ChatGPT
    Embed the descriptions and feed embeddings into a Transformer
    Average the output over all features
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks={"manual": torch.ones(config.tokens_per_block)},
            embedding_tables=nn.ModuleDict(
                {
                    "manual": nn.Embedding(
                        self.tokenizer["manual"].vocab_size, config.embed_dim
                    )
                }
            ),
        )

    def __repr__(self) -> str:
        return "parsed_feature_encoder"

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        assert tokens.dim() == 3

        shape = tokens.shape
        tokens = rearrange(tokens, "b n l -> (b n) l")

        num_steps = tokens.size(1)
        sequences = self.embedder(tokens, num_steps, 0) + self.pos_emb(
            torch.arange(num_steps, device=tokens.device)
        )
        x = self.transformer(sequences)
        x = x.reshape(*shape[:2], *x.shape[1:])

        return x


class RawTextEncoder(BaseEncoder):
    """Transformer encoder for raw manuals
    Use BERT to embed the descriptions
    Unlike the StandardEncoder, this one does not concatenate the descriptions
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        bert_model = AutoModel.from_pretrained("bert-base-uncased")
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_encoder = BatchedEncoder(
            model=bert_model,
            tokenizer=bert_tokenizer,
            device=config.device,
            max_length=36,
        )
        self.linear_proj = nn.Linear(768, config.embed_dim)

    def __repr__(self) -> str:
        return "raw_text_encoder"

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embeds, _ = self.bert_encoder.encode(tokens)
        embeds = self.linear_proj(embeds)

        shape = embeds.shape
        embeds = rearrange(embeds, "b n l e -> (b n) l e")
        x = self.transformer(embeds)
        x = x.reshape(*shape[:2], *x.shape[1:])

        return x
