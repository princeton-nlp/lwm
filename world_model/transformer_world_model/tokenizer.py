import torch
import random
import math

from .utils import ENTITY_IDS, MOVEMENT_IDS, ROLE_IDS


class Tokenizer:
    def id_to_word(self, id):
        return self.id_to_word_map[id]

    def word_to_id(self, word):
        return self.word_to_id_map[word]

    def words_to_ids(self, words):
        return [self.word_to_id(w) for w in words]

    @property
    def vocab_size(self):
        return len(self.vocab)


class ObservationTokenizer(Tokenizer):
    offset = {"id": 0, "row": 17, "col": 27}
    obs_shape = [10, 10, 4]

    def __init__(self):
        self.vocab = []

        for i in range(17):
            self.vocab.append(f"id_{i}")
        for i in range(10):
            self.vocab.append(f"row_{i}")
        for i in range(10):
            self.vocab.append(f"col_{i}")
        for i in range(4):
            self.vocab.append(f"entity_{i}")

        for w in MOVEMENT_IDS:
            self.vocab.append(f"movement_{w}")
        for w in ROLE_IDS:
            self.vocab.append(f"role_{w}")

        for i in range(4):
            self.vocab.append(f"entity_{i}")
        self.vocab.append("[MASK]")

        self.word_to_id_map = {w: i for i, w in enumerate(self.vocab)}
        self.id_to_word_map = {i: w for i, w in enumerate(self.vocab)}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape  # (..., H, W, C)

        # print(x[0][0].view(100, 4).max(0)[0])
        # print(x[0][0].view(100, 4).max(0)[1])
        # print(x[0][0].sum(-1))
        # print(x[0, 0].sum(-1))

        x = x.reshape(-1, *shape[-3:])
        b, h, w, c = x.shape
        id, loc = x.reshape(b, -1, c).max(1)

        row = torch.div(loc + 1e-6, h, rounding_mode="trunc")
        col = loc % h
        row[id == 0] = 0
        col[id == 0] = 0

        ent = torch.arange(c).to(x.device).view(1, c).expand_as(id)

        tokens = []
        for i in range(c):
            tokens.extend(
                [
                    # ent[:, i] + self.offset["entity"],
                    id[:, i] + self.offset["id"],
                    row[:, i] + self.offset["row"],
                    col[:, i] + self.offset["col"],
                ]
            )
        tokens = torch.stack(tokens, dim=1).long()
        tokens = tokens.reshape(*shape[:-3], -1)

        # print(tokens[0, 0].tolist())
        # for v in tokens[0, 0].tolist():
        #    print(self.id_to_word(v), end=' ')
        # print()
        # input()

        assert tokens.shape[-1] == c * len(self.offset)

        """
        # check
        rand_id = random.randint(0, tokens.size(0) - 1)
        # rand_id = 0
        for i, token_id in enumerate(tokens[rand_id, 0].tolist()):
            # print(self.id_to_word[token_id])
            # if i % 4 == 0:
            #    assert "entity_" in self.id_to_word[token_id]
            if i % 3 == 0:
                assert "id_" in self.id_to_word(token_id)
            if i % 3 == 1:
                assert "row_" in self.id_to_word(token_id)
            if i % 3 == 2:
                assert "col_" in self.id_to_word(token_id)
        """

        return tokens

    def decode(self, obs_tokens: torch.Tensor) -> torch.FloatTensor:
        shape = obs_tokens.shape
        obs_tokens = obs_tokens.reshape(-1, obs_tokens.shape[-1])
        obs = (
            torch.zeros([math.prod(shape[:-1])] + self.obs_shape)
            .to(obs_tokens.device)
            .long()
        )
        n_channels = self.obs_shape[-1]
        tokens_per_entity = len(self.offset)
        for i in range(n_channels):
            id = obs_tokens[:, i * tokens_per_entity] - self.offset["id"]
            id = torch.clamp(id, 0, 16)
            row = obs_tokens[:, i * tokens_per_entity + 1] - self.offset["row"]
            row = torch.clamp(row, 0, 9)
            col = obs_tokens[:, i * tokens_per_entity + 2] - self.offset["col"]
            col = torch.clamp(col, 0, 9)
            obs[torch.arange(obs.shape[0]), row, col, i] = id

        obs = obs.reshape(*shape[:-1], *obs.shape[1:])

        return obs
