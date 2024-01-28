"""
Code which implements the original EMMA model
Adapted from: https://github.com/ahjwang/messenger-emma/blob/main/training/model.py
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from numpy import sqrt as sqrt

from messenger.models.utils import nonzero_mean


class ConvEMMA(nn.Module):
    def __init__(
        self,
        state_h=10,
        state_w=10,
        action_dim=5,
        hist_len=3,
        n_latent_var=128,
        emb_dim=256,
        f_maps=64,
        kernel_size=2,
        n_hidden_layers=1,
    ):
        super().__init__()

        # calculate dimensions after flattening the conv layer output
        lin_dim = f_maps * (state_h - (kernel_size - 1)) * (state_w - (kernel_size - 1))
        self.conv = nn.Conv2d(hist_len * 256, f_maps, kernel_size)  # conv layer

        self.state_h = state_h
        self.state_w = state_w
        self.action_dim = action_dim
        self.emb_dim = emb_dim
        self.attn_scale = sqrt(emb_dim)

        self.sprite_emb = nn.Embedding(
            25, emb_dim, padding_idx=0
        )  # sprite embedding layer

        hidden_layers = (
            nn.Linear(n_latent_var, n_latent_var),
            nn.LeakyReLU(),
        ) * n_hidden_layers
        self.action_layer = nn.Sequential(
            nn.Linear(lin_dim, n_latent_var),
            nn.LeakyReLU(),
            *hidden_layers,
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(lin_dim, n_latent_var),
            nn.LeakyReLU(),
            *hidden_layers,
            nn.Linear(n_latent_var, 1)
        )

        # key value transforms
        self.txt_key = nn.Linear(768, emb_dim)
        self.scale_key = nn.Sequential(nn.Linear(768, 1), nn.Softmax(dim=-2))

        self.txt_val = nn.Linear(768, emb_dim)
        self.scale_val = nn.Sequential(nn.Linear(768, 1), nn.Softmax(dim=-2))

    def __repr__(self):
        return "ConvEMMA"

    def reset(self, n):
        pass

    def freeze_attention(self):
        # prevent gradient updates from changing attention weights
        for key, param in self.named_parameters():
            if "key" in key:
                param.requires_grad = False
        self.sprite_emb.weight.requires_grad = False

    def attention(self, query, key, value):
        """
        Cell by cell attention mechanism. Uses the sprite embeddings as query. Key is
        text embeddings
        """
        kq = query @ key.t()  # dot product attention
        mask = kq != 0  # keep zeroed-out entries zero
        kq = kq / self.attn_scale  # scale to prevent vanishing grads
        weights = F.softmax(kq, dim=-1) * mask
        return torch.mean(weights.unsqueeze(-1) * value, dim=-2), weights

    def batch_attention(self, query, key, value):
        """Identical to attention() but does everything with additional batch dimension"""
        bs = query.shape[0]
        kq = torch.bmm(query.view(bs, -1, self.emb_dim), key.permute(0, 2, 1))
        kq = kq / self.attn_scale
        weights = F.softmax(kq, dim=-1)
        weights = weights * (kq != 0)
        weights = weights.view(*query.shape[:-1], -1)

        return torch.mean(
            weights.unsqueeze(-1) * value.view(bs, 1, 1, 1, *value.shape[1:]), dim=-2
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, temb, memory, output_weights=False):
        """
        called when sampling actions from the policy
        """

        # split the state tensor into objects and avatar
        obj_state, avt_state = torch.split(state, [state.shape[-1] - 1, 1], dim=-1)

        # embedding for the avatar object, which will not attend to text
        avatar_emb = nonzero_mean(self.sprite_emb(avt_state))

        # take the non_zero mean of embedded objects, which will act as attention query
        query = nonzero_mean(self.sprite_emb(obj_state))

        # Attention
        key = self.txt_key(temb)
        key_scale = self.scale_key(temb)  # (num sent, sent_len, 1)
        key = key * key_scale
        key = torch.sum(key, dim=1)

        value = self.txt_val(temb)
        val_scale = self.scale_val(temb)
        value = value * val_scale
        value = torch.sum(value, dim=1)

        state_emb, weights = self.attention(query, key, value)

        if output_weights:
            return weights

        # compress the channels from KHWC to HWC' where K is history length
        state_emb = state_emb.view(self.state_h, self.state_w, -1)
        avatar_emb = avatar_emb.view(self.state_h, self.state_w, -1)

        # Take the average between state_emb and avatar_emb in case of overlap
        state_emb = (state_emb + avatar_emb) / 2.0

        # permute from HWC to NCHW and do convolution
        state_emb = state_emb.permute(2, 0, 1).unsqueeze(0)
        state_emb = F.leaky_relu(self.conv(state_emb)).view(-1)

        action_probs = self.action_layer(state_emb)

        if self.training:
            dist = Categorical(action_probs)
            action = dist.sample()

            # store everything in memory
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
            memory.texts.append(temb)
            return action.item()

        else:
            action = torch.argmax(action_probs).item()
            """
            if random.random() < 0.05:  # random action with 0.05 prob
                action = random.randrange(0, self.action_dim)
            """
            return action

    def evaluate(self, state, action, temb):
        """
        identical to self.act(), but with an additional batch dimension called by PPO algorithm when updating model params
        """
        bs = state.shape[0]  # batch size

        # split the state tensor into objects and avatar
        obj_state, avt_state = torch.split(state, [state.shape[-1] - 1, 1], dim=-1)

        # embedding for the avatar object, which will not attend to text
        avatar = nonzero_mean(self.sprite_emb(avt_state))

        # take non_zero mean of embedded state which acts as query
        query = nonzero_mean(self.sprite_emb(obj_state))

        # attention
        key = self.txt_key(temb)  # (bs, num_sent, sent_len, emb_dim)
        key_scale = self.scale_key(temb)  # (bs, num sent, sent_len, 1)
        key = key * key_scale
        key = torch.sum(
            key, dim=2
        )  # (bs, num_sent, sent_len, emb_dim) -> (bs, num_sent, emb_dim)

        value = self.txt_val(temb)
        val_scale = self.scale_val(temb)
        value = value * val_scale
        value = torch.sum(value, dim=2)

        state = self.batch_attention(query, key, value)

        # compress the channels from NKHWC to BHWC' where K is history length
        state = state.view(bs, self.state_h, self.state_w, -1)
        avatar = avatar.view(bs, self.state_h, self.state_w, -1)

        # Take the average between state_emb and avatar_emb in case of overlap
        state = (state + avatar) / 2.0

        # permute from NHWC to NCHW and do convolution
        state = state.permute(0, 3, 1, 2)

        state = F.leaky_relu(self.conv(state)).reshape(bs, -1)

        # get action probs and values
        action_probs = self.action_layer(state)

        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy
