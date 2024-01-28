from dataclasses import dataclass
from collections import defaultdict
import os
import sys
import math

sys.path.append("..")
import argparse
import json
import random
import pickle
import pprint
import time

import wandb
import numpy as np
import torch
import torch.nn.functional as F

from chatgpt_groundings.utils import (
    ENTITY_GROUNDING_LOOKUP,
    MOVEMENT_GROUNDING_LOOKUP,
    ROLE_GROUNDING_LOOKUP,
    load_gpt_groundings,
)

from transformer_world_model.dataset import EpisodesDataset
from transformer_world_model.utils import ENTITY_IDS
from transformer_world_model.world_model import WorldModel
from transformer_world_model.tokenizer import ObservationTokenizer
from transformer_world_model.transformer import TransformerConfig

import flags


def make_model(args):
    encoder_config = TransformerConfig(
        tokens_per_block=args.encoder_tokens_per_block,
        max_blocks=args.encoder_max_blocks,
        num_layers=args.encoder_layers,
        num_heads=args.encoder_num_heads,
        embed_dim=args.hidden_size,
        embed_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        tokenizers={"manual": ObservationTokenizer},
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
        has_external_memory=(args.manual in ["standard", "standardv2"]),
        device=args.device,
    )

    world_model = WorldModel(
        args,
        encoder_config,
        decoder_config,
    ).to(args.device)

    if args.wm_weights_path is not None:
        world_model.load_state_dict(torch.load(args.wm_weights_path))
        print(f"Loaded model from {args.wm_weights_path}")

    print(world_model)
    total_params = sum(p.numel() for p in world_model.parameters())
    print(f"Number of parameters: {total_params}")

    return world_model


def make_dataset(args, return_gpt_groundings=False):
    gpt_groundings = load_gpt_groundings(args)

    with open(args.dataset_path, "rb") as f:
        dataset_obj = pickle.load(f)

    dataset = {}
    splits = list(dataset_obj["rollouts"].keys())
    # exclude all test splits during training
    exclude_prefix = "null" if args.eval_mode else "test"
    splits = [split for split in splits if exclude_prefix not in split]
    for split in splits:
        k = "train" if "train" in split else split
        dataset[k] = EpisodesDataset(dataset_obj, split, gpt_groundings, args.seed)
        print(f"Loaded {split} split with {len(dataset[k])} episodes")

    if return_gpt_groundings:
        return dataset, gpt_groundings

    return dataset


def make_optimizer(model, learning_rate, weight_decay, *blacklist_module_names):
    """Credits to https://github.com/karpathy/minGPT"""
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            if any(
                [fpn.startswith(module_name) for module_name in blacklist_module_names]
            ):
                no_decay.add(fpn)
            elif "bias" in pn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert (
        len(param_dict.keys() - union_params) == 0
    ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer


def _to_device(batch, device):
    for k in batch:
        try:
            batch[k] = batch[k].to(device)
        except:
            pass


def _update_metric(args, metric, logits, labels, batch):
    def _cross_entropy_loss(name, logit, label):
        logit = logit.reshape(-1, logit.shape[-1])
        label = label.reshape(-1)
        out = F.cross_entropy(logit, label, reduction="none")
        out = out.masked_select((label != -100)).tolist()
        return out

    L = logits["reward"].size(1)
    logits["observation"] = logits["observation"].view(
        logits["observation"].size(0), L, -1, logits["observation"].size(-1)
    )
    labels["observation"] = labels["observation"].view(
        labels["observation"].size(0), L, -1
    )

    for k in logits:
        metric[f"{k}_loss"].extend(_cross_entropy_loss(k, logits[k], labels[k]))

    for k in logits:
        for l in range(L):
            metric[f"{k}_loss_len_{l + 1}"].extend(
                _cross_entropy_loss(k, logits[k][:, l], labels[k][:, l])
            )
            for ll in range(l + 1):
                metric[f"{k}_loss_len_upto_{l + 1}"].extend(
                    _cross_entropy_loss(k, logits[k][:, ll], labels[k][:, ll])
                )


def train(args, world_model, dataset, optimizer):
    best_eval_metric = defaultdict(lambda: defaultdict(lambda: 1e9))
    train_metric = defaultdict(list)
    for i in range(args.max_batches):
        if i % args.log_every_batches == 0:
            # logging
            wandb_stats = {}
            wandb_stats["step"] = i
            wandb_stats["lr"] = optimizer.param_groups[0]["lr"]
            log_str = []
            for k in train_metric:
                train_metric[k] = np.average(train_metric[k])
                wandb_stats[f"train/{k}"] = train_metric[k]
                log_str.append(f"{k} {train_metric[k]:.2f}")
            log_str = ", ".join(log_str)
            print()
            print("After %d batches" % i)
            print("  TRAIN", log_str)
            print()

            # reset train metrics
            train_metric = defaultdict(list)

            # evaluation
            print("  REAL EVALUATION")
            for split in dataset:
                eval_metric = eval_real(
                    args,
                    world_model,
                    dataset,
                    split,
                    max_batches=10 if "train" in split else None,
                    best_metric=best_eval_metric[split],
                )
                for k in eval_metric:
                    wandb_stats[f"{split}/real_{k}"] = eval_metric[k]
                    wandb_stats[f"{split}_best/real_{k}"] = best_eval_metric[split][k]

            wandb.log(wandb_stats)

        if args.eval_mode:
            break

        # train
        world_model.train()
        optimizer.zero_grad()
        for _ in range(args.grad_acc_steps):
            batch = dataset["train"].sample_batch(
                args.batch_size,
                args.decoder_max_blocks,
                weights=None,
                sample_from_start=True,
            )
            _to_device(batch, args.device)

            losses, _, _ = world_model.compute_loss(batch)
            losses = losses / args.grad_acc_steps
            loss_total = losses.loss_total
            loss_total.backward()

            train_metric["total"].append(loss_total.item())
            for loss_name, loss_value in losses.intermediate_losses.items():
                train_metric[loss_name].append(loss_value)

        if args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), args.max_grad_norm)

        optimizer.step()


@torch.no_grad()
def eval_real(args, world_model, dataset, split, max_batches=None, best_metric=None):
    world_model.eval()
    # compute metrics
    metric = defaultdict(list)
    total_batches = 0
    for batch in dataset[split].iterate_batches(
        args.batch_size, args.decoder_max_blocks
    ):
        total_batches += 1
        if max_batches is not None and total_batches > max_batches:
            break
        _to_device(batch, args.device)
        _, logits, labels = world_model.compute_loss(batch)
        _update_metric(args, metric, logits, labels, batch)

    # average metrics
    for k in list(metric.keys()):
        metric[k] = np.average(metric[k])
        metric[k.replace("loss", "perp")] = math.exp(metric[k])
    metric["total_loss"] = sum(
        [metric[k] for k in metric if "loss" in k and "len_" not in k]
    )

    # logging
    log_str = []
    for k in metric:
        if "loss" in k and ("len_" not in k or k.endswith("len_1")):
            log_str.append(f"{k} {metric[k]:.2f}")
    log_str = ", ".join(log_str)
    print("  *", split)
    print("     ", log_str)

    # update best models
    if args.eval_mode == 0 and best_metric is not None:
        for k in metric:
            if metric[k] < best_metric[k]:
                best_metric[k] = metric[k]
                if "loss" in k and "len_" not in k:
                    model_path = f"{args.save_dir}/{split}_best_{k}.ckpt"
                    torch.save(world_model.state_dict(), model_path)
                    print(f"Save best {split} {k} to {model_path}")
    print()

    return metric


if __name__ == "__main__":
    args = flags.make()

    # set exp dir
    if args.save_dir is None:
        assert args.exp_name is not None, "Experiment name is not provided!"
        args.save_dir = f"experiments/{args.exp_name}"

    # make exp dir of not exist
    if args.eval_mode == 0:
        if os.path.exists(args.save_dir):
            print(f"Output folder {args.save_dir} exists")
            sys.exit(1)
        os.makedirs(args.save_dir)

    # print arguments
    print(pprint.pformat(vars(args), indent=2))

    # start wandb logging
    wandb.init(
        project="lwm-messenger-transformer",
        entity=args.wandb_entity,
        group=f"{args.version}_{args.manual}_{args.hidden_size}_{args.encoder_layers}_{args.decoder_layers}",
        name=f"{args.exp_name}_{str(int(time.time()))}",
        mode=args.mode if args.use_wandb else "disabled",
    )
    wandb.config.update(args)

    dataset = make_dataset(args)
    world_model = make_model(args)
    optimizer = make_optimizer(world_model, args.learning_rate, args.weight_decay)
    train(args, world_model, dataset, optimizer)
