import os
import sys

sys.path.append("..")
import pprint
import wandb
import time
import json

from collections import defaultdict

import numpy as np
import torch

torch.backends.cudnn.deterministic = True

from downstream import (
    ImitationTrainer,
    FilteredBCTrainer,
    SelfImitationTrainer,
    Evaluator,
    ConvEMMA,
    TransformerEMMA,
)
from messenger.envs import make_env
from train_wm import make_dataset, make_model
import flags


def make_policy(args):
    if args.emma_policy.base_arch == "conv":
        policy = ConvEMMA().to(args.device)
    elif args.emma_policy.base_arch == "transformer":
        policy = TransformerEMMA(args).to(args.device)
        assert args.emma_policy.hist_len == 1
    else:
        print("Architecture not supported!")
        sys.exit(1)

    if args.emma_policy.weights_path is not None:
        policy.load_state_dict(torch.load(args.emma_policy.weights_path))
        print(f"Loaded model from {args.emma_policy.weights_path}")

    return policy


def train(
    args,
    trainer,
    evaluator,
    policy,
    train_env,
    sim_eval_env,
    real_eval_env,
    optimizer,
    game,
    split,
    eval_episodes=48,
):
    best_eval_metric = defaultdict(lambda: None)
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
            print("  ", split, game)
            print("  TRAIN", log_str)
            print()

            # reset train metric
            train_metric = defaultdict(list)

            print("  EVALUATION")
            for name in ["sim", "real"]:
                eval_env = sim_eval_env if name == "sim" else real_eval_env
                eval_metric = evaluator.evaluate(
                    policy,
                    eval_env,
                    eval_episodes,
                    best_eval_metric,
                    game,
                    split,
                    name=name,
                )
                for k in eval_metric:
                    wandb_stats[f"eval/{name}_{k}"] = eval_metric[k]
                    wandb_stats[f"eval_best/{name}_{k}"] = best_eval_metric[k]

            wandb.log(wandb_stats)

        if args.eval_mode:
            break

        # save current policy
        if args.save_every_batches is not None and i % args.save_every_batches == 0:
            file_path = f"{args.save_dir}/policy_{i}.ckpt"
            torch.save(policy.state_dict(), file_path)
            print(f"Saved current policy as {file_path}")

        train_last_metric = trainer.train(
            policy, train_env, optimizer, game=game, split=split
        )
        for k in train_last_metric:
            train_metric[k].append(train_last_metric[k])

    return best_eval_metric


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
        group=f"{args.version}",
        name=f"{args.version}_{args.exp_name}_{str(int(time.time()))}",
        mode=args.mode if args.use_wandb else "disabled",
    )
    wandb.config.update(args)

    with open(args.downstream.splits_path) as f:
        split_games = json.load(f)

    if args.wm_weights_path is not None:
        _, gpt_groundings = make_dataset(args, return_gpt_groundings=True)
        world_model = make_model(args)
        train_env = make_env(
            args,
            world_model=world_model,
            gpt_groundings=gpt_groundings,
            env_name="transformer",
        )
        sim_eval_env = make_env(
            args,
            world_model=world_model,
            gpt_groundings=gpt_groundings,
            env_name="transformer",
        )
    else:
        train_env = make_env(args, env_name="custom")
        sim_eval_env = make_env(args, env_name="custom")
    real_eval_env = make_env(args, fix_order=True, env_name="custom")

    policy = make_policy(args)
    print(policy)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)

    if args.downstream.task == "imitation":
        trainer = ImitationTrainer(args)
    elif args.downstream.task == "filtered_bc":
        trainer = FilteredBCTrainer(args)
    elif args.downstream.task == "self_imitation":
        trainer = SelfImitationTrainer(args)

    evaluator = Evaluator(args)

    split_map = {
        "easy": "test_ne_sr_and_sm",
        "medium": "test_se_nr_or_nm",
        "hard": "test_ne_nr_or_nm",
    }

    split = split_map[args.downstream.fix_split]
    game = split_games[split][args.downstream.fix_game]

    train(
        args,
        trainer,
        evaluator,
        policy,
        train_env,
        sim_eval_env,
        real_eval_env,
        optimizer,
        game,
        split,
    )
