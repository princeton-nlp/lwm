import os
import sys

sys.path.append("..")
import pprint
import wandb
import time

from collections import defaultdict

import numpy as np
import torch

torch.backends.cudnn.deterministic = True

from downstream import ImitationTrainer, Evaluator, ConvEMMA, TransformerEMMA
from messenger.envs import make_env
from train_wm import make_dataset
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
    dataset,
    policy,
    train_env,
    eval_env,
    optimizer,
    eval_episodes=24,
):
    best_eval_metric = defaultdict(lambda: defaultdict(lambda: None))
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

            # reset train metric
            train_metric = defaultdict(list)

            print("  EVALUATION")
            for split in dataset:
                eval_metric = evaluator.evaluate(
                    policy,
                    eval_env,
                    eval_episodes,
                    best_eval_metric[split],
                    split=split,
                )
                for k in eval_metric:
                    wandb_stats[f"{split}/{k}"] = eval_metric[k]
                    wandb_stats[f"{split}_best/{k}"] = best_eval_metric[split][k]

            wandb.log(wandb_stats)

        if args.eval_mode:
            break

        # save current policy
        if args.save_every_batches is not None and i % args.save_every_batches == 0:
            file_path = f"{args.save_dir}/policy_{i}.ckpt"
            torch.save(policy.state_dict(), file_path)
            print(f"Saved current policy as {file_path}")

        train_last_metric = trainer.train(policy, train_env, optimizer, dataset=dataset)
        for k in train_last_metric:
            train_metric[k].append(train_last_metric[k])


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
        group=f"downstream",
        name=f"{args.exp_name}_{str(int(time.time()))}",
        mode=args.mode if args.use_wandb else "disabled",
    )
    wandb.config.update(args)

    if args.wm_weights_path is not None:
        world_model = make_model(args)
    else:
        world_model = None

    dataset = make_dataset(args)
    train_env = make_env(args, world_model=world_model, fix_order=False)
    eval_env = make_env(args, world_model=world_model, fix_order=True)

    policy = make_policy(args)
    print(policy)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)

    trainer = ImitationTrainer(args)
    evaluator = Evaluator(args)

    train(args, trainer, evaluator, dataset, policy, train_env, eval_env, optimizer)
