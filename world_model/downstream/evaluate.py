import json

import torch
import numpy as np

from .rollout import RolloutGenerator


class Evaluator:
    def __init__(self, args):
        self.rollout_generator = RolloutGenerator(args)
        self.save_dir = args.save_dir
        self.eval_mode = args.eval_mode
        self.load_splits(args)

    def load_splits(self, args):
        with open(args.downstream.splits_path) as f:
            self.split_games = json.load(f)

    def _is_better(self, name, x, y):
        if y is None:
            return True
        if name == "avg_episode_len":
            return x < y
        if name == "avg_return":
            return x > y

    def _format_metric(self, x):
        if isinstance(x, int):
            return str(x)
        if isinstance(x, float):
            return f"{x:.2f}"

    def evaluate(
        self, policy, env, num_episodes, best_metric, game=None, split=None, name=None
    ):
        policy.eval()

        if game is None:
            games = self.split_games[split]
        else:
            games = [game]

        memory = self.rollout_generator.generate(
            policy, env, split, games, num_episodes
        )

        ep_lens = []
        returns = []
        steps = 0
        total_reward = 0
        for reward, done in zip(memory.rewards, memory.is_terminals):
            total_reward += reward
            steps += 1
            if done:
                ep_lens.append(steps)
                returns.append(total_reward)
                total_reward = steps = 0

        metric = {
            "avg_episode_len": np.average(ep_lens),
            "avg_return": np.average(returns),
        }

        if name is None:
            name = split

        self._log_and_save_best_model(policy, name, metric, best_metric)

        return metric

    def _log_and_save_best_model(self, policy, name, metric, best_metric):
        log_str = " , ".join(
            [f"{k:15} {self._format_metric(v):10} " for k, v in metric.items()]
        )
        print(f"      {name:20}  {log_str}")

        for k in metric:
            if self._is_better(k, metric[k], best_metric[k]):
                best_metric[k] = metric[k]
                if not self.eval_mode and k == "avg_return":
                    file_path = f"{self.save_dir}/{name}_best_{k}.ckpt"
                    torch.save(policy.state_dict(), file_path)
                    print(f"Saved best {name} {k} as {file_path}")
