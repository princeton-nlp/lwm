import os
import sys

sys.path.append("..")
import random
import json

from collections import defaultdict
import numpy as np
import pprint

from chatgpt_groundings.utils import load_gpt_groundings
from messenger.envs import make_env
from train_wm import make_dataset, make_model
from transformer_world_model.utils import ENTITY_IDS
import flags


ACTIONS = [(0, -1, 0), (1, 1, 0), (2, 0, -1), (3, 0, 1), (4, 0, 0)]


def get_entities(o):
    return o.reshape(-1, 4).max(0).tolist()


def get_distance(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def find_entity_coordinates(o, e=None):
    entities = get_entities(o)
    if e is None:
        e = 15 if 15 in entities else 16
    if e not in entities:
        return None, None
    c = entities.index(e)
    p = o.reshape(-1, 4)[:, c].tolist().index(e)
    x = p // 10
    y = p % 10
    return x, y


def compute_distances(observations, e):
    dists = []
    for o in observations:
        ex, ey = find_entity_coordinates(o, e)
        px, py = find_entity_coordinates(o)
        # player is dead, skip
        if px is None:
            continue
        # player is alive
        if ex is not None:
            d = get_distance(ex, ey, px, py)
        else:
            d = -1
        dists.append(d)
    return dists


def next_positon(x, y, a):
    nx = x + ACTIONS[a][1]
    ny = y + ACTIONS[a][2]
    nx = max(min(nx, 9), 0)
    ny = max(min(ny, 9), 0)
    return nx, ny


def get_role_by_id(manual, id):
    for e in manual:
        if ENTITY_IDS[e[0]] == id:
            return e[-1]


def get_id_by_role(manual, role):
    for e in manual:
        if e[-1] == role:
            return ENTITY_IDS[e[0]]


def get_movement_by_role(manual, role):
    for e in manual:
        if e[-1] == role:
            return e[1]


def will_collide(e, po, co, a, manual):
    # player position in last obs
    px, py = find_entity_coordinates(po)

    # no player, no collision
    if px is None:
        return 0

    # player current position
    cpx, cpy = next_positon(px, py, a)

    e_id = get_id_by_role(manual, e)
    ex, ey = find_entity_coordinates(po, e_id)

    # no entity, no collision
    if ex is None:
        return 0

    cex, cey = find_entity_coordinates(co, e_id)
    e_move = get_movement_by_role(manual, e)
    if e_move == "chaser":
        if get_distance(cpx, cpy, ex, ey) <= 2:
            return 1
    else:
        if get_distance(cpx, cpy, ex, ey) <= 1:
            return 1
    return 0


def is_reward_correct(po, co, a, r, manual):
    if r not in [-1, 0.5, 1]:
        return 0

    # detect collision
    entities = get_entities(po)

    # player bumps into enemy or goal (before collecting message)
    if r == -1:
        entities_to_check = ["enemy", "goal"]
    elif r == 0.5:
        entities_to_check = ["message"]
    else:
        assert r == 1, r
        entities_to_check = ["goal"]

    collided_entities = []
    for e in entities_to_check:
        if will_collide(e, po, co, a, manual):
            collided_entities.append(e)

    if not collided_entities:
        return 0

    # colliding with goal after obtaining message cannot result in a reward of -1
    if r == -1 and collided_entities == ["goal"] and 16 in entities:
        return 0

    return 1


def is_done_correct(po, co, a, manual):
    # player bumps into enemy or goal
    entities_to_check = ["enemy", "goal"]

    collided_entities = []
    for e in entities_to_check:
        if will_collide(e, po, co, a, manual):
            collided_entities.append(e)

    if not collided_entities:
        return 0

    return 1


def compute_metrics(split, episode, env):
    true_observations = episode.observations.numpy()
    actions = episode.actions.tolist()

    while True:
        obs, info = env.reset(split=split, entities=episode.true_parsed_manual)
        if (obs == true_observations[0]).all():
            break

    pred_observations = [obs]
    pred_rewards = [0]
    pred_dones = [0]

    assert actions[0] == 0
    for a in actions[1:]:
        obs, reward, done, info = env.step(a)
        pred_observations.append(obs)
        pred_rewards.append(reward)
        pred_dones.append(done)
        if done:
            break

    l = min(len(true_observations), len(pred_observations))

    entities = get_entities(true_observations[0])
    dists = []
    for e in entities[:-1]:
        true_dists = compute_distances(true_observations[:l], e)
        pred_dists = compute_distances(pred_observations[:l], e)
        role = get_role_by_id(episode.true_parsed_manual, e)
        for i, (x, y) in enumerate(zip(true_dists, pred_dists)):
            dists.append((role, i, x, y))

    reward_accs = []
    done_accs = []
    true_rewards = episode.rewards.tolist()
    true_dones = episode.dones.tolist()

    for i, (r, d) in enumerate(zip(pred_rewards, pred_dones)):
        if i == 0:
            continue
        po = pred_observations[i - 1]
        co = pred_observations[i]
        a = actions[i]
        if r != 0:
            reward_accs.append(
                is_reward_correct(po, co, a, r, episode.true_parsed_manual)
            )

        if d == 1:
            if i == 32:
                done_accs.append(1)
            else:
                done_accs.append(is_done_correct(po, co, a, episode.true_parsed_manual))

    return dists, reward_accs, done_accs


def eval(dataset, env):
    args.result_file_path = args.wm_weights_path.replace(
        ".ckpt", "_imagine_results.json"
    )
    print(f"Will save results to {args.result_file_path}")

    metrics = defaultdict(lambda: defaultdict(list))
    for split in dataset:
        if "train" in split:
            continue
        print(split)
        n_examples = len(dataset[split])
        chosen_indices = random.sample(list(range(n_examples)), 50)
        for i, episode in enumerate(dataset[split].episodes):
            if i not in chosen_indices:
                continue
            dists, reward_accs, done_accs = compute_metrics(split, episode, env)
            metrics[split]["dist"].extend(dists)
            metrics[split]["reward"].extend(reward_accs)
            metrics[split]["done"].extend(done_accs)

    with open(args.result_file_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved results to {args.result_file_path}")

    return metrics


if __name__ == "__main__":
    args = flags.make()

    # set exp dir
    if args.save_dir is None:
        assert args.exp_name is not None, "Experiment name is not provided!"
        args.save_dir = f"experiments/{args.exp_name}"

    # print arguments
    print(pprint.pformat(vars(args), indent=2))

    dataset, gpt_groundings = make_dataset(args, return_gpt_groundings=True)
    world_model = make_model(args)

    env = make_env(
        args,
        world_model=world_model,
        gpt_groundings=gpt_groundings,
        env_name="transformer",
        fix_order=True,
    )

    eval(dataset, env)
