from messenger.envs.stage_one import StageOne
from messenger.envs.stage_two import StageTwo
from messenger.envs.stage_two_custom import StageTwoCustom
from messenger.envs.stage_one_custom import StageOneCustom
from messenger.envs.stage_three import StageThree
from messenger.envs.stage_two_transformer import StageTwoTransformer
from messenger.envs.wrappers import PermuteChannelsWrapper, GridObsWrapper

import gym


def make_env(
    args,
    world_model=None,
    gpt_groundings=None,
    max_episode_steps=32,
    permute_channels=True,
    grid_obs=True,
    fix_order=False,
    env_name=None,
    env_stage=None,
):
    """Make the Messenger environment."""

    if env_name is None:
        env_name = args.env.name

    if env_stage is None:
        env_stage = args.env.stage

    if env_name == "transformer":
        env = gym.make(
            f"msgr-{env_name}-v{env_stage}",
            world_model=world_model,
            gpt_groundings=gpt_groundings,
            shuffle_obs=False,
            fix_order=fix_order,
        )
    else:
        env = gym.make(
            f"msgr-{env_name}-v{env_stage}",
            shuffle_obs=False,
            fix_order=fix_order,
        )

    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    if permute_channels:
        env = PermuteChannelsWrapper(env)

    if grid_obs:
        env = GridObsWrapper(env)

    return env
