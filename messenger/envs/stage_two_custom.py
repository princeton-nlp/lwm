"""
Classes that follows a gym-like interface and implements stage two of the Messenger
environment. Uses custom assignments of entity-dynamic-role.
"""

import random
import json
import os
from pathlib import Path
from os import environ

# hack to stop PyGame from printing to stdout
environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from vgdl.interfaces.gym import VGDLEnv
import numpy as np
import torch

from messenger.envs.base import MessengerEnv, Grid, Position
import messenger.envs.config as config
from messenger.envs.utils import get_game

GAME_FILE_TEMPLATE = """
BasicGame block_size=2
	SpriteSet
		background > Immovable randomtiling=0.9 img=oryx/floor3 hidden=True
		root >
			enemy > %s stype=avatar speed=0.5 img=oryx/alien1
			message > %s stype=avatar speed=0.5 img=oryx/bear1
			goal > %s stype=avatar speed=0.5 img=oryx/cyclop1
			wall > Immovable img=oryx/wall11
			avatar > MovingAvatar
				no_message > img=oryx/swordman1_0
				with_message > img=oryx/swordmankey1_0
	InteractionSet
		root wall > stepBack
		root EOS > stepBack
		avatar enemy > killSprite scoreChange=-1
		no_message goal > killSprite scoreChange=-1
		no_message message > transformTo stype=with_message scoreChange=0.5
		message avatar > killSprite
		goal with_message > killSprite scoreChange=1
	TerminationSet
		SpriteCounter stype=avatar limit=0 win=False
		SpriteCounter stype=goal limit=0 win=True
	LevelMapping
		. > background
		E > background enemy
		M > background message
		G > background goal
		X > background no_message
		Y > background with_message
		W > background wall
"""


class StageTwoCustom(MessengerEnv):
    """
    Full messenger environment with mobile sprites. Uses Py-VGDL as game engine.
    To avoid the need to instantiate a large number of games, (since there are
    P(12,3) = 1320 possible entity to role assignments) We apply a wrapper on top
    of the text and game state which masks the role archetypes (enemy, message goal)
    into entities (e.g. alien, knight, mage).
    """

    def __init__(self, shuffle_obs: bool = False, fix_order: bool = False):
        super().__init__()
        self.shuffle_obs = shuffle_obs  # shuffle the entity layers
        self.this_folder = Path(__file__).parent

        with open(
            self.this_folder.joinpath(
                "texts",
                "custom_text_splits",
                "custom_text_splits_with_messenger_names.json",
            ),
            "r",
        ) as f:
            self.custom_text_splits = json.load(f)

        self.init_states = [
            str(path)
            for path in self.this_folder.joinpath(
                "vgdl_files", "stage_2", "init_states"
            ).glob("*.txt")
        ]

        # entities tracked by VGDLEnv
        self.notable_sprites = [
            "enemy",
            "message",
            "goal",
            "no_message",
            "with_message",
        ]
        self.env = None  # the VGDLEnv
        self.manual = None  # the text manual
        self.fix_order = fix_order
        self.next_init_state_idx = 0

    def _convert_obs(self, vgdl_obs):
        """
        Return a grid built from the vgdl observation which is a
        KeyValueObservation object (see vgdl code for details).
        """
        entity_locs = Grid(layers=3, shuffle=self.shuffle_obs)
        avatar_locs = Grid(layers=1)

        # try to add each entity one by one, if it's not there move on.
        if "enemy.1" in vgdl_obs:
            entity_locs.add(self.game.enemy, Position(*vgdl_obs["enemy.1"]["position"]))
        if "message.1" in vgdl_obs:
            entity_locs.add(
                self.game.message, Position(*vgdl_obs["message.1"]["position"])
            )
        else:
            # advance the entity counter, Oracle model requires special order.
            # TODO: maybe used named layers to make this more understandable.
            entity_locs.entity_count += 1
        if "goal.1" in vgdl_obs:
            entity_locs.add(self.game.goal, Position(*vgdl_obs["goal.1"]["position"]))

        if "no_message.1" in vgdl_obs:
            """
            Due to a quirk in VGDL, the avatar is no_message if it starts as no_message
            even if the avatar may have acquired the message at a later point.
            To check, if it has a message, check that the class vector corresponding to
            with_message is == 1.
            """
            avatar_pos = Position(*vgdl_obs["no_message.1"]["position"])
            # with_key is last position, see self.notable_sprites
            if vgdl_obs["no_message.1"]["class"][-1] == 1:
                avatar = config.WITH_MESSAGE
            else:
                avatar = config.NO_MESSAGE

        elif "with_message.1" in vgdl_obs:
            # this case only occurs if avatar begins as with_message at start of episode
            avatar_pos = Position(*vgdl_obs["with_message.1"]["position"])
            avatar = config.WITH_MESSAGE

        else:  # the avatar is not in observation, so is probably dead
            return {"entities": entity_locs.grid, "avatar": avatar_locs.grid}

        avatar_locs.add(avatar, avatar_pos)  # if not dead, add it.

        return {"entities": entity_locs.grid, "avatar": avatar_locs.grid}

    def reset(self, split, entities, freeze_manual: bool = False, **kwargs):
        """
        Resets the current environment. NOTE: We remake the environment each time.
        This is a workaround to a bug in py-vgdl, where env.reset() does not
        properly reset the environment. kwargs go to get_document().

        split is one of the split keywords in custom splits
        entities is one of the games in a custom split
        """

        entities_by_role = {
            "enemy": None,
            "message": None,
            "goal": None,
        }
        for entity in entities:
            entities_by_role[entity[2]] = [entity[0], entity[1]]
        if None in entities_by_role.values():
            raise RuntimeError
        self.game = get_game(
            (
                entities_by_role["enemy"][0],
                entities_by_role["message"][0],
                entities_by_role["goal"][0],
            )
        )

        game_file_path = self.this_folder.joinpath(
            f"custom_game_file_{os.getpid()}.txt"
        )
        with open(game_file_path, "w") as f:
            f.write(
                GAME_FILE_TEMPLATE
                % (
                    entities_by_role["enemy"][1].title(),
                    entities_by_role["message"][1].title(),
                    entities_by_role["goal"][1].title(),
                )
            )

        if not self.fix_order:
            self.next_init_state_idx = random.randrange(len(self.init_states))
        init_state = self.init_states[self.next_init_state_idx]  # inital state file

        # args that will go into VGDL Env.
        self._envargs = {
            "game_file": game_file_path,
            "level_file": init_state,
            "notable_sprites": self.notable_sprites.copy(),
            "obs_type": "objects",  # track the objects
            "block_size": 34,  # rendering block size
        }
        self.env = VGDLEnv(**self._envargs)
        vgdl_obs = self.env.reset()

        try:
            os.remove(game_file_path)
        except:
            pass

        if not freeze_manual or self.manual is None:
            manual = [
                random.choice(
                    self.custom_text_splits[entity[0]][entity[1]][entity[2]][split]
                )
                for entity in entities
            ]
            self.manual = manual
        else:
            manual = self.manual

        true_parsed_manual = [(entity[0], entity[1], entity[2]) for entity in entities]

        obs = self._convert_obs(vgdl_obs)

        merged_obs = np.concatenate((obs["entities"], obs["avatar"]), axis=-1)
        # self.cur_grid = torch.from_numpy(merged_obs).unsqueeze(0)

        info = {"raw_manual": manual, "true_parsed_manual": true_parsed_manual}

        if self.fix_order:
            self.next_init_state_idx += 1
            if self.next_init_state_idx >= len(self.init_states):
                self.next_init_state_idx = 0

        return obs, info

    def step(self, action):
        vgdl_obs, reward, done, info = self.env.step(action)
        obs = self._convert_obs(vgdl_obs)

        merged_obs = np.concatenate((obs["entities"], obs["avatar"]), axis=-1)
        # self.cur_grid = torch.from_numpy(merged_obs).unsqueeze(0)

        return obs, reward, done, info
