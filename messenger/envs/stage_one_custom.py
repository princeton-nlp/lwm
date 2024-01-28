"""
Classes that follows a gym-like interface and implements stage one of the Messenger
environment.
"""

import json
import random
from collections import namedtuple
from pathlib import Path

import numpy as np

from messenger.envs.base import MessengerEnv, Position
import messenger.envs.config as config
from messenger.envs.manual import TextManual
from messenger.envs.utils import get_game


# Used to track sprites in StageOne, where we do not use VGDL to handle sprites.
Sprite = namedtuple("Sprite", ["name", "id", "position"])

ENTITY_KEYWORD_TABLE = {
    "robot": "robot",
    "airplane": "airplane",
    "thief": "thief",
    "scientist": "scientist",
    "queen": "queen",
    "ship": "ship",
    "dog": "dog",
    "bird": "bird",
    "fish": "fish",
    "mage": "mage",
    "orb": "ball",
    "sword": "sword",
}


class StageOneCustom(MessengerEnv):
    def __init__(self, message_prob=0.2, shuffle_obs=True):
        """
        Stage one where objects are all immovable. Since the episode length is short and entities
        do not move, we do not use VGDL engine for efficiency.
        message_prob:
            the probability that the avatar starts with the message
        shuffle_obs:
            shuffle the observation including the text manual
        """
        super().__init__()
        self.message_prob = message_prob
        self.shuffle_obs = shuffle_obs
        this_folder = Path(__file__).parent

        with open(
            this_folder.joinpath(
                "texts", "custom_text_splits", "custom_text_splits.json"
            ),
            "r",
        ) as f:
            self.custom_text_splits = json.load(f)

        self.positions = [  # all possible entity locations
            Position(y=3, x=5),
            Position(y=5, x=3),
            Position(y=5, x=7),
            Position(y=7, x=5),
        ]
        self.avatar_start_pos = Position(y=5, x=5)
        self.avatar = None
        self.enemy = None
        self.message = None
        self.neutral = None
        self.goal = None

        self.manual = None

    def _get_obs(self):
        entities = np.zeros((config.STATE_HEIGHT, config.STATE_WIDTH, 1))
        avatar = np.zeros((config.STATE_HEIGHT, config.STATE_WIDTH, 1))
        for sprite in (self.enemy, self.message, self.goal):
            entities[sprite.position.y, sprite.position.x, 0] = sprite.id

        avatar[self.avatar.position.y, self.avatar.position.x, 0] = self.avatar.id

        return {"entities": entities, "avatar": avatar}

    def reset(self, split, entities, freeze_manual: bool = False):
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
                ENTITY_KEYWORD_TABLE[entities_by_role["enemy"][0]],
                ENTITY_KEYWORD_TABLE[entities_by_role["message"][0]],
                ENTITY_KEYWORD_TABLE[entities_by_role["goal"][0]],
            )
        )
        enemy, message, goal = self.game.enemy, self.game.message, self.game.goal

        # randomly choose where to put enemy, key, goal
        shuffled_pos = random.sample(self.positions, 4)
        self.enemy = Sprite(name=enemy.name, id=enemy.id, position=shuffled_pos[0])
        self.message = Sprite(
            name=message.name, id=message.id, position=shuffled_pos[1]
        )
        self.goal = Sprite(name=goal.name, id=goal.id, position=shuffled_pos[2])

        if random.random() < self.message_prob:
            self.avatar = Sprite(
                name=config.WITH_MESSAGE.name,
                id=config.WITH_MESSAGE.id,
                position=self.avatar_start_pos,
            )

        else:  # decide whether avatar has message or not
            self.avatar = Sprite(
                name=config.NO_MESSAGE.name,
                id=config.NO_MESSAGE.id,
                position=self.avatar_start_pos,
            )

        obs = self._get_obs()

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

        ground_truth = [(entity[0], entity[1], entity[2]) for entity in entities]

        return obs, manual, ground_truth

    def _move_avatar(self, action):
        if action == config.ACTIONS.stay:
            return

        elif action == config.ACTIONS.up:
            if self.avatar.position.y <= 0:
                return
            else:
                new_position = Position(
                    y=self.avatar.position.y - 1, x=self.avatar.position.x
                )

        elif action == config.ACTIONS.down:
            if self.avatar.position.y >= config.STATE_HEIGHT - 1:
                return
            else:
                new_position = Position(
                    y=self.avatar.position.y + 1, x=self.avatar.position.x
                )

        elif action == config.ACTIONS.left:
            if self.avatar.position.x <= 0:
                return
            else:
                new_position = Position(
                    y=self.avatar.position.y, x=self.avatar.position.x - 1
                )

        elif action == config.ACTIONS.right:
            if self.avatar.position.x >= config.STATE_WIDTH - 1:
                return
            else:
                new_position = Position(
                    y=self.avatar.position.y, x=self.avatar.position.x + 1
                )

        else:
            raise Exception(f"{action} is not a valid action.")

        self.avatar = Sprite(
            name=self.avatar.name, id=self.avatar.id, position=new_position
        )

    def _overlap(self, sprite_1, sprite_2):
        if (
            sprite_1.position.x == sprite_2.position.x
            and sprite_1.position.y == sprite_2.position.y
        ):
            return True
        else:
            return False

    def step(self, action):
        self._move_avatar(action)
        obs = self._get_obs()
        if self._overlap(self.avatar, self.enemy):
            return obs, -1.0, True, None  # state, reward, done, info

        if self._overlap(self.avatar, self.message):
            if self.avatar.name == config.WITH_MESSAGE.name:
                return obs, -1.0, True, None
            elif self.avatar.name == config.NO_MESSAGE.name:
                return obs, 1.0, True, None
            else:
                raise Exception("Unknown avatar name {avatar.name}")

        if self._overlap(self.avatar, self.goal):
            if self.avatar.name == config.WITH_MESSAGE.name:
                return obs, 1.0, True, None
            elif self.avatar.name == config.NO_MESSAGE.name:
                return obs, -1.0, True, None
            else:
                raise Exception("Unknown avatar name {avatar.name}")

        return obs, 0.0, False, {}
