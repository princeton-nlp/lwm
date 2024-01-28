from typing import List, Dict
import json

ENTITY_GROUNDING_LOOKUP = {
    2: "airplane",
    3: "mage",
    4: "dog",
    5: "bird",
    6: "fish",
    7: "scientist",
    8: "thief",
    9: "ship",
    10: "ball",
    11: "robot",
    12: "queen",
    13: "sword",
    -1: "unknown",
}
MOVEMENT_GROUNDING_LOOKUP = {
    0: "chaser",
    1: "fleeing",
    2: "immovable",
    -1: "unknown",
}
ROLE_GROUNDING_LOOKUP = {
    0: "enemy",
    1: "message",
    2: "goal",
    -1: "unknown",
}


def load_gpt_groundings(args):
    with open(args.gpt_groundings_path, "r") as f:
        gpt_groundings = json.load(f)
        # convert groundings into keywords
        for e, grounding in gpt_groundings.items():
            gpt_groundings[e] = [
                ENTITY_GROUNDING_LOOKUP[grounding[0]],
                MOVEMENT_GROUNDING_LOOKUP[grounding[1]],
                ROLE_GROUNDING_LOOKUP[grounding[2]],
            ]

    return gpt_groundings


def parse_manuals(manuals: List[List[str]], gpt_groundings: Dict):
    return [[gpt_groundings[e] for e in manual] for manual in manuals]
