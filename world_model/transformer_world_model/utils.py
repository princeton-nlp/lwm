from messenger.envs.config import NPCS, NO_MESSAGE, WITH_MESSAGE


ENTITY_IDS = {entity.name: entity.id for entity in NPCS}
MOVEMENT_IDS = {
    "chaser": 0,
    "fleeing": 1,
    "immovable": 2,
}
ROLE_IDS = {
    "enemy": 0,
    "message": 1,
    "goal": 2,
}


def process_parsed_manual_for_encoding(manual):
    out = []
    for e in manual:
        out.append([f"id_{ENTITY_IDS[e[0]]}", f"movement_{e[1]}", f"role_{e[2]}"])
    return out
